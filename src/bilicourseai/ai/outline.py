from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from bilicourseai.ai.common import find_report_part
from bilicourseai.json_utils import best_effort_json_value as _best_effort_json_value
from bilicourseai.llm_client import create_client as _client, extra_body as _extra_body
from bilicourseai.models import KnowledgeBlock, VideoReport
from bilicourseai.outline.boundaries import apply_boundary_adjustments
from bilicourseai.outline.normalization import normalize_outline_nodes, quality_fields
from bilicourseai.outline.part_tree import bind_block_to_part, find_part_outline_node
from bilicourseai.outline.quality import apply_outline_quality_gate, merge_short_adjacent_nodes
from bilicourseai.outline.prompts import (
    DIRECT_PART_OUTLINE_MAX_CHARS,
    boundary_review_prompt,
    outline_whole_part_prompt,
    outline_window_prompt,
    plan_part_outline_prompt,
    reduce_part_outline_prompt,
    root_node_count_guidance,
)
from bilicourseai.outline.windows import (
    DIRECT_PART_OUTLINE_SECONDS,
    SHORT_PART_AS_LEAF_SECONDS,
    part_duration,
    root_windows,
)
from bilicourseai.settings import LLMSettings
from bilicourseai.transcripts.transcript import lines_for_range, transcript_char_count


async def outline_report(
    report: VideoReport,
    settings: LLMSettings,
    outline_window_seconds: int = 720,
    outline_overlap_seconds: int = 75,
    short_part_as_leaf_seconds: int = SHORT_PART_AS_LEAF_SECONDS,
    part_page: int | None = None,
    max_windows: int = 0,
    request_delay: float = 0.0,
    outline_concurrency: int = 1,
    progress: Callable[[str], None] | None = None,
) -> None:
    def emit(message: str) -> None:
        if progress:
            progress(message)

    if not settings.text_model:
        raise ValueError("LLM text model is required.")

    client = _client(settings)
    emit("Building internal outline windows...")
    windows = root_windows(
        report,
        outline_window_seconds=outline_window_seconds,
        outline_overlap_seconds=outline_overlap_seconds,
        short_part_as_leaf_seconds=short_part_as_leaf_seconds,
    )
    if part_page is not None:
        windows = [window for window in windows if int(window["part_page"]) == part_page]
    if max_windows > 0:
        windows = windows[:max_windows]
    emit(f"Outline windows: {len(windows)}")

    windows_by_part: dict[int, list[dict[str, Any]]] = {}
    for window in windows:
        windows_by_part.setdefault(int(window["part_page"]), []).append(window)

    for page, part_windows in windows_by_part.items():
        part = find_report_part(report, page)
        if not part:
            continue
        emit(f"P{page}: outline windows={len(part_windows)}")
        part.blocks = []
        if len(part_windows) == 1 and part_windows[0].get("leaf_ready"):
            window = part_windows[0]
            root_start = float(window["start"])
            root_end = float(window["end"])
            root_lines = lines_for_range(part.transcript, root_start, root_end)
            root = KnowledgeBlock(
                id=f"p{part.page}-o1",
                title=str(window.get("root_title") or part.title).strip(),
                start=root_start,
                end=root_end,
                summary="这个分 P 较短，适合直接生成完整学习笔记。",
                node_type="leaf",
                status="skeleton",
                expandable=True,
                depth=0,
                source_part_page=part.page,
                transcript=root_lines,
            )
            part.blocks.append(root)
            synced = _sync_part_tree_node(report, part)
            emit(f"P{page}: short part marked as leaf")
            if synced:
                emit(f"P{page}: synced outline into part tree")
            emit(f"P{page}: root nodes=1")
            continue

        part_plan: dict[str, Any] | None = None
        if (
            max_windows == 0
            and part_duration(part) <= DIRECT_PART_OUTLINE_SECONDS
            and transcript_char_count(part.transcript) <= DIRECT_PART_OUTLINE_MAX_CHARS
        ):
            guidance = root_node_count_guidance(part)
            emit(
                f"P{page}: direct part outline "
                f"(target root nodes={guidance['suggested']}, duration={part_duration(part):.1f}s)"
            )
            reduced = await _outline_whole_part(
                client,
                settings,
                report,
                part,
                guidance,
            )
            if request_delay > 0:
                await asyncio.sleep(request_delay)
        else:
            if len(part_windows) > 1:
                emit(f"P{page}: global part plan")
                part_plan = await _plan_part_outline(client, settings, report, part)
                if request_delay > 0:
                    await asyncio.sleep(request_delay)

            guidance = root_node_count_guidance(part, part_plan=part_plan)
            async def outline_one_window(window_index: int, window: dict[str, Any]) -> list[dict[str, Any]]:
                emit(
                    f"P{page}: LLM outline window {window_index}/{len(part_windows)} "
                    f"({float(window['start']):.1f}s-{float(window['end']):.1f}s)"
                )
                payload = await _outline_window(
                    client,
                    settings,
                    report,
                    window,
                    part_plan=part_plan,
                )
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
                window_candidates: list[dict[str, Any]] = []
                for item in payload.get("candidates", payload.get("children", [])):
                    if not isinstance(item, dict):
                        continue
                    item["source_window_id"] = f"p{page}-w{window_index}"
                    item["source_window_start"] = window["start"]
                    item["source_window_end"] = window["end"]
                    window_candidates.append(item)
                return window_candidates

            candidates: list[dict[str, Any]] = []
            if outline_concurrency <= 1 or len(part_windows) <= 1:
                for window_index, window in enumerate(part_windows, start=1):
                    candidates.extend(await outline_one_window(window_index, window))
            else:
                semaphore = asyncio.Semaphore(max(1, outline_concurrency))

                async def guarded_outline_one_window(window_index: int, window: dict[str, Any]) -> list[dict[str, Any]]:
                    async with semaphore:
                        return await outline_one_window(window_index, window)

                results = await asyncio.gather(
                    *(
                        guarded_outline_one_window(window_index, window)
                        for window_index, window in enumerate(part_windows, start=1)
                    )
                )
                for window_candidates in results:
                    candidates.extend(window_candidates)

            emit(
                f"P{page}: reducing {len(candidates)} candidates "
                f"(target root nodes={guidance['suggested']})"
            )
            reduced = await _reduce_part_outline(
                client,
                settings,
                report,
                part,
                candidates,
                part_plan=part_plan,
                guidance=guidance,
            )
            if request_delay > 0:
                await asyncio.sleep(request_delay)

        block_index = 1
        part_start = part.transcript[0].start
        part_end = part.transcript[-1].end
        for item in normalize_outline_nodes(reduced.get("nodes", []), part_start, part_end):
            start = float(item["start"])
            end = float(item["end"])
            quality = quality_fields(item, end - start)
            block = KnowledgeBlock(
                id=f"p{part.page}-n{block_index}",
                title=str(item.get("title") or f"知识节点 {block_index}").strip(),
                start=start,
                end=end,
                summary=str(item.get("summary") or "").strip(),
                node_type=str(item.get("node_type") or "branch").strip() or "branch",
                status="skeleton",
                expandable=bool(item.get("expandable", True)),
                depth=0,
                source_part_page=part.page,
                transcript=lines_for_range(part.transcript, start, end),
                **quality,
            )
            apply_outline_quality_gate(block)
            part.blocks.append(block)
            block_index += 1

        merged_short = merge_short_adjacent_nodes(part.blocks)
        if merged_short:
            emit(f"P{page}: merged short ordinary nodes={merged_short}")

        if len(part.blocks) > 1:
            emit(f"P{page}: reviewing {len(part.blocks) - 1} boundaries")
            boundary_payload = await _review_part_boundaries(client, settings, report, part)
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            changed = apply_boundary_adjustments(part, boundary_payload.get("boundaries", []))
            if changed:
                for block in part.blocks:
                    apply_outline_quality_gate(block)
                emit(f"P{page}: adjusted boundaries={changed}")
        generated_count = len(part.blocks)
        synced = _sync_part_tree_node(report, part)
        if synced:
            emit(f"P{page}: synced outline into part tree")
        emit(f"P{page}: root nodes={generated_count}")

    report.llm_notes.append(
        f"Generated outline with {sum(len(part.blocks) for part in report.parts)} root nodes."
    )


def _sync_part_tree_node(report: VideoReport, part) -> bool:
    part_node = find_part_outline_node(report, part.page)
    if part_node is None:
        return False
    source_blocks = part.blocks
    if not source_blocks:
        return False
    part_node.source_part_page = part.page
    for block in source_blocks:
        bind_block_to_part(block, part.page)
    if len(source_blocks) == 1:
        replacement = source_blocks[0]
        part_node.title = replacement.title
        part_node.start = replacement.start
        part_node.end = replacement.end
        part_node.summary = replacement.summary
        part_node.node_type = replacement.node_type
        part_node.status = replacement.status
        part_node.expandable = replacement.expandable
        part_node.granularity = replacement.granularity
        part_node.should_expand = replacement.should_expand
        part_node.expand_reason = replacement.expand_reason
        part_node.boundary_confidence = replacement.boundary_confidence
        part_node.split_hints = replacement.split_hints
        part_node.key_points = replacement.key_points
        part_node.transcript = replacement.transcript
        part_node.children = replacement.children
        part_node.sections = replacement.sections
        part_node.visual_requests = replacement.visual_requests
        part_node.frames = replacement.frames
        part_node.visual_analyses = replacement.visual_analyses
    else:
        part_node.node_type = "branch"
        part_node.status = "expanded"
        part_node.expandable = False
        part_node.should_expand = True
        part_node.children = source_blocks
        part_node.sections = []
        part_node.visual_requests = []
        part_node.frames = []
        part_node.visual_analyses = []
    part.blocks = []
    return True


def _payload_object(value: Any, list_key: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {list_key: value}
    return {}


async def _outline_window(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    window: dict[str, Any],
    part_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = outline_window_prompt(report, window, part_plan=part_plan)
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You create navigable study outlines for course videos.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    return _payload_object(_best_effort_json_value(response.choices[0].message.content or "{}"), "candidates")


async def _plan_part_outline(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part,
) -> dict[str, Any]:
    prompt = plan_part_outline_prompt(report, part)
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You make global structure maps for course videos before detailed outlining.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.15,
        extra_body=_extra_body(settings),
    )
    return _payload_object(_best_effort_json_value(response.choices[0].message.content or "{}"), "plan")


async def _outline_whole_part(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part,
    guidance: dict[str, int],
) -> dict[str, Any]:
    prompt = outline_whole_part_prompt(report, part, guidance)
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You create clean learner-facing knowledge trees for course videos.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    return _payload_object(_best_effort_json_value(response.choices[0].message.content or "{}"), "nodes")


async def _review_part_boundaries(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part,
) -> dict[str, Any]:
    prompt = boundary_review_prompt(report, part)
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict reviewer for course-video topic boundaries.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.1,
        extra_body=_extra_body(settings),
    )
    return _payload_object(_best_effort_json_value(response.choices[0].message.content or "{}"), "boundaries")


async def _reduce_part_outline(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part,
    candidates: list[dict[str, Any]],
    part_plan: dict[str, Any] | None = None,
    guidance: dict[str, int] | None = None,
) -> dict[str, Any]:
    if guidance is None:
        guidance = root_node_count_guidance(part, part_plan=part_plan)
    prompt = reduce_part_outline_prompt(
        report,
        part,
        candidates,
        part_plan=part_plan,
        guidance=guidance,
    )
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You organize course-video topics into a clean learner-facing knowledge tree.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    return _payload_object(_best_effort_json_value(response.choices[0].message.content or "{}"), "nodes")
