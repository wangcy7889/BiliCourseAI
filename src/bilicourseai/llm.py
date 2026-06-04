from __future__ import annotations

import base64
import asyncio
import json
import mimetypes
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from openai import OpenAIError

from bilicourseai.boundary_review import (
    apply_boundary_adjustments,
    apply_boundary_adjustments_to_blocks,
    boundary_review_payload,
)
from bilicourseai.models import FrameArtifact, KnowledgeBlock, NoteSection, TranscriptLine, VideoReport, VisualAnalysis, VisualRequest
from bilicourseai.node_quality import (
    MAX_LEAF_SECONDS,
    SHORT_BLOCK_AS_LEAF_SECONDS,
    STRONGLY_EXPAND_SECONDS,
    apply_outline_quality_gate,
)
from bilicourseai.outline_prompts import (
    DIRECT_PART_OUTLINE_MAX_CHARS,
    boundary_review_prompt,
    child_boundary_review_prompt,
    outline_whole_part_prompt,
    outline_window_prompt,
    plan_part_outline_prompt,
    reduce_part_outline_prompt,
    root_node_count_guidance,
)
from bilicourseai.settings import LLMSettings
from bilicourseai.tree import find_block
from bilicourseai.transcript_utils import (
    lines_for_range,
    transcript_char_count,
)


def _client(settings: LLMSettings) -> AsyncOpenAI:
    if not settings.base_url or not settings.api_key:
        raise ValueError("LLM requires base_url and api_key.")
    return AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)


def _extra_body(settings: LLMSettings) -> dict[str, Any]:
    return {"enable_thinking": settings.enable_thinking}


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _json_object_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _salvage_json_object(text: str) -> dict[str, Any]:
    json_text = _json_object_text(text)
    try:
        return json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        pass

    json_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_text)
    json_text = _escape_invalid_json_string_backslashes(json_text)
    try:
        return json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        return {}


def _escape_invalid_json_string_backslashes(text: str) -> str:
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            output.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char in valid_escapes:
                output.append(char)
                output.append(next_char)
                index += 2
            else:
                output.append("\\\\")
                index += 1
            continue

        output.append(char)
        index += 1
    return "".join(output)


def _best_effort_json_object(text: str) -> dict[str, Any]:
    try:
        return _extract_json_object(text)
    except json.JSONDecodeError:
        return _salvage_json_object(text)


def _all_block_payload(report: VideoReport) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in report.parts:
        for block in part.blocks:
            transcript_excerpt = "".join(line.text for line in block.transcript[:36])
            blocks.append(
                {
                    "part_page": part.page,
                    "part_title": part.title,
                    "block_id": block.id,
                    "start": block.start,
                    "end": block.end,
                    "title": block.title,
                    "summary": block.summary,
                    "transcript_excerpt": transcript_excerpt[:1800],
                }
            )
    return blocks


def _find_block(report: VideoReport, block_id: str):
    return find_block(report, block_id)


def visual_candidate_timestamps(start: float, end: float, timestamp: float, radius: float = 8.0) -> list[float]:
    duration = max(0.0, end - start)
    local = [timestamp - radius, timestamp - radius / 2, timestamp, timestamp + radius / 2, timestamp + radius]
    if duration >= 24:
        anchors = [
            start + duration * 0.18,
            start + duration * 0.35,
            start + duration * 0.52,
            start + duration * 0.70,
            start + duration * 0.88,
        ]
    else:
        anchors = [start, start + duration * 0.5, end]
    values = [max(start, min(end, value)) for value in [*local, *anchors]]
    return sorted({round(value, 3) for value in values})


def _transcript_windows(report: VideoReport, window_seconds: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for part in report.parts:
        if not part.transcript:
            continue
        current: list[TranscriptLine] = []
        window_start = part.transcript[0].start
        for line in part.transcript:
            if current and line.start - window_start >= window_seconds:
                windows.append(_window_payload(part.page, part.title, current))
                current = []
                window_start = line.start
            current.append(line)
        if current:
            windows.append(_window_payload(part.page, part.title, current))
    return windows


def _window_payload(part_page: int, part_title: str, lines: list[TranscriptLine]) -> dict[str, Any]:
    return {
        "part_page": part_page,
        "part_title": part_title,
        "start": lines[0].start,
        "end": lines[-1].end,
        "transcript": [
            {"start": line.start, "end": line.end, "text": line.text}
            for line in lines
        ],
    }


def _find_part(report: VideoReport, page: int):
    for part in report.parts:
        if part.page == page:
            return part
    return None


def _duration(part) -> float:
    if part.duration is not None:
        return float(part.duration)
    if part.transcript:
        return float(part.transcript[-1].end)
    return 0.0


BOUNDARY_STARTERS = (
    "接下来",
    "下面",
    "然后",
    "所以",
    "因此",
    "但是",
    "另外",
    "最后",
    "总结",
    "我们来看",
    "我们看",
    "回到",
    "现在",
    "好",
)


def _boundary_score(lines: list[TranscriptLine], index: int) -> float:
    if index <= 0 or index >= len(lines):
        return 0.0
    previous = lines[index - 1]
    current = lines[index]
    gap = max(0.0, current.start - previous.end)
    text = current.text.strip()
    score = 0.0
    if gap >= 1.2:
        score += 4.0
    elif gap >= 0.7:
        score += 2.5
    elif gap >= 0.35:
        score += 1.0
    if previous.text.strip().endswith(("。", "！", "？", ".", "!", "?")):
        score += 2.0
    if text.startswith(BOUNDARY_STARTERS):
        score += 2.5
    if len(text) <= 16 and previous.text.strip().endswith(("。", "！", "？")):
        score += 1.0
    return score


def _semantic_ranges_for_part(part, target_seconds: int) -> list[tuple[float, float]]:
    lines = part.transcript
    if not lines:
        return []
    if _duration(part) <= target_seconds:
        return [(lines[0].start, lines[-1].end)]

    ranges: list[tuple[float, float]] = []
    start = lines[0].start
    min_seconds = max(180.0, target_seconds * 0.55)
    max_seconds = max(float(target_seconds) * 1.35, min_seconds + 60.0)

    while start < lines[-1].end:
        target = start + target_seconds
        min_end = start + min_seconds
        max_end = start + max_seconds
        best_index: int | None = None
        best_score = -1.0
        fallback_index: int | None = None

        for index, line in enumerate(lines):
            if line.start <= start:
                continue
            if line.start < min_end:
                continue
            if line.start > max_end:
                break
            fallback_index = index
            score = _boundary_score(lines, index) - abs(line.start - target) / max(target_seconds, 1) * 1.5
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            best_index = fallback_index
        if best_index is None:
            ranges.append((start, lines[-1].end))
            break

        end = max(start, lines[best_index - 1].end)
        if end <= start:
            end = lines[best_index].start
        ranges.append((start, end))
        start = lines[best_index].start
        if lines[-1].end - start <= target_seconds * 0.45:
            ranges.append((start, lines[-1].end))
            break

    if len(ranges) >= 2:
        tail_start, tail_end = ranges[-1]
        if tail_end - tail_start < target_seconds * 0.35:
            previous_start, _ = ranges[-2]
            ranges[-2] = (previous_start, tail_end)
            ranges.pop()
    return ranges


def _window_payload_for_range(
    part_page: int,
    part_title: str,
    lines: list[TranscriptLine],
    start: float,
    end: float,
    context_start: float,
    context_end: float,
) -> dict[str, Any]:
    context_lines = lines_for_range(lines, context_start, context_end)
    return {
        "part_page": part_page,
        "part_title": part_title,
        "start": start,
        "end": end,
        "context_start": context_start,
        "context_end": context_end,
        "transcript": [
            {"start": line.start, "end": line.end, "text": line.text}
            for line in context_lines
        ],
    }


SHORT_PART_AS_LEAF_SECONDS = 600
DIRECT_PART_OUTLINE_SECONDS = 2400
BOUNDARY_REVIEW_CONTEXT_SECONDS = 180


def _root_windows(
    report: VideoReport,
    outline_window_seconds: int,
    outline_overlap_seconds: int,
    short_part_as_leaf_seconds: int = SHORT_PART_AS_LEAF_SECONDS,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for part in report.parts:
        if not part.transcript:
            continue
        if len(report.parts) > 1 and _duration(part) <= short_part_as_leaf_seconds:
            payload = _window_payload_for_range(
                part.page,
                part.title,
                part.transcript,
                part.transcript[0].start,
                part.transcript[-1].end,
                part.transcript[0].start,
                part.transcript[-1].end,
            )
            payload["root_title"] = part.title
            payload["leaf_ready"] = True
            windows.append(payload)
            continue
        ranges = _semantic_ranges_for_part(part, target_seconds=outline_window_seconds)
        for index, (start, end) in enumerate(ranges, start=1):
            context_start = max(part.transcript[0].start, start - outline_overlap_seconds)
            context_end = min(part.transcript[-1].end, end + outline_overlap_seconds)
            suffix = f" · 片段 {index}" if len(ranges) > 1 else ""
            payload = _window_payload_for_range(
                part.page,
                f"{part.title}{suffix}",
                part.transcript,
                start,
                end,
                context_start,
                context_end,
            )
            payload["root_title"] = f"片段 {index}" if len(ranges) > 1 else part.title
            windows.append(payload)
    return windows


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "是", "需要", "expand"}:
        return True
    if text in {"false", "no", "n", "0", "否", "不需要"}:
        return False
    return default


def _string_list(value: Any, limit: int = 8) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _quality_fields(item: dict[str, Any], duration: float) -> dict[str, Any]:
    granularity = str(item.get("granularity") or "").strip()
    if not granularity:
        if duration > STRONGLY_EXPAND_SECONDS:
            granularity = "coarse"
        elif duration > MAX_LEAF_SECONDS:
            granularity = "medium"
        else:
            granularity = "fine"
    return {
        "granularity": granularity,
        "should_expand": _truthy(item.get("should_expand"), default=duration > MAX_LEAF_SECONDS),
        "expand_reason": str(item.get("expand_reason") or "").strip(),
        "boundary_confidence": str(item.get("boundary_confidence") or "").strip(),
        "split_hints": _string_list(item.get("split_hints")),
    }


def _normalize_outline_nodes(
    raw_nodes: list[dict[str, Any]],
    part_start: float,
    part_end: float,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for item in raw_nodes:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start") or part_start)
            end = float(item.get("end") or part_end)
        except (TypeError, ValueError):
            start = part_start
            end = part_end
        start = max(part_start, min(part_end, start))
        end = max(start, min(part_end, end))
        nodes.append({**item, "start": start, "end": end})
    nodes.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    if not nodes:
        return []

    nodes[0]["start"] = part_start
    nodes[-1]["end"] = part_end
    for index in range(len(nodes) - 1):
        current = nodes[index]
        following = nodes[index + 1]
        current_end = float(current["end"])
        next_start = float(following["start"])
        if next_start > current_end:
            boundary = round((current_end + next_start) / 2, 3)
            current["end"] = boundary
            following["start"] = boundary
        elif next_start < current_end:
            boundary = round((current_end + next_start) / 2, 3)
            current["end"] = max(float(current["start"]), boundary)
            following["start"] = min(float(following["end"]), boundary)
    return nodes


async def outline_report(
    report: VideoReport,
    settings: LLMSettings,
    outline_window_seconds: int = 720,
    outline_overlap_seconds: int = 75,
    short_part_as_leaf_seconds: int = SHORT_PART_AS_LEAF_SECONDS,
    part_page: int | None = None,
    max_windows: int = 0,
    request_delay: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> None:
    def emit(message: str) -> None:
        if progress:
            progress(message)

    if not settings.text_model:
        raise ValueError("LLM text model is required.")

    client = _client(settings)
    emit("Building internal outline windows...")
    windows = _root_windows(
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
        part = _find_part(report, page)
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
                transcript=root_lines,
            )
            part.blocks.append(root)
            emit(f"P{page}: short part marked as leaf")
            continue

        part_plan: dict[str, Any] | None = None
        if (
            max_windows == 0
            and _duration(part) <= DIRECT_PART_OUTLINE_SECONDS
            and transcript_char_count(part.transcript) <= DIRECT_PART_OUTLINE_MAX_CHARS
        ):
            guidance = root_node_count_guidance(part)
            emit(
                f"P{page}: direct part outline "
                f"(target root nodes={guidance['suggested']}, duration={_duration(part):.1f}s)"
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
            candidates: list[dict[str, Any]] = []
            for window_index, window in enumerate(part_windows, start=1):
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
                for item in payload.get("candidates", payload.get("children", [])):
                    if not isinstance(item, dict):
                        continue
                    item["source_window_id"] = f"p{page}-w{window_index}"
                    item["source_window_start"] = window["start"]
                    item["source_window_end"] = window["end"]
                    candidates.append(item)

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
        for item in _normalize_outline_nodes(reduced.get("nodes", []), part_start, part_end):
            start = float(item["start"])
            end = float(item["end"])
            quality = _quality_fields(item, end - start)
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
                transcript=lines_for_range(part.transcript, start, end),
                **quality,
            )
            apply_outline_quality_gate(block)
            part.blocks.append(block)
            block_index += 1

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
        emit(f"P{page}: root nodes={len(part.blocks)}")

    report.llm_notes.append(
        f"Generated outline with {sum(len(part.blocks) for part in report.parts)} root nodes."
    )


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
    return _best_effort_json_object(response.choices[0].message.content or "{}")


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
    return _best_effort_json_object(response.choices[0].message.content or "{}")


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
    return _best_effort_json_object(response.choices[0].message.content or "{}")


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
    return _best_effort_json_object(response.choices[0].message.content or "{}")


async def _review_child_boundaries(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part,
    parent: KnowledgeBlock,
) -> dict[str, Any]:
    prompt = child_boundary_review_prompt(report, part, parent, parent.children)
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict reviewer for child topic boundaries in course-video outlines.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.1,
        extra_body=_extra_body(settings),
    )
    return _best_effort_json_object(response.choices[0].message.content or "{}")


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
    return _best_effort_json_object(response.choices[0].message.content or "{}")


async def expand_block(
    report: VideoReport,
    block_id: str,
    settings: LLMSettings,
    max_visual_requests: int = 3,
    request_delay: float = 0.0,
) -> list[VisualRequest]:
    if not settings.text_model:
        raise ValueError("LLM text model is required.")
    block = _find_block(report, block_id)
    if not block:
        raise ValueError(f"Block not found: {block_id}")
    part = _part_for_block(report, block)
    if not part:
        raise ValueError(f"Part not found for block: {block_id}")
    if block.status == "skeleton":
        apply_outline_quality_gate(block)

    client = _client(settings)
    payload = await _expand_block_payload(client, settings, report, part.page, block, max_visual_requests)
    if request_delay > 0:
        await asyncio.sleep(request_delay)

    mode = str(payload.get("mode") or "").strip().lower()
    if mode == "branch":
        block.children = []
        block.key_points = []
        block.visual_requests = []
        block.frames = []
        block.visual_analyses = []
        block.node_type = "branch"
        block.status = "expanded"
        block.expandable = False
        block.summary = str(payload.get("summary") or block.summary).strip()
        child_index = 1
        for item in payload.get("children", []):
            if not isinstance(item, dict):
                continue
            start = max(block.start, min(block.end, float(item.get("start") or block.start)))
            end = max(start, min(block.end, float(item.get("end") or block.end)))
            quality = _quality_fields(item, end - start)
            child = KnowledgeBlock(
                id=f"{block.id}-n{child_index}",
                title=str(item.get("title") or f"子节点 {child_index}").strip(),
                start=start,
                end=end,
                summary=str(item.get("summary") or "").strip(),
                node_type=str(item.get("node_type") or "branch").strip() or "branch",
                status="skeleton",
                expandable=bool(item.get("expandable", True)),
                depth=block.depth + 1,
                transcript=lines_for_range(part.transcript, start, end),
                **quality,
            )
            apply_outline_quality_gate(child)
            block.children.append(child)
            child_index += 1
        if len(block.children) > 1:
            boundary_payload = await _review_child_boundaries(client, settings, report, part, block)
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            changed = apply_boundary_adjustments_to_blocks(block.children, part.transcript, boundary_payload.get("boundaries", []))
            if changed:
                for child in block.children:
                    apply_outline_quality_gate(child)
                report.llm_notes.append(f"Reviewed {block.id} child boundaries and adjusted {changed}.")
        report.llm_notes.append(f"Expanded {block.id} into {len(block.children)} child nodes.")
        return []

    block.node_type = "leaf"
    block.status = "expanded"
    block.expandable = False
    block.children = [
        KnowledgeBlock(
            id=f"{block.id}-raw",
            title="字幕证据",
            start=block.start,
            end=block.end,
            summary="",
            node_type="raw",
            status="expanded",
            expandable=False,
            depth=block.depth + 1,
            transcript=block.transcript,
        )
    ]
    block.summary = str(payload.get("notes") or payload.get("summary") or block.summary).strip()
    block.key_points = [
        str(point).strip()
        for point in payload.get("key_points", [])
        if str(point).strip()
    ]
    block.sections = []
    for section_index, section in enumerate(payload.get("sections", []), start=1):
        if not isinstance(section, dict):
            continue
        block.sections.append(
            NoteSection(
                id=f"{block.id}-s{section_index}",
                title=str(section.get("title") or f"笔记段落 {section_index}").strip(),
                body=str(section.get("body") or "").strip(),
            )
        )
    if not block.sections and block.summary:
        block.sections.append(
            NoteSection(
                id=f"{block.id}-s1",
                title=block.title,
                body=block.summary,
            )
        )
    block.visual_requests = []
    block.frames = []
    block.visual_analyses = []
    visual_requests: list[VisualRequest] = []
    request_index = 1
    section_by_index = {str(index): section for index, section in enumerate(block.sections, start=1)}
    section_by_id = {section.id: section for section in block.sections}

    raw_visuals = list(payload.get("visual_requests", []))
    for section_index, section_payload in enumerate(payload.get("sections", []), start=1):
        if isinstance(section_payload, dict):
            for visual in section_payload.get("visual_requests", []):
                if isinstance(visual, dict):
                    visual = {**visual, "section_index": section_index}
                    raw_visuals.append(visual)

    if max_visual_requests <= 0:
        report.llm_notes.append(f"Expanded {block.id} into a leaf note with 0 visual requests.")
        return visual_requests

    seen_visual_keys: set[tuple[str, float, str, str]] = set()
    for visual in raw_visuals:
        if not isinstance(visual, dict) or not visual.get("needed", True):
            continue
        section = section_by_id.get(str(visual.get("section_id") or "")) or section_by_index.get(
            str(visual.get("section_index") or "")
        )
        if section is None and block.sections:
            section = block.sections[min(request_index - 1, len(block.sections) - 1)]
        timestamp = max(block.start, min(block.end, float(visual.get("timestamp") or ((block.start + block.end) / 2))))
        reason = str(visual.get("reason") or "").strip()
        prompt = (
            str(visual.get("prompt") or "").strip()
            or "Analyze this frame as a visual aid for understanding the note block."
        )
        visual_key = (
            section.id if section is not None else "",
            round(timestamp, 3),
            " ".join(reason.split()),
            " ".join(prompt.split()),
        )
        if visual_key in seen_visual_keys:
            continue
        seen_visual_keys.add(visual_key)
        request = VisualRequest(
            id=f"{block.id}-vr{request_index}",
            part_page=part.page,
            block_id=block.id,
            timestamp=timestamp,
            candidate_timestamps=visual_candidate_timestamps(block.start, block.end, timestamp),
            reason=reason,
            prompt=prompt,
            section_id=section.id if section is not None else None,
        )
        if section is not None:
            section.visual_requests.append(request)
        block.visual_requests.append(request)
        visual_requests.append(request)
        request_index += 1
        if len(visual_requests) >= max_visual_requests:
            break
    report.llm_notes.append(f"Expanded {block.id} into a leaf note with {len(visual_requests)} visual requests.")
    return visual_requests


def _part_for_block(report: VideoReport, target: KnowledgeBlock):
    if target.source_part_page is not None:
        part = _find_part(report, target.source_part_page)
        if part is not None:
            return part
    for part in report.parts:
        for block in part.blocks:
            if block is target:
                return part
            if _contains_block(block, target.id):
                return part
    return None


def _contains_block(block: KnowledgeBlock, block_id: str) -> bool:
    if block.id == block_id:
        return True
    return any(_contains_block(child, block_id) for child in block.children)


async def _expand_block_payload(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part_page: int,
    block: KnowledgeBlock,
    max_visual_requests: int,
) -> dict[str, Any]:
    duration = block.end - block.start
    force_leaf = duration <= SHORT_BLOCK_AS_LEAF_SECONDS and block.node_type == "leaf"
    prompt = {
        "video_title": report.title,
        "block": {
            "id": block.id,
            "title": block.title,
            "start": block.start,
            "end": block.end,
            "summary": block.summary,
            "depth": block.depth,
        },
        "task": (
            "Expand this selected course-video node. Decide whether it should remain an outline branch "
            "or become a leaf study note. If the selected range still contains multiple substantial topics, "
            "return mode='branch' with child nodes. If it is coherent enough, return mode='leaf' with Markdown notes."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "If force_leaf=true, you must return mode='leaf'. Do not split into child nodes.",
            "If the selected block is marked node_type='branch', treat it as an intermediate node and prefer mode='branch' unless its range is already a self-contained leaf topic.",
            "If mode='branch', do not write full notes or request images.",
            f"If force_leaf=false, prefer mode='branch' for ranges longer than {MAX_LEAF_SECONDS} seconds unless the content is truly one simple topic.",
            f"Never return child node_type='leaf' for a child longer than {MAX_LEAF_SECONDS} seconds.",
            "For branch children, set should_expand=true when the child still contains multiple concepts, examples, or algorithm/proof steps.",
            "If mode='leaf', write learner-facing note sections instead of one long essay.",
            "Leaf sections should be 2-5 coherent sections with titles and Markdown bodies.",
            "Bind visual requests to the section where the image should appear, using section_index.",
            "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
            "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
            "For leaf visual requests, choose timestamps where the slide/board/UI is likely content-rich, not transition/typing-empty frames.",
            "A leaf may request zero, one, or multiple images according to learning value.",
        ],
        "force_leaf": force_leaf,
        "limits": {"max_visual_requests": max_visual_requests},
        "schema": {
            "mode": "branch|leaf",
            "summary": "Markdown string",
            "children": [
                {
                    "title": "string",
                    "start": "number",
                    "end": "number",
                    "summary": "short Markdown string",
                    "node_type": "branch|leaf",
                    "expandable": "boolean",
                    "granularity": "coarse|medium|fine",
                    "should_expand": "boolean",
                    "expand_reason": "short Chinese string",
                    "boundary_confidence": "high|medium|low",
                    "split_hints": ["short Chinese child-topic hint"],
                }
            ],
            "notes": "Markdown string, only for leaf",
            "key_points": ["Markdown string, only for leaf"],
            "sections": [
                {
                    "title": "string, only for leaf",
                    "body": "Markdown string, only for leaf",
                    "visual_requests": [
                        {
                            "needed": "boolean",
                            "timestamp": "number",
                            "reason": "Markdown string",
                            "prompt": "Markdown string",
                        }
                    ],
                }
            ],
            "visual_requests": [
                {
                    "needed": "boolean",
                    "timestamp": "number",
                    "section_index": "number, 1-based, if this image belongs to a leaf section",
                    "reason": "Markdown string",
                    "prompt": "Markdown string",
                }
            ],
        },
        "transcript": [
            {"start": line.start, "end": line.end, "text": line.text}
            for line in block.transcript
        ],
    }
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You expand selected nodes in an incremental course-study report.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    return _best_effort_json_object(response.choices[0].message.content or "{}")


async def ai_segment_report(
    report: VideoReport,
    settings: LLMSettings,
    window_seconds: int = 480,
    max_visual_requests: int = 0,
    request_delay: float = 0.0,
) -> list[VisualRequest]:
    if not settings.text_model:
        raise ValueError("LLM text model is required.")

    client = _client(settings)
    windows = _transcript_windows(report, window_seconds=window_seconds)
    visual_requests: list[VisualRequest] = []
    request_index = 1

    for part in report.parts:
        part.blocks = []

    part_block_counts: dict[int, int] = {}
    for window in windows:
        payload = await _segment_window(client, settings, report, window)
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        part = _find_part(report, int(window["part_page"]))
        if not part:
            continue

        for item in payload.get("blocks", []):
            if not isinstance(item, dict):
                continue
            start = float(item.get("start") or window["start"])
            end = float(item.get("end") or window["end"])
            start = max(float(window["start"]), min(float(window["end"]), start))
            end = max(start, min(float(window["end"]), end))
            part_block_counts[part.page] = part_block_counts.get(part.page, 0) + 1
            block_id = f"p{part.page}-ai{part_block_counts[part.page]}"
            evidence = lines_for_range(part.transcript, start, end)
            block = KnowledgeBlock(
                id=block_id,
                title=str(item.get("title") or f"学习片段 {part_block_counts[part.page]}").strip(),
                start=start,
                end=end,
                summary=str(item.get("notes") or item.get("summary") or "").strip(),
                key_points=[
                    str(point).strip()
                    for point in item.get("key_points", [])
                    if str(point).strip()
                ],
                transcript=evidence,
                children=[
                    KnowledgeBlock(
                        id=f"{block_id}-raw",
                        title="字幕证据",
                        start=start,
                        end=end,
                        summary="",
                        transcript=evidence,
                    )
                ],
            )
            part.blocks.append(block)

            raw_visuals = item.get("visual_requests")
            if not isinstance(raw_visuals, list):
                legacy_visual = item.get("visual_request")
                raw_visuals = [legacy_visual] if isinstance(legacy_visual, dict) else []

            for visual in raw_visuals:
                has_visual_budget = max_visual_requests <= 0 or len(visual_requests) < max_visual_requests
                if not has_visual_budget or not isinstance(visual, dict) or not visual.get("needed"):
                    continue
                timestamp = float(visual.get("timestamp") or ((start + end) / 2))
                timestamp = max(start, min(end, timestamp))
                request = VisualRequest(
                    id=f"vr-{request_index}",
                    part_page=part.page,
                    block_id=block_id,
                    timestamp=timestamp,
                    reason=str(visual.get("reason") or "").strip(),
                    prompt=str(visual.get("prompt") or "").strip()
                    or "Analyze this frame as a visual aid for understanding the note block.",
                )
                block.visual_requests.append(request)
                visual_requests.append(request)
                request_index += 1

    report.llm_notes.append(
        f"AI segmented subtitles into {sum(len(part.blocks) for part in report.parts)} learning blocks and requested {len(visual_requests)} visual frames."
    )
    return visual_requests


async def _segment_window(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    window: dict[str, Any],
) -> dict[str, Any]:
    prompt = {
        "video_title": report.title,
        "part": {"page": window["part_page"], "title": window["part_title"]},
        "window": {"start": window["start"], "end": window["end"]},
        "task": (
            "Convert raw ASR subtitles into clean learning-note blocks. Filter filler words, greetings, repeated phrases, "
            "self-talk, and irrelevant chatter. Preserve only study-relevant information. Choose meaningful segment "
            "boundaries based on topic changes, not fixed time."
        ),
        "rules": [
            "Return valid JSON only.",
            "Each block should be a concise study note, not a transcript summary.",
            "Use the original timestamp range for each block.",
            "Prefer 2-5 blocks for this window, fewer if the topic is continuous.",
            "Keep notes in Chinese unless the source term must remain English.",
            "Write all knowledge-bearing text fields in Markdown: notes, key_points, visual request reason, and prompt.",
            "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
            "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
            "Prefer clear formulas over vague prose for definitions, derivations, constraints, and transformations.",
            "If slides, formulas, diagrams, tables, code, or visible written examples would improve understanding, request visual aid frames.",
            "A block may need multiple visual aid frames when the board/slide evolves across steps, formulas transform, examples are solved in stages, or several distinct diagrams are discussed.",
            "Do not force one-image-per-block. Use zero, one, or multiple images according to learning value.",
        ],
        "schema": {
            "blocks": [
                {
                    "title": "string",
                    "start": "number",
                    "end": "number",
                    "notes": "Markdown string with formulas when helpful",
                    "key_points": ["Markdown string"],
                    "visual_requests": [
                        {
                            "needed": "boolean",
                            "timestamp": "number",
                            "reason": "Markdown string",
                            "prompt": "Markdown string",
                        }
                    ],
                }
            ]
        },
        "transcript": window["transcript"],
    }

    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You are a learning-note architect. You filter noisy ASR into useful study notes.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    content = response.choices[0].message.content or "{}"
    payload = _best_effort_json_object(content)
    if not payload:
        report.llm_notes.append(
            f"AI segmentation skipped window {window['start']:.1f}-{window['end']:.1f}s because model returned invalid JSON."
        )
        return {"blocks": []}
    return payload


async def enrich_report_text(
    report: VideoReport,
    settings: LLMSettings,
    max_blocks: int,
    max_visual_requests: int,
    batch_size: int = 12,
    request_delay: float = 0.0,
) -> list[VisualRequest]:
    if not settings.text_model:
        raise ValueError("LLM text model is required.")

    all_blocks = _all_block_payload(report)
    blocks = all_blocks[:max_blocks] if max_blocks > 0 else all_blocks
    if not blocks:
        report.llm_notes.append("No subtitle blocks available for text enrichment.")
        return []

    client = _client(settings)
    requests: list[VisualRequest] = []
    request_index = 1

    for batch_start in range(0, len(blocks), batch_size):
        batch = blocks[batch_start : batch_start + batch_size]
        unlimited_visual_requests = max_visual_requests <= 0
        if unlimited_visual_requests:
            visual_budget_for_batch = len(batch)
        else:
            remaining_visual_requests = max_visual_requests - len(requests)
            if remaining_visual_requests <= 0:
                visual_budget_for_batch = 0
            else:
                visual_budget_for_batch = min(remaining_visual_requests, max(1, len(batch) // 3))

        payload = await _enrich_block_batch(
            client=client,
            settings=settings,
            report=report,
            blocks=batch,
            max_visual_requests=visual_budget_for_batch,
        )
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        for update in payload.get("block_updates", []):
            if not isinstance(update, dict):
                continue
            block = _find_block(report, str(update.get("block_id", "")))
            if not block:
                continue
            title = str(update.get("title") or "").strip()
            summary = str(update.get("summary") or "").strip()
            if title:
                block.title = title
            if summary:
                block.summary = summary

        if visual_budget_for_batch <= 0:
            continue

        for item in payload.get("visual_requests", [])[:visual_budget_for_batch]:
            if not isinstance(item, dict):
                continue
            block_id = str(item.get("block_id") or "").strip()
            block = _find_block(report, block_id)
            if not block:
                continue
            timestamp = float(item.get("timestamp") or ((block.start + block.end) / 2))
            timestamp = max(block.start, min(block.end, timestamp))
            part_page = int(block_id.split("-")[0].removeprefix("p")) if block_id.startswith("p") else 0
            request = VisualRequest(
                id=f"vr-{request_index}",
                part_page=part_page,
                block_id=block_id,
                timestamp=timestamp,
                reason=str(item.get("reason") or "").strip(),
                prompt=str(item.get("prompt") or "").strip()
                or "Describe the educational content visible in this frame.",
            )
            block.visual_requests.append(request)
            requests.append(request)
            request_index += 1
            if not unlimited_visual_requests and len(requests) >= max_visual_requests:
                break

    report.llm_notes.append(
        f"Text model enriched {len(blocks)} blocks in batches and requested {len(requests)} visual frames."
    )
    return requests


async def _enrich_block_batch(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    blocks: list[dict[str, Any]],
    max_visual_requests: int,
) -> dict[str, Any]:
    prompt = {
        "video_title": report.title,
        "task": (
            "Act as a learning analyst for a course video. Improve block titles and learner-facing "
            "summaries, and decide which blocks need visual aids. Your goal is to guide studying, "
            "not merely summarize content."
        ),
        "rules": [
            "Return valid JSON only.",
            "Keep titles concise and learner-oriented.",
            "Summaries should be Markdown and explain the learning point, why it matters, and how it connects to the surrounding lesson.",
            "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
            "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
            "Use Markdown for all knowledge-bearing text fields: summary, visual request reason, and visual request prompt.",
            "Use timestamps within each block start/end.",
            "For visual requests, choose the most relevant precise moment. Use decimal seconds when useful; do not round to whole seconds by default.",
            "Do not request screenshots for purely conversational sections.",
            "When a block depends on slides, diagrams, formulas, code, UI, tables, architecture, or workflow visuals, request a screenshot. Good course notes should keep useful visual aids.",
            "Prefer at most one screenshot per block unless multiple distinct visuals are necessary.",
        ],
        "limits": {
            "max_visual_requests_total": max_visual_requests if max_visual_requests > 0 else "unlimited",
            "max_visual_requests_this_batch": max_visual_requests,
        },
        "schema": {
            "block_updates": [
                {
                    "block_id": "string",
                    "title": "string",
                    "summary": "Markdown string with formulas when helpful",
                }
            ],
            "visual_requests": [
                {
                    "block_id": "string",
                    "timestamp": "number",
                    "reason": "Markdown string",
                    "prompt": "Markdown string",
                }
            ],
        },
        "blocks": blocks,
    }

    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You turn course-video subtitles into structured study reports.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    content = response.choices[0].message.content or "{}"
    return _extract_json_object(content)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def analyze_frames(
    report: VideoReport,
    settings: LLMSettings,
    frames: list[FrameArtifact],
    request_delay: float = 0.0,
) -> list[VisualAnalysis]:
    if not frames:
        return []
    if not settings.vision_model:
        report.llm_notes.append("Vision model not configured; frames were saved without visual analysis.")
        return []

    analyses: list[VisualAnalysis] = []
    client = _client(settings)
    for frame in frames:
        if frame.error:
            continue
        block = _find_block(report, frame.block_id)
        if not block:
            continue
        request = next(
            (item for item in block.visual_requests if item.id == frame.request_id),
            None,
        )
        image_path = Path(frame.path)
        prompt = {
            "task": (
                "Analyze this course-video frame as a concise Chinese learning-note supplement. "
                "Do not write an image report. Do not merely describe the picture. Explain only what helps "
                "the learner understand the surrounding note block."
            ),
            "rules": [
                "Return valid JSON only.",
                "All visible learner-facing text must be Simplified Chinese, except unavoidable symbols, formulas, or source terms.",
                "Use Markdown for all knowledge-bearing text fields.",
                "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
                "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
                "Be brief. summary <= 80 Chinese characters, learning_value <= 80 Chinese characters, guidance <= 120 Chinese characters.",
                "Do not copy, paraphrase, or restate the whole block note. Focus only on this image's added value.",
                "For math course frames, identify the visible formula, diagram, table, or solving step first, then explain how it supports the current note.",
                "When the frame contains multiple merged visual_focus directions, synthesize them into one coherent explanation instead of listing duplicate image reports.",
                "If the requested visual_focus is not actually visible in the frame, say what is visible and how it can still help; do not invent missing formulas or diagrams.",
                "Prefer problem-solving guidance: what structure to recognize, what transformation is being used, and which exam mistake it prevents.",
                "Do not use Markdown headings such as #, ##, or ### in visual analysis fields.",
                "summary must be one short sentence about the image, not a sectioned note.",
                "observed_elements should contain 1-4 short items. pitfalls should contain 0-3 short items.",
                "Use concrete study guidance: what to look at, how it connects to the note, and what mistake it prevents.",
                "Do not mention confidence, OCR limitations, or generic phrases unless genuinely necessary.",
            ],
            "block": {
                "id": block.id,
                "title": block.title,
                "summary": block.summary,
                "timestamp": frame.timestamp,
            },
            "visual_focus": {
                "reason": request.reason if request is not None else "",
                "prompt": request.prompt if request is not None else "",
            },
            "return_json_schema": {
                "summary": "Markdown：这张图在讲什么，中文短句，不要标题",
                "observed_elements": ["Markdown：图中最值得看的元素，中文短句"],
                "learning_value": "Markdown：为什么这张图值得放进笔记，中文短句",
                "guidance": "Markdown：读者应该怎样看这张图，中文短句",
                "pitfalls": ["Markdown：容易误解或漏看的点，中文短句"],
                "confidence": "low|medium|high",
            },
        }
        try:
            response = await client.chat.completions.create(
                model=settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                            {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                        ],
                    }
                ],
                temperature=0.2,
                extra_body=_extra_body(settings),
            )
        except OpenAIError as exc:
            report.llm_notes.append(
                f"Vision analysis stopped after {len(analyses)} frames: {exc}"
            )
            return analyses
        content = response.choices[0].message.content or "{}"
        payload = _best_effort_json_object(content)
        fallback_summary = content.strip()
        if len(fallback_summary) > 900:
            fallback_summary = fallback_summary[:899].rstrip() + "..."
        analysis = VisualAnalysis(
            request_id=frame.request_id,
            part_page=frame.part_page,
            block_id=frame.block_id,
            timestamp=frame.timestamp,
            image_path=frame.path,
            summary=str(payload.get("summary") or fallback_summary).strip(),
            observed_elements=[
                str(item).strip()
                for item in payload.get("observed_elements", [])
                if str(item).strip()
            ],
            learning_value=str(payload.get("learning_value") or "").strip(),
            guidance=str(payload.get("guidance") or "").strip(),
            pitfalls=[
                str(item).strip()
                for item in payload.get("pitfalls", [])
                if str(item).strip()
            ],
            confidence=str(payload.get("confidence") or "unknown").strip(),
        )
        block.visual_analyses.append(analysis)
        if request and request.section_id:
            for section in block.sections:
                if section.id == request.section_id:
                    section.visual_analyses.append(analysis)
                    break
        analyses.append(analysis)
        if request_delay > 0:
            await asyncio.sleep(request_delay)

    report.llm_notes.append(f"Vision model analyzed {len(analyses)} frames.")
    return analyses


async def choose_visual_frames(
    report: VideoReport,
    settings: LLMSettings,
    requests: list[VisualRequest],
    candidates_by_request: dict[str, list[FrameArtifact]],
    request_delay: float = 0.0,
) -> list[VisualRequest]:
    if not requests:
        return []
    if not settings.vision_model:
        return requests

    client = _client(settings)
    chosen: list[VisualRequest] = []
    for request in requests:
        block = _find_block(report, request.block_id)
        candidates = [
            frame
            for frame in candidates_by_request.get(request.id, [])
            if frame.path and not frame.error
        ]
        if not block or not candidates:
            chosen.append(request)
            continue

        prompt = {
            "task": (
                "Choose the best frame for a course-note visual aid. The frame should be content-rich, "
                "visually informative, and avoid transition moments, mostly blank slides, or partially typed input."
            ),
            "rules": [
                "Return valid JSON only.",
                "Pick one candidate only if it is genuinely good enough for a study note.",
                "If all candidates are poor, mostly blank, transitional, or show only partial typing/input, return retry_timestamp instead of choosing the least bad candidate.",
                "For UI typing or animated build-up, prefer a later completed state.",
                "The retry timestamp may be anywhere inside the block range; use the transcript/context to estimate a richer moment.",
                "Prefer frames where the slide/board/UI is visually complete and useful for learning.",
            ],
            "block": {
                "id": block.id,
                "title": block.title,
                "summary": block.summary,
                "start": block.start,
                "end": block.end,
            },
            "visual_request": {
                "reason": request.reason,
                "prompt": request.prompt,
                "original_timestamp": request.timestamp,
            },
            "candidates": [
                {"index": index, "timestamp": frame.timestamp}
                for index, frame in enumerate(candidates, start=1)
            ],
            "return_json_schema": {
                "decision": "use_candidate|retry_timestamp|skip",
                "candidate_index": "number, if use_candidate",
                "timestamp": "number, if retry_timestamp",
                "reason": "short Chinese reason",
            },
        }
        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, ensure_ascii=False)}]
        for index, frame in enumerate(candidates, start=1):
            content.append({"type": "text", "text": f"candidate_index={index}, timestamp={frame.timestamp:.3f}s"})
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(frame.path))}})

        try:
            response = await client.chat.completions.create(
                model=settings.vision_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.1,
                extra_body=_extra_body(settings),
            )
        except OpenAIError as exc:
            report.llm_notes.append(f"Visual frame choice failed for {request.id}: {exc}")
            chosen.append(request)
            continue

        payload = _best_effort_json_object(response.choices[0].message.content or "{}")
        decision = str(payload.get("decision") or "").strip()
        if decision == "skip":
            report.llm_notes.append(f"Skipped visual request {request.id}: {payload.get('reason') or ''}")
            continue
        if decision == "retry_timestamp":
            timestamp = max(block.start, min(block.end, float(payload.get("timestamp") or request.timestamp)))
            candidate_timestamps = visual_candidate_timestamps(block.start, block.end, timestamp, radius=12.0)
        else:
            index = int(payload.get("candidate_index") or 1)
            index = max(1, min(len(candidates), index))
            timestamp = candidates[index - 1].timestamp
            candidate_timestamps = []
        chosen.append(
            VisualRequest(
                id=request.id,
                part_page=request.part_page,
                block_id=request.block_id,
                timestamp=timestamp,
                candidate_timestamps=candidate_timestamps,
                reason=request.reason,
                prompt=request.prompt,
                section_id=request.section_id,
            )
        )
        if request_delay > 0:
            await asyncio.sleep(request_delay)

    report.llm_notes.append(f"Vision model selected {len(chosen)} final visual frames from candidates.")
    return chosen


async def rewrite_visual_notes(
    report: VideoReport,
    settings: LLMSettings,
    request_delay: float = 0.0,
) -> int:
    if not settings.text_model:
        raise ValueError("LLM text model is required.")

    client = _client(settings)
    rewritten = 0
    for part in report.parts:
        for block in part.blocks:
            for analysis in block.visual_analyses:
                payload = await _rewrite_visual_note(client, settings, block, analysis)
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
                if not payload:
                    continue
                analysis.summary = str(payload.get("summary") or analysis.summary).strip()
                analysis.learning_value = str(
                    payload.get("learning_value") or analysis.learning_value
                ).strip()
                analysis.guidance = str(payload.get("guidance") or analysis.guidance).strip()
                analysis.observed_elements = [
                    str(item).strip()
                    for item in payload.get("observed_elements", analysis.observed_elements)
                    if str(item).strip()
                ][:4]
                analysis.pitfalls = [
                    str(item).strip()
                    for item in payload.get("pitfalls", analysis.pitfalls)
                    if str(item).strip()
                ][:3]
                rewritten += 1

    report.llm_notes.append(f"Rewrote {rewritten} visual notes into concise Chinese study aids.")
    return rewritten


async def _rewrite_visual_note(
    client: AsyncOpenAI,
    settings: LLMSettings,
    block: KnowledgeBlock,
    analysis: VisualAnalysis,
) -> dict[str, Any]:
    prompt = {
        "task": (
            "Rewrite an existing visual analysis into a concise Simplified Chinese study-note supplement. "
            "The reader is studying a Bilibili course report. Preserve useful mathematical/formula details, "
            "but remove verbosity, English report tone, and generic descriptions."
        ),
        "rules": [
            "Return valid JSON only.",
            "All learner-facing text must be Simplified Chinese, except formulas, symbols, and source terms.",
            "Use Markdown for all knowledge-bearing text fields.",
            "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
            "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
            "summary <= 80 Chinese characters.",
            "learning_value <= 80 Chinese characters.",
            "guidance <= 120 Chinese characters.",
            "observed_elements: 1-4 short items.",
            "pitfalls: 0-3 short items.",
            "Write as a note beside an image, not as an essay.",
        ],
        "block": {
            "title": block.title,
            "summary": block.summary,
        },
        "current_visual_analysis": {
            "summary": analysis.summary,
            "observed_elements": analysis.observed_elements,
            "learning_value": analysis.learning_value,
            "guidance": analysis.guidance,
            "pitfalls": analysis.pitfalls,
        },
        "return_json_schema": {
            "summary": "Markdown：看什么",
            "observed_elements": ["Markdown：图中元素"],
            "learning_value": "Markdown：为什么有用",
            "guidance": "Markdown：怎么用",
            "pitfalls": ["Markdown：易错点"],
        },
    }
    response = await client.chat.completions.create(
        model=settings.text_model,
        messages=[
            {
                "role": "system",
                "content": "You rewrite verbose visual analyses into concise Chinese study notes.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
        extra_body=_extra_body(settings),
    )
    content = response.choices[0].message.content or "{}"
    return _best_effort_json_object(content)
