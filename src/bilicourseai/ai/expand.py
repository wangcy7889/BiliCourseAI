from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI

from bilicourseai.ai.common import find_report_block, find_report_part
from bilicourseai.json_utils import best_effort_json_object as _best_effort_json_object
from bilicourseai.llm_client import create_client as _client, extra_body as _extra_body
from bilicourseai.models import KnowledgeBlock, NoteSection, VideoReport, VisualRequest
from bilicourseai.outline.boundaries import apply_boundary_adjustments_to_blocks
from bilicourseai.outline.quality import (
    MAX_LEAF_SECONDS,
    SHORT_BLOCK_AS_LEAF_SECONDS,
    apply_outline_quality_gate,
    merge_short_adjacent_nodes,
)
from bilicourseai.outline.normalization import quality_fields
from bilicourseai.outline.prompts import child_boundary_review_prompt
from bilicourseai.settings import LLMSettings
from bilicourseai.transcripts.transcript import lines_for_range
from bilicourseai.visual.timing import candidate_timestamps_from_payload as _candidate_timestamps_from_payload


async def expand_block(
    report: VideoReport,
    block_id: str,
    settings: LLMSettings,
    max_visual_requests: int = 3,
    request_delay: float = 0.0,
) -> list[VisualRequest]:
    if not settings.text_model:
        raise ValueError("LLM text model is required.")
    block = find_report_block(report, block_id)
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
    leaf_issue = _leaf_payload_quality_issue(payload, block) if mode != "branch" else ""
    if leaf_issue:
        report.llm_notes.append(f"Retried {block.id} leaf note because it looked too thin: {leaf_issue}")
        payload = await _expand_block_payload(
            client,
            settings,
            report,
            part.page,
            block,
            max_visual_requests,
            force_leaf_override=True,
            leaf_quality_feedback=leaf_issue,
        )
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
            quality = quality_fields(item, end - start)
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
        merged_short = merge_short_adjacent_nodes(block.children)
        if merged_short:
            report.llm_notes.append(f"Merged {merged_short} short ordinary child node(s) under {block.id}.")
        if len(block.children) < 2:
            report.llm_notes.append(f"Retried {block.id} as leaf because branch expansion produced fewer than 2 child nodes.")
            payload = await _expand_block_payload(
                client,
                settings,
                report,
                part.page,
                block,
                max_visual_requests,
                force_leaf_override=True,
            )
            if request_delay > 0:
                await asyncio.sleep(request_delay)
            mode = "leaf"
            leaf_issue = _leaf_payload_quality_issue(payload, block)
            if leaf_issue:
                report.llm_notes.append(f"Retried {block.id} forced leaf note because it looked too thin: {leaf_issue}")
                payload = await _expand_block_payload(
                    client,
                    settings,
                    report,
                    part.page,
                    block,
                    max_visual_requests,
                    force_leaf_override=True,
                    leaf_quality_feedback=leaf_issue,
                )
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            block.children = []
            block.key_points = []
            block.visual_requests = []
            block.frames = []
            block.visual_analyses = []
        else:
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
            candidate_timestamps=_candidate_timestamps_from_payload(visual, block.start, block.end, timestamp),
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
        part = find_report_part(report, target.source_part_page)
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


def _leaf_payload_quality_issue(payload: dict[str, Any], block: KnowledgeBlock) -> str:
    if str(payload.get("mode") or "").strip().lower() == "branch":
        return ""

    duration = block.end - block.start
    if duration < 120:
        return ""

    sections = [section for section in payload.get("sections", []) if isinstance(section, dict)]
    bodies = [str(section.get("body") or "").strip() for section in sections]
    nonempty_bodies = [body for body in bodies if body]
    body_chars = sum(len(body) for body in nonempty_bodies)
    key_points = [str(point).strip() for point in payload.get("key_points", []) if str(point).strip()]
    notes = str(payload.get("notes") or payload.get("summary") or "").strip()
    has_structured_markdown = any(
        re.search(r"(^|\n)\s*(?:[-*]|\d+[.．、])\s+", body) or re.search(r"\n\s*\|.+\|\s*\n", body)
        for body in nonempty_bodies
    )

    issues: list[str] = []
    if len(nonempty_bodies) < (3 if duration >= 240 else 2):
        issues.append("section 数量不足")
    if body_chars < (900 if duration >= 240 else 600):
        issues.append("section 正文过短")
    if len(key_points) < 3:
        issues.append("key_points 过少")
    if duration >= 180 and not has_structured_markdown:
        issues.append("缺少列表、表格或分层结构")
    if len(nonempty_bodies) == 1 and notes and nonempty_bodies[0].strip() == notes:
        issues.append("section 只是重复 summary")
    return "；".join(issues)


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


async def _expand_block_payload(
    client: AsyncOpenAI,
    settings: LLMSettings,
    report: VideoReport,
    part_page: int,
    block: KnowledgeBlock,
    max_visual_requests: int,
    force_leaf_override: bool = False,
    leaf_quality_feedback: str | None = None,
) -> dict[str, Any]:
    duration = block.end - block.start
    force_leaf = force_leaf_override or (duration <= SHORT_BLOCK_AS_LEAF_SECONDS and block.node_type == "leaf")
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
            "Do not return mode='branch' if you can only produce one meaningful child; return mode='leaf' instead.",
            "For short ranges, if the possible child topics would be only 1-2 tiny adjacent items, keep them inside one leaf note with sections or bullets instead of creating a one-child branch.",
            "If mode='branch', do not write full notes or request images.",
            f"If force_leaf=false, prefer mode='branch' for ranges longer than {MAX_LEAF_SECONDS} seconds unless the content is truly one simple topic.",
            f"Never return child node_type='leaf' for a child longer than {MAX_LEAF_SECONDS} seconds.",
            "Avoid standalone child nodes shorter than 75 seconds unless the child is a complete example/problem, formula derivation/proof, full procedure/algorithm, or independent exercise.",
            "When several adjacent short items are just a list of terms, principles, model types, rules, or comparison dimensions, merge them into one coherent child and present the items later as a table/list inside the leaf note.",
            "For branch children, set should_expand=true when the child still contains multiple concepts, examples, or algorithm/proof steps.",
            "If mode='leaf', write a complete learner-facing study unit, not a summary paragraph.",
            "For leaf notes, keep `notes` as a short overview and put the actual learning content in `sections`.",
            "Leaf sections should be 2-5 coherent sections with titles and Markdown bodies. For leaf ranges over 240 seconds, use at least 3 sections unless the transcript is almost empty.",
            "Do not return a single section whose title merely repeats the block title and whose body merely restates the summary.",
            "Leaf notes should usually include 3-6 key_points that help a learner review the concept quickly.",
            "When the leaf compares languages, rules, categories, examples, conditions, procedures, or pros/cons, use a Markdown table or clear bullet list inside a section.",
            "Preserve concrete examples from the transcript when they teach the concept, but clean subtitle noise and mark uncertain exact wording when needed.",
            "Bind visual requests to the section where the image should appear, using section_index.",
            "Use Markdown math for formulas whenever useful: inline `$...$`, display `$$...$$`, or `\\(...\\)` / `\\[...\\]`.",
            "Because the response is JSON, escape LaTeX backslashes correctly as `\\\\` inside JSON strings.",
            "Treat subtitles as noisy evidence, not ground truth. Do not imply 100% confidence when the note depends on a precise word, number, formula, variable, code identifier, graph node, path label, table label, or proper noun extracted from subtitles.",
            "For high-risk details, cross-check surrounding context and request a visual aid when the slide/board can disambiguate it. If still uncertain, mark it as uncertain or say it appears to be so, instead of guessing.",
            "For leaf visual requests, choose timestamps where the slide/board/UI is likely content-rich, not transition/typing-empty frames.",
            "For each visual request, provide candidate_timestamps: 3-7 precise seconds inside this block where the relevant visual may be most complete. Choose them semantically from the transcript and topic flow; do not use mechanical offsets like t-8,t-4,t,t+4,t+8.",
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
                            "candidate_timestamps": ["number, 3-7 semantic candidate seconds inside the block"],
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
                    "candidate_timestamps": ["number, 3-7 semantic candidate seconds inside the block"],
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
    if leaf_quality_feedback:
        prompt["leaf_quality_retry"] = {
            "problem": leaf_quality_feedback,
            "instruction": (
                "Retry as mode='leaf'. Make the note dense enough to teach the selected range: "
                "use multiple titled sections, concrete examples, key points, and tables/lists where helpful. "
                "Do not split into child nodes."
            ),
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
