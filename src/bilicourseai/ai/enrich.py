from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from bilicourseai.ai.common import all_block_payload, find_report_block
from bilicourseai.json_utils import extract_json_object as _extract_json_object
from bilicourseai.llm_client import create_client as _client, extra_body as _extra_body
from bilicourseai.models import VideoReport, VisualRequest
from bilicourseai.settings import LLMSettings
from bilicourseai.visual.timing import candidate_timestamps_from_payload as _candidate_timestamps_from_payload


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

    all_blocks = all_block_payload(report)
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
            block = find_report_block(report, str(update.get("block_id", "")))
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
            block = find_report_block(report, block_id)
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
                candidate_timestamps=_candidate_timestamps_from_payload(item, block.start, block.end, timestamp),
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
            "Treat subtitles as noisy evidence, not ground truth. If a block contains high-risk details such as exact terms, numbers, formulas, variables, code identifiers, table labels, graph nodes, or flowchart paths, use the visual request prompt to ask the vision model to verify them when possible.",
            "For each visual request, provide candidate_timestamps: 3-7 precise seconds inside the block where the relevant visual may be most complete. Choose them semantically from the transcript and topic flow; do not use mechanical offsets like t-8,t-4,t,t+4,t+8.",
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
                    "candidate_timestamps": ["number, 3-7 semantic candidate seconds inside the block"],
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
