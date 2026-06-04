from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from bilicourseai.ai.common import find_report_part, transcript_windows
from bilicourseai.json_utils import best_effort_json_object as _best_effort_json_object
from bilicourseai.llm_client import create_client as _client, extra_body as _extra_body
from bilicourseai.models import KnowledgeBlock, VideoReport, VisualRequest
from bilicourseai.settings import LLMSettings
from bilicourseai.transcripts.transcript import lines_for_range
from bilicourseai.visual.timing import candidate_timestamps_from_payload as _candidate_timestamps_from_payload


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
    windows = transcript_windows(report, window_seconds=window_seconds)
    visual_requests: list[VisualRequest] = []
    request_index = 1

    for part in report.parts:
        part.blocks = []

    part_block_counts: dict[int, int] = {}
    for window in windows:
        payload = await _segment_window(client, settings, report, window)
        if request_delay > 0:
            await asyncio.sleep(request_delay)

        part = find_report_part(report, int(window["part_page"]))
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
                    candidate_timestamps=_candidate_timestamps_from_payload(visual, start, end, timestamp),
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
            "For each visual request, provide candidate_timestamps: 3-7 precise seconds inside the block where the relevant visual may be most complete. Choose them semantically from the transcript and topic flow; do not use mechanical offsets like t-8,t-4,t,t+4,t+8.",
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
                            "candidate_timestamps": ["number, 3-7 semantic candidate seconds inside the block"],
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
