from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from bilicourseai.json_utils import best_effort_json_object as _best_effort_json_object
from bilicourseai.llm_client import create_client as _client, extra_body as _extra_body
from bilicourseai.models import FrameArtifact, KnowledgeBlock, VideoReport, VisualAnalysis, VisualRequest
from bilicourseai.settings import LLMSettings
from bilicourseai.tree import find_block
from bilicourseai.visual.timing import visual_candidate_timestamps


def _find_block(report: VideoReport, block_id: str):
    return find_block(report, block_id)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def analyze_frames(
    report: VideoReport,
    settings: LLMSettings,
    frames: list[FrameArtifact],
    request_delay: float = 0.0,
    concurrency: int = 1,
) -> list[VisualAnalysis]:
    if not frames:
        return []
    if not settings.vision_model:
        report.llm_notes.append("Vision model not configured; frames were saved without visual analysis.")
        return []

    client = _client(settings, role="vision")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def analyze_one(frame: FrameArtifact) -> tuple[VisualAnalysis | None, Any, Any, str | None]:
        if frame.error:
            return None, None, None, None
        block = _find_block(report, frame.block_id)
        if not block:
            return None, None, None, None
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
                "For any visible symbol-like text, such as formulas, variables, code identifiers, table labels, graph nodes, or flowchart paths, transcribe the exact visible symbols before interpreting them.",
                "Subtitles and prior notes may contain noise. If they conflict with the visible image, point out the mismatch and state which reading is better supported. If the image is unclear, say it is unclear and lower confidence.",
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
        async with semaphore:
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
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            except OpenAIError as exc:
                return None, None, None, f"Vision analysis failed for {frame.request_id}: {exc}"
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
        section = None
        if request and request.section_id:
            for item in block.sections:
                if item.id == request.section_id:
                    section = item
                    break
        return analysis, block, section, None

    results = await asyncio.gather(*(analyze_one(frame) for frame in frames))
    analyses: list[VisualAnalysis] = []
    for analysis, block, section, error in results:
        if error:
            report.llm_notes.append(error)
            continue
        if analysis is None or block is None:
            continue
        block.visual_analyses.append(analysis)
        if section is not None:
            section.visual_analyses.append(analysis)
        analyses.append(analysis)

    report.llm_notes.append(f"Vision model analyzed {len(analyses)} frames.")
    return analyses


async def choose_visual_frames(
    report: VideoReport,
    settings: LLMSettings,
    requests: list[VisualRequest],
    candidates_by_request: dict[str, list[FrameArtifact]],
    request_delay: float = 0.0,
    concurrency: int = 1,
) -> list[VisualRequest]:
    if not requests:
        return []
    if not settings.vision_model:
        return requests

    client = _client(settings, role="vision")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def choose_one(request: VisualRequest) -> tuple[VisualRequest | None, str | None]:
        block = _find_block(report, request.block_id)
        candidates = [
            frame
            for frame in candidates_by_request.get(request.id, [])
            if frame.path and not frame.error
        ]
        if not block or not candidates:
            return request, None

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
                "The candidates are semantic guesses, not fixed offsets. If they miss the best moment, retry_timestamp may be anywhere inside the block range; use the transcript/context to estimate a richer moment.",
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

        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=settings.vision_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.1,
                    extra_body=_extra_body(settings),
                )
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
            except OpenAIError as exc:
                return request, f"Visual frame choice failed for {request.id}: {exc}"

        payload = _best_effort_json_object(response.choices[0].message.content or "{}")
        decision = str(payload.get("decision") or "").strip()
        if decision == "skip":
            return None, f"Skipped visual request {request.id}: {payload.get('reason') or ''}"
        if decision == "retry_timestamp":
            timestamp = max(block.start, min(block.end, float(payload.get("timestamp") or request.timestamp)))
            candidate_timestamps = visual_candidate_timestamps(block.start, block.end, timestamp)
        else:
            index = int(payload.get("candidate_index") or 1)
            index = max(1, min(len(candidates), index))
            timestamp = candidates[index - 1].timestamp
            candidate_timestamps = []
        return (
            VisualRequest(
                id=request.id,
                part_page=request.part_page,
                block_id=request.block_id,
                timestamp=timestamp,
                candidate_timestamps=candidate_timestamps,
                reason=request.reason,
                prompt=request.prompt,
                section_id=request.section_id,
            ),
            None,
        )

    results = await asyncio.gather(*(choose_one(request) for request in requests))
    chosen: list[VisualRequest] = []
    for request, note in results:
        if note:
            report.llm_notes.append(note)
        if request is not None:
            chosen.append(request)

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
            "Preserve exact symbols, formulas, code identifiers, graph nodes, and flowchart paths from the visual analysis only when the analysis is confident. If it marks something as unclear or conflicting, keep that uncertainty instead of smoothing it away.",
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
