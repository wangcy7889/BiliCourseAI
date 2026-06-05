from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from bilicourseai.models import KnowledgeBlock, VideoReport, VisualRequest
from bilicourseai.settings import LLMSettings
from bilicourseai.source import capture_candidate_frames, capture_requested_frames, cleanup_candidate_frames
from bilicourseai.visual.llm import analyze_frames, choose_visual_frames


ProgressCallback = Callable[[str], None]
MAX_VISUAL_CHOICE_ROUNDS = 3


def _walk_blocks(blocks: list[KnowledgeBlock]):
    for block in blocks:
        yield block
        yield from _walk_blocks(block.children)


def _drop_visual_request_refs(report: VideoReport, request_ids: set[str]) -> None:
    if not request_ids:
        return
    for part in report.parts:
        for block in _walk_blocks(part.blocks):
            block.visual_requests = [request for request in block.visual_requests if request.id not in request_ids]
            block.frames = [frame for frame in block.frames if frame.request_id not in request_ids]
            block.visual_analyses = [
                analysis for analysis in block.visual_analyses if analysis.request_id not in request_ids
            ]
            for section in block.sections:
                section.visual_requests = [
                    request for request in section.visual_requests if request.id not in request_ids
                ]
                section.frames = [frame for frame in section.frames if frame.request_id not in request_ids]
                section.visual_analyses = [
                    analysis for analysis in section.visual_analyses if analysis.request_id not in request_ids
                ]


def _append_unique_text(base: str, addition: str) -> str:
    base = base.strip()
    addition = addition.strip()
    if not addition or addition in base:
        return base
    if not base:
        return addition
    return f"{base}\n\n补充观察方向：{addition}"


def _sync_visual_request(report: VideoReport, merged_request: VisualRequest) -> None:
    for part in report.parts:
        for block in _walk_blocks(part.blocks):
            for index, request in enumerate(block.visual_requests):
                if request.id == merged_request.id:
                    block.visual_requests[index] = merged_request
            for section in block.sections:
                for index, request in enumerate(section.visual_requests):
                    if request.id == merged_request.id:
                        section.visual_requests[index] = merged_request


def _merge_duplicate_final_visual_requests(report: VideoReport, requests: list[VisualRequest]) -> tuple[list[VisualRequest], int]:
    seen: dict[tuple[str, int, float], VisualRequest] = {}
    kept: list[VisualRequest] = []
    dropped_ids: set[str] = set()
    for request in requests:
        key = (request.block_id, request.part_page, round(request.timestamp, 3))
        existing = seen.get(key)
        if existing is not None:
            existing.reason = _append_unique_text(existing.reason, request.reason)
            existing.prompt = _append_unique_text(existing.prompt, request.prompt)
            _sync_visual_request(report, existing)
            dropped_ids.add(request.id)
            continue
        seen[key] = request
        kept.append(request)
    if dropped_ids:
        _drop_visual_request_refs(report, dropped_ids)
        report.llm_notes.append(
            f"Merged {len(dropped_ids)} duplicate final visual request(s) into existing frame analysis directions."
        )
    return kept, len(dropped_ids)


async def run_visual_pipeline(
    report: VideoReport,
    requests: list[VisualRequest],
    settings: LLMSettings,
    output_dir: Path,
    *,
    report_dir: Path | None = None,
    prefer_stream_frames: bool = True,
    request_delay: float = 0.0,
    progress: ProgressCallback | None = None,
    max_choice_rounds: int = MAX_VISUAL_CHOICE_ROUNDS,
) -> dict[str, Any]:
    if not isinstance(report, VideoReport):
        raise TypeError(f"run_visual_pipeline expected VideoReport for report, got {type(report).__name__}")
    if not isinstance(requests, list):
        raise TypeError(f"run_visual_pipeline expected list for requests, got {type(requests).__name__}")

    frames_count = 0
    analyses_count = 0
    vision_skipped = False

    has_vision_client = bool(
        settings.vision_model and settings.effective_vision_base_url and settings.effective_vision_api_key
    )

    if requests and not has_vision_client:
        vision_skipped = True
        if progress:
            progress("Visual requests skipped: vision model/base_url/api_key is not configured.")
    elif requests:
        if progress:
            progress(f"Visual requests: {len(requests)}")
        chosen_requests: list[VisualRequest] = []
        pending_requests = requests
        max_choice_rounds = max(1, max_choice_rounds)
        for round_index in range(1, max_choice_rounds + 1):
            if progress:
                progress(f"Visual candidate round {round_index}/{max_choice_rounds}: {len(pending_requests)} request(s)")
            candidates = await capture_candidate_frames(
                report,
                pending_requests,
                output_dir,
                prefer_stream=prefer_stream_frames,
                report_dir=report_dir,
            )
            round_choices = await choose_visual_frames(
                report,
                settings,
                pending_requests,
                candidates,
                request_delay=request_delay,
            )
            retry_requests = [request for request in round_choices if request.candidate_timestamps]
            stable_requests = [request for request in round_choices if not request.candidate_timestamps]
            chosen_requests.extend(stable_requests)
            if not retry_requests:
                break
            if round_index >= max_choice_rounds:
                report.llm_notes.append(
                    f"Visual frame choice stopped after {max_choice_rounds} round(s); "
                    f"using {len(retry_requests)} latest retry request(s) as final timestamps."
                )
                chosen_requests.extend(
                    VisualRequest(
                        id=request.id,
                        part_page=request.part_page,
                        block_id=request.block_id,
                        timestamp=request.timestamp,
                        candidate_timestamps=[],
                        reason=request.reason,
                        prompt=request.prompt,
                        section_id=request.section_id,
                    )
                    for request in retry_requests
                )
                break
            if progress:
                progress(f"Visual retries requested: {len(retry_requests)}")
            pending_requests = retry_requests

        chosen_requests, duplicate_count = _merge_duplicate_final_visual_requests(report, chosen_requests)
        if progress and duplicate_count:
            progress(f"Duplicate final frames skipped: {duplicate_count}")

        frames = await capture_requested_frames(
            report,
            chosen_requests,
            output_dir,
            prefer_stream=prefer_stream_frames,
            report_dir=report_dir,
        )
        ok_frames = [frame for frame in frames if not frame.error]
        frames_count = len(ok_frames)
        if progress:
            progress(f"Final frames saved: {len(ok_frames)}/{len(frames)}")
        analyses = await analyze_frames(report, settings, ok_frames, request_delay=request_delay)
        analyses_count = len(analyses)
        if progress:
            progress(f"Vision analyses: {len(analyses)}")
        if cleanup_candidate_frames(report, output_dir, report_dir=report_dir) and progress:
            progress("Candidate frames cleaned.")

    return {
        "visual_requests": len(requests),
        "frames": frames_count,
        "analyses": analyses_count,
        "vision_skipped": vision_skipped,
    }
