from __future__ import annotations

from typing import Any

from bilicourseai.models import KnowledgeBlock, TranscriptLine, VideoPart
from bilicourseai.transcripts.transcript import lines_for_range, transcript_items_limited


BOUNDARY_REVIEW_MAX_CHARS = 32000


def boundary_review_payload_for_blocks(
    blocks: list[KnowledgeBlock],
    transcript: list[TranscriptLine],
    *,
    context_start: float | None = None,
    context_end: float | None = None,
) -> dict[str, Any]:
    boundaries: list[dict[str, Any]] = []
    if len(blocks) < 2 or not transcript:
        return {"nodes": [], "boundaries": []}

    transcript_start = transcript[0].start if context_start is None else context_start
    transcript_end = transcript[-1].end if context_end is None else context_end
    per_boundary_budget = max(800, BOUNDARY_REVIEW_MAX_CHARS // max(1, len(blocks) - 1))
    for index in range(len(blocks) - 1):
        left = blocks[index]
        right = blocks[index + 1]
        lookahead = blocks[index + 2] if index + 2 < len(blocks) else None
        boundary = right.start
        local_start = max(transcript_start, left.start)
        local_end = min(transcript_end, (lookahead.end if lookahead else right.end))
        context_lines = lines_for_range(
            transcript,
            max(transcript_start, local_start - 10.0),
            min(transcript_end, local_end + 10.0),
        )
        boundaries.append(
            {
                "index": index + 1,
                "left_id": left.id,
                "left_title": left.title,
                "right_id": right.id,
                "right_title": right.title,
                "lookahead": (
                    {
                        "id": lookahead.id,
                        "title": lookahead.title,
                        "start": lookahead.start,
                        "end": lookahead.end,
                        "summary": lookahead.summary,
                    }
                    if lookahead
                    else None
                ),
                "current_boundary": boundary,
                "allowed_min": left.start + 5.0,
                "allowed_max": right.end - 5.0,
                "context": transcript_items_limited(context_lines, per_boundary_budget),
            }
        )

    return {
        "nodes": [
            {
                "id": block.id,
                "title": block.title,
                "start": block.start,
                "end": block.end,
                "summary": block.summary,
            }
            for block in blocks
        ],
        "boundaries": boundaries,
    }


def boundary_review_payload(part: VideoPart) -> dict[str, Any]:
    return boundary_review_payload_for_blocks(part.blocks, part.transcript)


def apply_boundary_adjustments_to_blocks(
    blocks: list[KnowledgeBlock],
    transcript: list[TranscriptLine],
    adjustments: list[dict[str, Any]],
) -> int:
    if len(blocks) < 2 or not transcript:
        return 0
    changed = 0
    by_index = {
        int(item.get("index")): item
        for item in adjustments
        if isinstance(item, dict) and str(item.get("decision") or "").strip() == "adjust"
    }

    for index in range(1, len(blocks)):
        item = by_index.get(index)
        if not item:
            continue
        left = blocks[index - 1]
        right = blocks[index]
        current = right.start
        try:
            new_boundary = float(item.get("new_boundary"))
        except (TypeError, ValueError):
            continue
        lower = left.start + 5.0
        upper = right.end - 5.0
        new_boundary = max(lower, min(upper, new_boundary))
        if abs(new_boundary - current) < 3.0:
            continue

        left.end = new_boundary
        right.start = new_boundary
        left.transcript = lines_for_range(transcript, left.start, left.end)
        right.transcript = lines_for_range(transcript, right.start, right.end)
        left.boundary_confidence = str(item.get("confidence") or left.boundary_confidence or "").strip()
        right.boundary_confidence = str(item.get("confidence") or right.boundary_confidence or "").strip()
        reason = str(item.get("reason") or "").strip()
        if reason:
            note = f"边界复核：{reason}"
            right.expand_reason = f"{right.expand_reason}；{note}" if right.expand_reason else note
        changed += 1
    return changed


def apply_boundary_adjustments(part: VideoPart, adjustments: list[dict[str, Any]]) -> int:
    return apply_boundary_adjustments_to_blocks(part.blocks, part.transcript, adjustments)
