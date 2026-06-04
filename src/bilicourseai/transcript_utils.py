from __future__ import annotations

from typing import Any

from bilicourseai.models import TranscriptLine


def transcript_char_count(lines: list[TranscriptLine]) -> int:
    return sum(len(line.text) for line in lines)


def lines_for_range(lines: list[TranscriptLine], start: float, end: float) -> list[TranscriptLine]:
    return [line for line in lines if line.end > start and line.start < end]


def transcript_items(lines: list[TranscriptLine]) -> list[dict[str, Any]]:
    return [{"start": line.start, "end": line.end, "text": line.text} for line in lines]


def transcript_items_limited(
    lines: list[TranscriptLine],
    max_chars: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used = 0
    for line in lines:
        text = line.text
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 3)].rstrip() + "..."
        items.append({"start": line.start, "end": line.end, "text": text})
        used += len(text)
    return items


def compact_transcript_slices(
    lines: list[TranscriptLine],
    slice_seconds: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not lines:
        return []

    raw_slices: list[dict[str, Any]] = []
    slice_start = lines[0].start
    current: list[TranscriptLine] = []
    for line in lines:
        if current and line.start - slice_start >= slice_seconds:
            raw_slices.append(
                {
                    "start": current[0].start,
                    "end": current[-1].end,
                    "text": "".join(item.text for item in current),
                }
            )
            current = []
            slice_start = line.start
        current.append(line)

    if current:
        raw_slices.append(
            {
                "start": current[0].start,
                "end": current[-1].end,
                "text": "".join(item.text for item in current),
            }
        )

    if not raw_slices:
        return []

    per_slice_budget = max(180, max_chars // len(raw_slices) - 20)
    slices: list[dict[str, Any]] = []
    for item in raw_slices:
        text = str(item["text"]).strip()
        if len(text) > per_slice_budget:
            text = text[: per_slice_budget - 3].rstrip() + "..."
        slices.append({**item, "text": text})
    return slices


def transcript_prompt_payload(
    lines: list[TranscriptLine],
    max_chars: int,
    slice_seconds: int,
) -> tuple[str, list[dict[str, Any]]]:
    if transcript_char_count(lines) <= max_chars:
        return "full_transcript", transcript_items(lines)
    return "time_slices", compact_transcript_slices(
        lines,
        slice_seconds=slice_seconds,
        max_chars=max_chars,
    )
