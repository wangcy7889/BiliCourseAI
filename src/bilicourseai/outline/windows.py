from __future__ import annotations

from typing import Any

from bilicourseai.models import TranscriptLine, VideoReport
from bilicourseai.transcripts.transcript import lines_for_range


def part_duration(part) -> float:
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
    if part_duration(part) <= target_seconds:
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


def root_windows(
    report: VideoReport,
    outline_window_seconds: int,
    outline_overlap_seconds: int,
    short_part_as_leaf_seconds: int = SHORT_PART_AS_LEAF_SECONDS,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for part in report.parts:
        if not part.transcript:
            continue
        if len(report.parts) > 1 and part_duration(part) <= short_part_as_leaf_seconds:
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
