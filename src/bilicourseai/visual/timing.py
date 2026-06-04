from __future__ import annotations

from typing import Any


def visual_candidate_timestamps(start: float, end: float, timestamp: float) -> list[float]:
    duration = max(0.0, end - start)
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
    values = [max(start, min(end, value)) for value in [timestamp, *anchors]]
    return sorted({round(value, 3) for value in values})


def candidate_timestamps_from_payload(
    visual: dict[str, Any],
    start: float,
    end: float,
    timestamp: float,
) -> list[float]:
    raw_values = visual.get("candidate_timestamps") or visual.get("candidate_times") or []
    candidates: list[float] = []
    if isinstance(raw_values, list):
        for value in raw_values:
            try:
                candidates.append(float(value))
            except (TypeError, ValueError):
                continue
    candidates.append(timestamp)
    cleaned = sorted({round(max(start, min(end, value)), 3) for value in candidates})
    if len(cleaned) >= 2:
        return cleaned[:7]
    return visual_candidate_timestamps(start, end, timestamp)
