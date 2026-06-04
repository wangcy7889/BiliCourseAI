from __future__ import annotations

from typing import Any

from bilicourseai.outline.quality import MAX_LEAF_SECONDS, STRONGLY_EXPAND_SECONDS


def truthy(value: Any, default: bool = False) -> bool:
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


def string_list(value: Any, limit: int = 8) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def quality_fields(item: dict[str, Any], duration: float) -> dict[str, Any]:
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
        "should_expand": truthy(item.get("should_expand"), default=duration > MAX_LEAF_SECONDS),
        "expand_reason": str(item.get("expand_reason") or "").strip(),
        "boundary_confidence": str(item.get("boundary_confidence") or "").strip(),
        "split_hints": string_list(item.get("split_hints")),
    }


def normalize_outline_nodes(
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
