from __future__ import annotations

from typing import Any

from bilicourseai.boundary_review import boundary_review_payload, boundary_review_payload_for_blocks
from bilicourseai.node_quality import MAX_LEAF_SECONDS
from bilicourseai.transcript_utils import transcript_prompt_payload


GLOBAL_PLAN_MAX_CHARS = 56000
GLOBAL_PLAN_SLICE_SECONDS = 120
DIRECT_PART_OUTLINE_MAX_CHARS = 64000


def _duration(part) -> float:
    if part.duration is not None:
        return float(part.duration)
    if part.transcript:
        return float(part.transcript[-1].end)
    return 0.0


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def suggest_root_node_count(part, part_plan: dict[str, Any] | None = None) -> int:
    if part_plan:
        planned = _safe_int(part_plan.get("suggested_root_nodes"), 0)
        if planned > 0:
            return _clamp_int(planned, 1, 16)

    duration = _duration(part)
    if duration <= 600:
        return 1
    if duration <= 1200:
        return 3
    if duration <= 2400:
        return _clamp_int(round(duration / 420), 4, 8)
    if duration <= 4200:
        return _clamp_int(round(duration / 540), 6, 11)
    return _clamp_int(round(duration / 600), 8, 16)


def root_node_count_guidance(part, part_plan: dict[str, Any] | None = None) -> dict[str, int]:
    suggested = suggest_root_node_count(part, part_plan=part_plan)
    if suggested <= 1:
        return {"suggested": 1, "minimum": 1, "maximum": 1}

    duration = _duration(part)
    minimum = max(2, suggested - 2)
    maximum = min(16, suggested + 4)
    if duration >= 5400:
        maximum = max(maximum, 14)
    return {"suggested": suggested, "minimum": minimum, "maximum": maximum}


def compact_part_plan(part_plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not part_plan:
        return None
    return {
        "suggested_root_nodes": part_plan.get("suggested_root_nodes"),
        "course_map": part_plan.get("course_map", [])[:20],
        "boundary_hints": part_plan.get("boundary_hints", [])[:20],
        "notes": str(part_plan.get("notes") or "")[:1200],
    }


def outline_window_prompt(report, window: dict[str, Any], part_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "video_title": report.title,
        "part": {"page": window["part_page"], "title": window["part_title"]},
        "part_plan": compact_part_plan(part_plan),
        "window": {
            "start": window["start"],
            "end": window["end"],
            "context_start": window.get("context_start", window["start"]),
            "context_end": window.get("context_end", window["end"]),
        },
        "task": (
            "Map this internal processing window into candidate knowledge topics. "
            "These candidates are intermediate evidence for a later part-level reduce step, not the final visible report tree."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Do not preserve or mention processing-window structure.",
            "Do not write full notes, do not request images, and do not analyze visuals.",
            "Extract meaningful candidate topics from this window; prefer 2-7 candidates according to actual content density.",
            "Avoid overly fine operational steps unless they are genuine concepts a learner may expand.",
            "Use part_plan as global orientation when present, but do not force every planned theme into this window.",
            "The transcript may include overlap context before/after the main window. Use it only to avoid broken boundaries.",
            "Each candidate timestamp must stay inside the main window start/end, not merely the overlap context.",
            "Use stable topic names that can be merged with neighboring-window candidates.",
        ],
        "schema": {
            "candidates": [
                {
                    "title": "string",
                    "start": "number",
                    "end": "number",
                    "summary": "short Markdown string",
                    "merge_hint": "short stable key such as entity_resolution or pipeline_setup",
                    "importance": "high|medium|low",
                }
            ],
        },
        "transcript": window["transcript"],
    }


def plan_part_outline_prompt(report, part) -> dict[str, Any]:
    transcript_kind, transcript = transcript_prompt_payload(
        part.transcript,
        max_chars=GLOBAL_PLAN_MAX_CHARS,
        slice_seconds=GLOBAL_PLAN_SLICE_SECONDS,
    )
    guidance = root_node_count_guidance(part)
    return {
        "video_title": report.title,
        "part": {
            "page": part.page,
            "title": part.title,
            "duration": part.duration,
            "transcript_kind": transcript_kind,
        },
        "task": (
            "Create a lightweight global course map for this Bilibili part before local window analysis. "
            "This is not the final report. It should guide later boundary decisions and merging."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Do not write study notes.",
            "Do not force exactly six themes. Choose a natural number based on the course structure.",
            "Prefer fewer themes for short coherent lessons and more themes for long dense lessons.",
            "Use time ranges as approximate guideposts; later passes may refine them.",
            "Identify transition/setup/review ranges so they can be merged instead of becoming visible nodes.",
        ],
        "node_count_guidance": guidance,
        "schema": {
            "suggested_root_nodes": "number",
            "course_map": [
                {
                    "title": "string",
                    "start": "number",
                    "end": "number",
                    "role": "main_topic|setup|transition|example|summary",
                    "summary": "short Markdown string",
                }
            ],
            "boundary_hints": [
                {
                    "timestamp": "number",
                    "reason": "short Chinese string",
                    "confidence": "high|medium|low",
                }
            ],
            "notes": "short Chinese string",
        },
        "transcript": transcript,
    }


def outline_whole_part_prompt(report, part, guidance: dict[str, int]) -> dict[str, Any]:
    transcript_kind, transcript = transcript_prompt_payload(
        part.transcript,
        max_chars=DIRECT_PART_OUTLINE_MAX_CHARS,
        slice_seconds=GLOBAL_PLAN_SLICE_SECONDS,
    )
    return {
        "video_title": report.title,
        "part": {
            "page": part.page,
            "title": part.title,
            "duration": part.duration,
            "transcript_kind": transcript_kind,
        },
        "task": (
            "Analyze this whole Bilibili part at once and create the visible first-layer knowledge tree. "
            "Use the global context to avoid mechanical time-window splits."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Do not write full notes and do not request images at outline stage.",
            "Do not force exactly six nodes. The number of nodes should follow the actual course structure.",
            "Use the suggested node count as a soft target, not a hard limit.",
            "Create more nodes when the lesson contains many distinct concepts, grammar rules, algorithms, examples, or phases.",
            "Create fewer nodes when the lesson is a single coherent explanation.",
            "Each node should represent a complete learning topic, not merely a processing chunk or one tiny operation.",
            "The returned nodes should cover the full part timeline from start to end with no large unexplained gaps.",
            "If a time range is transitional or setup, merge it into the nearest meaningful node instead of omitting it.",
            "Use node_type='branch' unless the node is already narrow and self-contained.",
            f"Never mark a node longer than {MAX_LEAF_SECONDS} seconds as leaf.",
            "For each node, set should_expand=true when it contains multiple concepts, multiple examples, or a multi-step algorithm/proof.",
            "For coarse nodes, provide split_hints that name likely child topics.",
        ],
        "node_count_guidance": guidance,
        "schema": {
            "nodes": [
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
        },
        "transcript": transcript,
    }


def boundary_review_prompt(report, part) -> dict[str, Any]:
    review = boundary_review_payload(part)
    return {
        "video_title": report.title,
        "part": {
            "page": part.page,
            "title": part.title,
            "duration": part.duration,
        },
        "task": (
            "Review only the boundaries between adjacent course-outline nodes. "
            "Do not rename nodes, do not merge nodes, and do not create new nodes. "
            "Move a boundary only when the current boundary clearly assigns transcript content to the wrong adjacent topic."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Prefer keeping the current boundary unless the local transcript clearly shows it is wrong.",
            "A good boundary is usually just before the first sentence that belongs to the right node's topic.",
            "Use the lookahead node to detect whether the right node starts too early and is eating the left node's topic.",
            "For each boundary, reason over the three-node neighborhood: left node, right node, and lookahead node when present.",
            "Do not move boundaries merely to make durations prettier.",
            "Do not make large moves unless the context clearly shows the topic change happens there.",
            "If the transcript around a boundary is ambiguous, keep it.",
            "new_boundary must stay within allowed_min and allowed_max.",
        ],
        "schema": {
            "boundaries": [
                {
                    "index": "number copied from input",
                    "decision": "keep|adjust",
                    "new_boundary": "number, only if adjust",
                    "confidence": "high|medium|low",
                    "reason": "short Chinese reason",
                }
            ],
        },
        "review_input": review,
    }


def child_boundary_review_prompt(report, part, parent, children) -> dict[str, Any]:
    review = boundary_review_payload_for_blocks(
        children,
        part.transcript,
        context_start=parent.start,
        context_end=parent.end,
    )
    return {
        "video_title": report.title,
        "part": {
            "page": part.page,
            "title": part.title,
        },
        "parent_node": {
            "id": parent.id,
            "title": parent.title,
            "start": parent.start,
            "end": parent.end,
            "summary": parent.summary,
        },
        "task": (
            "Review only the boundaries between adjacent child nodes under this parent course-outline node. "
            "Do not rename nodes, do not merge nodes, and do not create new nodes. "
            "Move a boundary only when the local transcript clearly assigns content to the wrong adjacent child topic."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Prefer keeping the current boundary unless the local transcript clearly shows it is wrong.",
            "A good boundary is usually just before the first sentence that belongs to the right child node's topic.",
            "For each boundary, reason over the three-node neighborhood: left child, right child, and lookahead child when present.",
            "Do not move boundaries merely to make durations prettier.",
            "Do not make large moves unless the context clearly shows the topic change happens there.",
            "If the transcript around a boundary is ambiguous, keep it.",
            "new_boundary must stay within allowed_min and allowed_max.",
        ],
        "schema": {
            "boundaries": [
                {
                    "index": "number copied from input",
                    "decision": "keep|adjust",
                    "new_boundary": "number, only if adjust",
                    "confidence": "high|medium|low",
                    "reason": "short Chinese reason",
                }
            ],
        },
        "review_input": review,
    }


def reduce_part_outline_prompt(
    report,
    part,
    candidates: list[dict[str, Any]],
    *,
    part_plan: dict[str, Any] | None = None,
    guidance: dict[str, int] | None = None,
) -> dict[str, Any]:
    if guidance is None:
        guidance = root_node_count_guidance(part, part_plan=part_plan)
    return {
        "video_title": report.title,
        "part": {
            "page": part.page,
            "title": part.title,
            "duration": part.duration,
        },
        "part_plan": compact_part_plan(part_plan),
        "task": (
            "Reduce window-level candidate topics into the visible part-level knowledge tree. "
            "The output will be shown to learners as the first layer under this Bilibili part."
        ),
        "rules": [
            "Return valid JSON only.",
            "Use Chinese unless a source term must remain English.",
            "Do not preserve processing-window boundaries such as 片段1/片段2/片段3.",
            "Do not create a chronological laundry list of tiny steps.",
            "Merge adjacent or duplicate candidates into coherent learning themes.",
            "Do not force exactly six nodes. Use the node_count_guidance as a soft target, not a hard limit.",
            "Create as many nodes as the actual course structure needs within the guidance range.",
            "If the part is long and dense, it is better to return 8-14 coherent nodes than to squeeze unrelated topics together.",
            "Each node should cover a complete learning phase, not merely one code line or one API call.",
            "The returned nodes should cover the full part timeline from start to end with no large unexplained gaps.",
            "If a time range is transitional or setup, merge it into the nearest meaningful node instead of omitting it.",
            "Use original timestamps spanning the merged candidate range.",
            "Set expandable=true for nodes that should be expanded into subtopics or leaf notes later.",
            "Use node_type='branch' unless the node is already narrow and self-contained.",
            f"Never mark a node longer than {MAX_LEAF_SECONDS} seconds as leaf.",
            "For each node, set should_expand=true when it contains multiple concepts, multiple examples, or a multi-step algorithm/proof.",
            "For coarse nodes, provide split_hints that name likely child topics.",
        ],
        "schema": {
            "nodes": [
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
        },
        "node_count_guidance": guidance,
        "candidates": candidates,
    }
