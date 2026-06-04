from __future__ import annotations

from bilicourseai.outline.boundaries import (
    apply_boundary_adjustments,
    apply_boundary_adjustments_to_blocks,
    boundary_review_payload,
    boundary_review_payload_for_blocks,
)
from bilicourseai.outline.normalization import normalize_outline_nodes, quality_fields
from bilicourseai.outline.quality import (
    MAX_LEAF_SECONDS,
    SHORT_BLOCK_AS_LEAF_SECONDS,
    STRONGLY_EXPAND_SECONDS,
    apply_outline_quality_gate,
    merge_short_adjacent_nodes,
)
from bilicourseai.outline.windows import (
    DIRECT_PART_OUTLINE_SECONDS,
    SHORT_PART_AS_LEAF_SECONDS,
    part_duration,
    root_windows,
)

__all__ = [
    "DIRECT_PART_OUTLINE_SECONDS",
    "MAX_LEAF_SECONDS",
    "SHORT_BLOCK_AS_LEAF_SECONDS",
    "SHORT_PART_AS_LEAF_SECONDS",
    "STRONGLY_EXPAND_SECONDS",
    "apply_boundary_adjustments",
    "apply_boundary_adjustments_to_blocks",
    "apply_outline_quality_gate",
    "boundary_review_payload",
    "boundary_review_payload_for_blocks",
    "merge_short_adjacent_nodes",
    "normalize_outline_nodes",
    "part_duration",
    "quality_fields",
    "root_windows",
]
