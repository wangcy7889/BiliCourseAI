from __future__ import annotations

import copy

from bilicourseai.models import KnowledgeBlock


SHORT_BLOCK_AS_LEAF_SECONDS = 600
MAX_LEAF_SECONDS = 600
STRONGLY_EXPAND_SECONDS = 720
MIN_STANDALONE_NODE_SECONDS = 75


_SHORT_NODE_KEEP_KEYWORDS = (
    "例题",
    "习题",
    "真题",
    "题目",
    "考题",
    "题解析",
    "公式",
    "推导",
    "证明",
    "定理",
    "引理",
    "算法",
    "流程图",
    "完整流程",
    "算法流程",
    "操作流程",
    "步骤解析",
    "路径推导",
    "实战",
    "案例",
    "实验",
)


def _merged_title(left: str, right: str) -> str:
    parts: list[str] = []
    for title in [left, right]:
        for part in title.split("、"):
            part = part.strip().removesuffix("等")
            if part and part not in parts:
                parts.append(part)
    if len(parts) <= 3:
        return "、".join(parts)
    return f"{'、'.join(parts[:3])}等"


def apply_outline_quality_gate(block: KnowledgeBlock) -> None:
    duration = block.end - block.start
    reasons: list[str] = []

    if duration > MAX_LEAF_SECONDS:
        if block.node_type == "leaf":
            reasons.append(f"节点时长 {duration:.0f}s 超过 {MAX_LEAF_SECONDS}s，自动改为可展开分支。")
        block.node_type = "branch"
        block.expandable = True
        block.should_expand = True
        if not block.granularity:
            block.granularity = "coarse" if duration > STRONGLY_EXPAND_SECONDS else "medium"

    if duration > STRONGLY_EXPAND_SECONDS:
        block.node_type = "branch"
        block.expandable = True
        block.should_expand = True
        block.granularity = "coarse"
        if not block.split_hints:
            block.split_hints = ["按关键概念、例题步骤或算法阶段继续展开。"]
        reasons.append(f"节点时长 {duration:.0f}s，建议继续拆成子节点。")

    if block.node_type == "branch":
        block.expandable = True

    if block.should_expand and not block.expandable:
        block.expandable = True

    if reasons:
        existing = block.expand_reason.strip()
        merged = "；".join(reasons)
        block.expand_reason = f"{existing}；{merged}" if existing else merged


def _short_node_should_stand_alone(block: KnowledgeBlock) -> bool:
    text = " ".join(
        [
            block.title,
            block.expand_reason,
            " ".join(block.split_hints),
        ]
    )
    return any(keyword in text for keyword in _SHORT_NODE_KEEP_KEYWORDS)


def _merge_pair(left: KnowledgeBlock, right: KnowledgeBlock) -> KnowledgeBlock:
    left.end = max(left.end, right.end)
    if right.title and right.title not in left.title:
        left.title = _merged_title(left.title, right.title)
    if right.summary:
        left.summary = "\n\n".join(part for part in [left.summary, right.summary] if part)
    left.node_type = "branch" if "branch" in {left.node_type, right.node_type} else "leaf"
    left.expandable = left.expandable or right.expandable or left.node_type == "branch"
    left.should_expand = left.should_expand or right.should_expand
    if not left.granularity or right.granularity == "coarse":
        left.granularity = right.granularity or left.granularity
    if right.expand_reason:
        left.expand_reason = "；".join(part for part in [left.expand_reason, right.expand_reason] if part)
    if right.boundary_confidence and not left.boundary_confidence:
        left.boundary_confidence = right.boundary_confidence
    left.split_hints.extend(hint for hint in right.split_hints if hint not in left.split_hints)
    left.key_points.extend(point for point in right.key_points if point not in left.key_points)
    left.transcript.extend(right.transcript)
    apply_outline_quality_gate(left)
    return left


def merge_short_adjacent_nodes(nodes: list[KnowledgeBlock]) -> int:
    """Merge ordinary short sibling nodes while keeping examples/proofs/flows intact."""

    if len(nodes) < 2:
        return 0

    original_nodes = copy.deepcopy(nodes)
    runs: list[KnowledgeBlock] = []
    merge_count = 0
    index = 0
    while index < len(nodes):
        node = nodes[index]
        duration = node.end - node.start
        is_mergeable = duration < MIN_STANDALONE_NODE_SECONDS and not _short_node_should_stand_alone(node)
        if not is_mergeable:
            runs.append(node)
            index += 1
            continue

        group = node
        index += 1
        while index < len(nodes):
            next_node = nodes[index]
            next_is_mergeable = (
                next_node.end - next_node.start < MIN_STANDALONE_NODE_SECONDS
                and not _short_node_should_stand_alone(next_node)
            )
            if not next_is_mergeable:
                break
            _merge_pair(group, next_node)
            merge_count += 1
            index += 1
        runs.append(group)

    if len(runs) > 1:
        index = 0
        while index < len(runs):
            node = runs[index]
            duration = node.end - node.start
            if duration < MIN_STANDALONE_NODE_SECONDS and not _short_node_should_stand_alone(node):
                if index > 0:
                    _merge_pair(runs[index - 1], runs.pop(index))
                    merge_count += 1
                    continue
                if index + 1 < len(runs):
                    _merge_pair(node, runs.pop(index + 1))
                    merge_count += 1
                    continue
            index += 1

    if merge_count:
        if len(runs) == 1:
            nodes[:] = original_nodes
            return 0
        nodes[:] = runs
    return merge_count
