from __future__ import annotations

from bilicourseai.models import KnowledgeBlock


SHORT_BLOCK_AS_LEAF_SECONDS = 600
MAX_LEAF_SECONDS = 600
STRONGLY_EXPAND_SECONDS = 720


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
