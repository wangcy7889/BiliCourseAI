from __future__ import annotations

from bilicourseai.models import KnowledgeBlock, VideoPart, VideoReport


def _part_start(part: VideoPart) -> float:
    if part.transcript:
        return float(part.transcript[0].start)
    return 0.0


def _part_end(part: VideoPart) -> float:
    if part.transcript:
        return float(part.transcript[-1].end)
    return float(part.duration or 0)


def build_part_outline_tree(report: VideoReport) -> None:
    """Create a no-LLM outline where each source part is one expandable leaf."""

    source_parts = [part for part in report.parts if part.page > 0]
    if not source_parts:
        source_parts = list(report.parts)

    if len(source_parts) == 1:
        part = source_parts[0]
        part.blocks = [_part_node_from_existing_blocks(part)]
        report.parts = [part]
        report.llm_notes.append("Generated no-LLM part tree for a single source part.")
        return

    blocks: list[KnowledgeBlock] = []
    for part in source_parts:
        blocks.append(_part_node_from_existing_blocks(part))

    tree_part = VideoPart(
        page=0,
        cid=source_parts[0].cid if source_parts else 0,
        title="分 P 目录",
        duration=sum(int(part.duration or 0) for part in source_parts),
        transcript=[],
        blocks=blocks,
    )
    for part in source_parts:
        part.blocks = []
    report.parts = [tree_part, *source_parts]
    report.llm_notes.append(f"Generated no-LLM part tree from {len(source_parts)} source parts.")


def _base_part_node(part: VideoPart) -> KnowledgeBlock:
    return KnowledgeBlock(
        id=f"part-p{part.page}",
        title=f"P{part.page} {part.title}".strip(),
        start=_part_start(part),
        end=_part_end(part),
        summary="按分 P 创建的初始节点。点击展开后，仅针对这个分 P 调用模型继续细分或生成笔记。",
        node_type="leaf",
        status="skeleton",
        expandable=True,
        depth=0,
        source_part_page=part.page,
        transcript=part.transcript,
    )


def _copy_block_into_part_node(part: VideoPart, block: KnowledgeBlock) -> KnowledgeBlock:
    block.id = f"part-p{part.page}"
    bind_block_to_part(block, part.page, depth=0)
    block.depth = 0
    return block


def bind_block_to_part(block: KnowledgeBlock, page: int, depth: int | None = None) -> None:
    block.source_part_page = page
    if depth is not None:
        block.depth = depth
    for request in block.visual_requests:
        request.part_page = page
    for frame in block.frames:
        frame.part_page = page
    for analysis in block.visual_analyses:
        analysis.part_page = page
    for section in block.sections:
        for request in section.visual_requests:
            request.part_page = page
        for frame in section.frames:
            frame.part_page = page
        for analysis in section.visual_analyses:
            analysis.part_page = page
    for child in block.children:
        bind_block_to_part(child, page, None if depth is None else depth + 1)


def _part_node_from_existing_blocks(part: VideoPart) -> KnowledgeBlock:
    if not part.blocks:
        return _base_part_node(part)
    if len(part.blocks) == 1:
        return _copy_block_into_part_node(part, part.blocks[0])

    node = _base_part_node(part)
    node.node_type = "branch"
    node.status = "expanded"
    node.expandable = False
    node.should_expand = True
    node.summary = f"{node.title} 已有 {len(part.blocks)} 个子节点。"
    node.children = part.blocks
    for child in node.children:
        bind_block_to_part(child, part.page, depth=max(1, child.depth))
    return node


def find_part_outline_node(report: VideoReport, page: int) -> KnowledgeBlock | None:
    tree_part = next((part for part in report.parts if part.page == 0 and part.title == "分 P 目录"), None)
    if tree_part is None:
        return None
    expected_id = f"part-p{page}"
    for block in tree_part.blocks:
        if block.id == expected_id or block.source_part_page == page:
            return block
    return None


def is_part_outline_root_node(report: VideoReport, block: KnowledgeBlock) -> bool:
    if block.source_part_page is None:
        return False
    return find_part_outline_node(report, block.source_part_page) is block
