from __future__ import annotations

from collections.abc import Iterable

from bilicourseai.models import KnowledgeBlock, VideoReport


def iter_blocks(blocks: Iterable[KnowledgeBlock]) -> Iterable[KnowledgeBlock]:
    for block in blocks:
        yield block
        yield from iter_blocks(block.children)


def find_block(report: VideoReport, block_id: str) -> KnowledgeBlock | None:
    for part in report.parts:
        for block in iter_blocks(part.blocks):
            if block.id == block_id:
                return block
    return None


def find_parent_blocks(report: VideoReport, block_id: str) -> list[KnowledgeBlock] | None:
    def visit(blocks: list[KnowledgeBlock], parents: list[KnowledgeBlock]) -> list[KnowledgeBlock] | None:
        for block in blocks:
            if block.id == block_id:
                return parents
            found = visit(block.children, [*parents, block])
            if found is not None:
                return found
        return None

    for part in report.parts:
        found = visit(part.blocks, [])
        if found is not None:
            return found
    return None


def next_child_id(parent: KnowledgeBlock, prefix: str = "n") -> str:
    base = parent.id
    existing = {child.id for child in parent.children}
    index = 1
    while f"{base}-{prefix}{index}" in existing:
        index += 1
    return f"{base}-{prefix}{index}"
