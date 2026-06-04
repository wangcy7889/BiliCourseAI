from __future__ import annotations

import re
from collections import OrderedDict

from bilicourseai.models import KnowledgeBlock, VideoPart, VideoReport


QUESTION_TYPES = ("选择题", "填空题", "解答题")


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", value).strip("-").lower() or "group"


def _parse_part_title(title: str) -> tuple[str, str, str]:
    chapter_match = re.search(r"(第[一二三四五六七八九十百零〇两\d]+章)", title)
    chapter = chapter_match.group(1) if chapter_match else "未分章"

    qtype = "题目"
    for candidate in QUESTION_TYPES:
        if candidate in title:
            qtype = candidate
            break

    number_match = re.search(r"第\s*([0-9一二三四五六七八九十百零〇两]+)\s*题", title)
    number = number_match.group(1) if number_match else str(title)
    return chapter, qtype, number


def _part_end(part: VideoPart) -> float:
    if part.transcript:
        return float(part.transcript[-1].end)
    return float(part.duration or 0)


def _part_start(part: VideoPart) -> float:
    if part.transcript:
        return float(part.transcript[0].start)
    return 0.0


def build_question_part_tree(report: VideoReport) -> None:
    original_parts = list(report.parts)
    groups: OrderedDict[str, OrderedDict[str, list[VideoPart]]] = OrderedDict()
    for part in original_parts:
        chapter, qtype, _ = _parse_part_title(part.title)
        groups.setdefault(chapter, OrderedDict()).setdefault(qtype, []).append(part)

    root_blocks: list[KnowledgeBlock] = []
    for chapter_index, (chapter, type_groups) in enumerate(groups.items(), start=1):
        chapter_parts = [part for parts in type_groups.values() for part in parts]
        chapter_block = KnowledgeBlock(
            id=f"q-c{chapter_index}",
            title=chapter,
            start=_part_start(chapter_parts[0]),
            end=_part_end(chapter_parts[-1]),
            summary=f"{chapter}题目合集，共 {len(chapter_parts)} 个分 P。",
            node_type="branch",
            status="expanded",
            expandable=False,
            depth=0,
            children=[],
        )
        for type_index, (qtype, parts) in enumerate(type_groups.items(), start=1):
            type_block = KnowledgeBlock(
                id=f"{chapter_block.id}-t{type_index}",
                title=qtype,
                start=_part_start(parts[0]),
                end=_part_end(parts[-1]),
                summary=f"{chapter}{qtype}，共 {len(parts)} 题。",
                node_type="branch",
                status="expanded",
                expandable=False,
                depth=1,
                children=[],
            )
            for question_index, part in enumerate(parts, start=1):
                _, _, number = _parse_part_title(part.title)
                question_title = part.title
                question_block = KnowledgeBlock(
                    id=f"{type_block.id}-p{part.page}",
                    title=question_title,
                    start=_part_start(part),
                    end=_part_end(part),
                    summary=f"来自 P{part.page}，题号：{number}。点击展开生成这道题的学习笔记。",
                    node_type="leaf",
                    status="skeleton",
                    expandable=True,
                    depth=2,
                    source_part_page=part.page,
                    transcript=part.transcript,
                )
                type_block.children.append(question_block)
            chapter_block.children.append(type_block)
        root_blocks.append(chapter_block)

    tree_part = VideoPart(
        page=0,
        cid=original_parts[0].cid if original_parts else 0,
        title="题目目录",
        duration=sum(int(part.duration or 0) for part in original_parts),
        transcript=[],
        blocks=root_blocks,
    )
    for part in original_parts:
        part.blocks = []
    report.parts = [tree_part, *original_parts]
    report.llm_notes.append(
        f"Generated question tree from {len(original_parts)} parts into {len(root_blocks)} chapter nodes."
    )
