from __future__ import annotations

import re

from bilicourseai.models import TranscriptLine


END_PUNCT = "。！？.!?"
MID_PUNCT = "，,；;：:"
ALL_PUNCT = END_PUNCT + MID_PUNCT
SOFT_BREAK_PUNCT = "，,；;：:"
HARD_BREAK_PUNCT = "。！？.!?"

QUESTION_PREFIXES = (
    "什么",
    "为什么",
    "怎么",
    "怎样",
    "如何",
    "是否",
    "能不能",
    "是不是",
    "有没有",
    "哪里",
    "哪种",
    "哪个",
)
QUESTION_SUFFIXES = ("吗", "呢", "么", "嘛", "是什么", "为什么")
CLAUSE_STARTERS = (
    "那么",
    "然后",
    "接下来",
    "所以",
    "因此",
    "但是",
    "然而",
    "不过",
    "另外",
    "同时",
    "比如",
    "例如",
    "事实上",
    "事实证明",
    "换句话说",
    "也就是说",
)
SENTENCE_STARTERS = (
    "我认为",
    "事实证明",
    "AI模型",
    "这意味着",
    "这里",
    "现在",
    "接下来",
    "让我们",
)
PARAGRAPH_STARTERS = (
    "那么",
    "然后",
    "接下来",
    "所以",
    "因此",
    "但是",
    "然而",
    "不过",
    "另外",
    "同时",
    "比如",
    "例如",
    "也就是说",
    "换句话说",
    "这里",
    "注意",
    "我们来看",
    "我们看",
)


def _has_punctuation(text: str) -> bool:
    return bool(text) and text[-1] in ALL_PUNCT


def _looks_question(text: str) -> bool:
    stripped = re.sub(r"\s+", "", text)
    return stripped.startswith(QUESTION_PREFIXES) or stripped.endswith(QUESTION_SUFFIXES)


def _gap_after(lines: list[TranscriptLine], index: int) -> float:
    if index >= len(lines) - 1:
        return 999.0
    return max(0.0, lines[index + 1].start - lines[index].end)


def _next_starts_new_clause(lines: list[TranscriptLine], index: int) -> bool:
    if index >= len(lines) - 1:
        return False
    text = lines[index + 1].text.strip()
    return text.startswith(CLAUSE_STARTERS)


def _next_starts_new_sentence(lines: list[TranscriptLine], index: int) -> bool:
    if index >= len(lines) - 1:
        return False
    text = lines[index + 1].text.strip()
    return text.startswith(SENTENCE_STARTERS)


def _mark_for_line(lines: list[TranscriptLine], index: int, current_sentence_chars: int) -> str:
    line = lines[index]
    text = line.text.strip()
    gap = _gap_after(lines, index)
    is_last = index == len(lines) - 1

    if _looks_question(text):
        return "？"
    if is_last:
        return "。"
    if gap >= 0.85 or current_sentence_chars >= 68 or _next_starts_new_sentence(lines, index):
        return "。"
    if (
        gap >= 0.32
        or len(text) >= 18
        or current_sentence_chars >= 30
        or _next_starts_new_clause(lines, index)
    ):
        return "，"
    return ""


def punctuate_lines(lines: list[TranscriptLine]) -> list[TranscriptLine]:
    result: list[TranscriptLine] = []
    current_sentence_chars = 0

    for index, line in enumerate(lines):
        text = line.text.strip()
        if not text:
            continue

        current_sentence_chars += len(text)
        if _has_punctuation(text):
            new_text = text
            if text[-1] in END_PUNCT:
                current_sentence_chars = 0
        else:
            mark = _mark_for_line(lines, index, current_sentence_chars)
            new_text = text + mark
            if mark in END_PUNCT:
                current_sentence_chars = 0

        result.append(
            TranscriptLine(
                start=line.start,
                end=line.end,
                text=new_text,
            )
        )

    return result


def _line_with_local_mark(lines: list[TranscriptLine], index: int, current_chars: int) -> str:
    text = lines[index].text.strip()
    if not text:
        return ""
    if _has_punctuation(text):
        return text
    mark = _mark_for_line(lines, index, current_chars + len(text))
    if not mark and index < len(lines) - 1:
        next_text = lines[index + 1].text.strip()
        if (
            len(text) >= 7
            or len(next_text) >= 7
            or current_chars + len(text) >= 22
            or next_text.startswith(CLAUSE_STARTERS)
        ):
            mark = "，"
    return text + mark


def _split_at_readable_boundary(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts: list[str] = []
    current = text
    while len(current) > max_chars:
        window = current[: max_chars + 1]
        boundary = -1
        for punct in HARD_BREAK_PUNCT + SOFT_BREAK_PUNCT:
            pos = window.rfind(punct)
            if pos >= max(28, max_chars // 2):
                boundary = max(boundary, pos)
        if boundary <= 0:
            boundary = max_chars
        parts.append(current[: boundary + 1].strip())
        current = current[boundary + 1 :].strip()
    if current:
        parts.append(current)
    return [part for part in parts if part]


def readable_transcript_paragraphs(
    lines: list[TranscriptLine],
    max_chars: int = 110,
    min_break_chars: int = 34,
) -> list[str]:
    paragraphs: list[str] = []
    current = ""
    current_chars = 0

    def flush() -> None:
        nonlocal current
        value = current.strip()
        current = ""
        if not value:
            return
        paragraphs.extend(_split_at_readable_boundary(value, max_chars=max_chars))

    for index, line in enumerate(lines):
        text = _line_with_local_mark(lines, index, current_chars)
        if not text:
            continue

        gap = _gap_after(lines, index)
        starts_new = text.startswith(PARAGRAPH_STARTERS)
        if current and starts_new and len(current) >= min_break_chars:
            flush()

        current += text
        current_chars += len(text)

        if text[-1] in HARD_BREAK_PUNCT:
            flush()
            current_chars = 0
            continue

        next_line = lines[index + 1].text.strip() if index < len(lines) - 1 else ""
        next_starts_new = next_line.startswith(PARAGRAPH_STARTERS)
        should_soft_break = (
            len(current) >= max_chars
            or (gap >= 0.75 and len(current) >= min_break_chars)
            or (next_starts_new and len(current) >= min_break_chars)
        )
        if should_soft_break:
            flush()
            current_chars = 0

    flush()
    return paragraphs
