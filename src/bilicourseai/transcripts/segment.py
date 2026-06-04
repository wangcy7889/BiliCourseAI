from __future__ import annotations

from bilicourseai.models import KnowledgeBlock, TranscriptLine, VideoPart
from bilicourseai.transcripts.punctuation import punctuate_lines


def _compact_summary(lines: list[TranscriptLine], max_chars: int = 160) -> str:
    text = "".join(line.text for line in lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _make_title(lines: list[TranscriptLine], fallback: str) -> str:
    for line in lines:
        text = line.text.strip()
        if text:
            return text[:32]
    return fallback


def segment_part(part: VideoPart, window_seconds: int = 180) -> VideoPart:
    if not part.transcript:
        part.blocks = []
        return part

    part.transcript = punctuate_lines(part.transcript)

    buckets: list[list[TranscriptLine]] = []
    current: list[TranscriptLine] = []
    window_start = part.transcript[0].start

    for line in part.transcript:
        if current and line.start - window_start >= window_seconds:
            buckets.append(current)
            current = []
            window_start = line.start
        current.append(line)

    if current:
        buckets.append(current)

    blocks: list[KnowledgeBlock] = []
    for index, lines in enumerate(buckets, start=1):
        block = KnowledgeBlock(
            id=f"p{part.page}-b{index}",
            title=_make_title(lines, f"Block {index}"),
            start=lines[0].start,
            end=lines[-1].end,
            summary=_compact_summary(lines),
            transcript=lines,
            children=[
                KnowledgeBlock(
                    id=f"p{part.page}-b{index}-raw",
                    title="字幕证据",
                    start=lines[0].start,
                    end=lines[-1].end,
                    summary="",
                    transcript=lines,
                )
            ],
        )
        blocks.append(block)

    part.blocks = blocks
    return part


def segment_report_parts(parts: list[VideoPart], window_seconds: int = 180) -> list[VideoPart]:
    return [segment_part(part, window_seconds=window_seconds) for part in parts]
