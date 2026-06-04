from __future__ import annotations

from typing import Any

from bilicourseai.models import TranscriptLine, VideoReport
from bilicourseai.tree import find_block


def all_block_payload(report: VideoReport) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in report.parts:
        for block in part.blocks:
            transcript_excerpt = "".join(line.text for line in block.transcript[:36])
            blocks.append(
                {
                    "part_page": part.page,
                    "part_title": part.title,
                    "block_id": block.id,
                    "start": block.start,
                    "end": block.end,
                    "title": block.title,
                    "summary": block.summary,
                    "transcript_excerpt": transcript_excerpt[:1800],
                }
            )
    return blocks


def find_report_block(report: VideoReport, block_id: str):
    return find_block(report, block_id)


def transcript_windows(report: VideoReport, window_seconds: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for part in report.parts:
        if not part.transcript:
            continue
        current: list[TranscriptLine] = []
        window_start = part.transcript[0].start
        for line in part.transcript:
            if current and line.start - window_start >= window_seconds:
                windows.append(_window_payload(part.page, part.title, current))
                current = []
                window_start = line.start
            current.append(line)
        if current:
            windows.append(_window_payload(part.page, part.title, current))
    return windows


def _window_payload(part_page: int, part_title: str, lines: list[TranscriptLine]) -> dict[str, Any]:
    return {
        "part_page": part_page,
        "part_title": part_title,
        "start": lines[0].start,
        "end": lines[-1].end,
        "transcript": [
            {"start": line.start, "end": line.end, "text": line.text}
            for line in lines
        ],
    }


def find_report_part(report: VideoReport, page: int):
    for part in report.parts:
        if part.page == page:
            return part
    return None
