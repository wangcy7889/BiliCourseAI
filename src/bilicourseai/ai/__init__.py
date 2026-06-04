from __future__ import annotations

from bilicourseai.ai.enrich import enrich_report_text
from bilicourseai.ai.expand import expand_block
from bilicourseai.ai.outline import outline_report
from bilicourseai.ai.segment import ai_segment_report

__all__ = [
    "ai_segment_report",
    "enrich_report_text",
    "expand_block",
    "outline_report",
]
