from __future__ import annotations

import re
from pathlib import Path

from bilicourseai.models import VideoReport


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_dir_name(value: str, max_chars: int = 72) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        return "untitled"
    if value.upper() in WINDOWS_RESERVED_NAMES:
        value = f"{value}_video"
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip(" .")


def report_folder_name(report: VideoReport) -> str:
    title = _safe_dir_name(report.title)
    return f"{title}__{report.bvid}"


def report_dir_for(report: VideoReport, output_dir: Path) -> Path:
    return output_dir / "reports" / report_folder_name(report)


def report_dir_from_json(report_json: Path) -> Path:
    return report_json.resolve().parent


def output_dir_for_report_dir(report_dir: Path, fallback_output_dir: Path) -> Path:
    report_dir = report_dir.resolve()
    if report_dir.parent.name == "reports":
        return report_dir.parent.parent
    return fallback_output_dir
