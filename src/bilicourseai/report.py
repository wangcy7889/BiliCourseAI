from __future__ import annotations

import json
import re
from pathlib import Path
from importlib.resources import as_file, files
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from markupsafe import Markup, escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

from bilicourseai.models import FrameArtifact, ReportArtifacts, VideoReport, VisualAnalysis
from bilicourseai.paths import report_dir_for
from bilicourseai.punctuation import readable_transcript_paragraphs


TEMPLATE_RESOURCE = files("bilicourseai") / "templates"


def _file_uri(path: str) -> str:
    if not path:
        return ""
    return Path(path).resolve().as_uri()


def _relative_asset(path: str, report_dir: Path) -> str:
    if not path:
        return ""
    try:
        relative = Path(path).resolve().relative_to(report_dir.resolve())
    except ValueError:
        return Path(path).name
    return relative.as_posix()


def _sentences(lines) -> list[str]:
    return readable_transcript_paragraphs(lines)


def _clip_text(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _visual_note_text(text: str, max_chars: int = 140) -> str:
    text = re.sub(r"(?m)^#{1,6}\s*", "", text or "")
    text = re.sub(r"\s+", " ", text).strip(" ：:;；。")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _time_label(timestamp: float) -> str:
    total_seconds = max(0, int(float(timestamp or 0)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _bilibili_time_url(source_url: str, part_page: int, timestamp: float) -> str:
    if not source_url:
        return ""
    parsed = urlparse(source_url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"p", "t"}
    ]
    if part_page > 1:
        query.append(("p", str(part_page)))
    query.append(("t", str(max(0, int(timestamp)))))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _existing_frames(frames: list[FrameArtifact]) -> list[FrameArtifact]:
    return [frame for frame in frames if frame.error or Path(frame.path).exists()]


def _analyses_for_frames(analyses: list[VisualAnalysis], frames: list[FrameArtifact]) -> list[VisualAnalysis]:
    existing_request_ids = {frame.request_id for frame in frames}
    return [analysis for analysis in analyses if analysis.request_id in existing_request_ids]


def _prune_missing_frame_refs(report: VideoReport) -> None:
    for part in report.parts:
        for root in part.blocks:
            stack = [root]
            while stack:
                block = stack.pop()
                stack.extend(block.children)
                block.frames = _existing_frames(block.frames)
                block.visual_analyses = _analyses_for_frames(block.visual_analyses, block.frames)
                for section in block.sections:
                    section.frames = _existing_frames(section.frames)
                    section.visual_analyses = _analyses_for_frames(section.visual_analyses, section.frames)


def _protect_spans(text: str) -> tuple[str, list[tuple[str, str]]]:
    spans: list[tuple[str, str]] = []

    def store(kind: str, value: str) -> str:
        spans.append((kind, value))
        return f"\u0000SPAN{len(spans) - 1}\u0000"

    math_pattern = re.compile(
        r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|(?<!\$)\$(?!\$)[\s\S]+?(?<!\$)\$(?!\$))"
    )
    text = math_pattern.sub(lambda match: store("math", match.group(0)), text)
    text = re.sub(r"`([^`\n]+)`", lambda match: store("code", match.group(1)), text)
    return text, spans


def _restore_spans(text: str, spans: list[tuple[str, str]]) -> str:
    for index, (kind, value) in enumerate(spans):
        placeholder = f"\u0000SPAN{index}\u0000"
        if kind == "code":
            replacement = f"<code>{escape(value)}</code>"
        else:
            replacement = str(escape(value))
        text = text.replace(placeholder, replacement)
    return text


def _md_inline(text: str) -> Markup:
    protected, spans = _protect_spans(text)
    html = str(escape(protected))
    html = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        html,
    )
    html = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", html)
    html = _restore_spans(html, spans)
    return Markup(html)


def _md_html(text: str) -> Markup:
    text = (text or "").strip()
    if not text:
        return Markup("")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(
        r"(?m)(^|[\s：:；;。！？!?])(\d{1,2}[.．、](?!\d))\s*(?=\S)",
        lambda match: f"{match.group(1).rstrip()}\n{match.group(2)} ",
        normalized,
    )
    lines = normalized.splitlines()
    html: list[str] = []
    list_stack: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def close_lists() -> None:
        while list_stack:
            html.append(f"</{list_stack.pop()}>")

    def flush_paragraph() -> None:
        if not paragraph:
            return
        value = " ".join(item.strip() for item in paragraph if item.strip())
        paragraph.clear()
        html.append(f"<p>{format_inline(value)}</p>")

    def ensure_list(tag: str) -> None:
        if list_stack and list_stack[-1] == tag:
            return
        flush_paragraph()
        close_lists()
        html.append(f"<{tag}>")
        list_stack.append(tag)

    def format_inline(value: str) -> Markup:
        rendered = str(_md_inline(value))
        return Markup(
            re.sub(
                r"^(?!<strong>)([^：:]{1,24}[：:])(.+)$",
                r"<strong>\1</strong>\2",
                rendered,
                count=1,
            )
        )

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```"):
            if in_code:
                code_text = "\n".join(code_lines)
                html.append(f"<pre><code>{escape(code_text)}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                close_lists()
                in_code = True
            continue
        if in_code:
            code_lines.append(raw_line)
            continue

        if not line:
            flush_paragraph()
            close_lists()
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_lists()
            level = min(6, len(heading.group(1)) + 2)
            html.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
            continue

        ordered = re.match(r"^(\d+)[.．、](?!\d)\s*(.+)$", line)
        unordered = re.match(r"^[-*]\s+(.+)$", line)
        if ordered:
            ensure_list("ol")
            html.append(f"<li>{format_inline(ordered.group(2))}</li>")
            continue
        if unordered:
            ensure_list("ul")
            html.append(f"<li>{format_inline(unordered.group(1))}</li>")
            continue

        if list_stack:
            close_lists()
        paragraph.append(line)

    if in_code:
        code_text = "\n".join(code_lines)
        html.append(f"<pre><code>{escape(code_text)}</code></pre>")
    flush_paragraph()
    close_lists()

    return Markup("\n".join(html))


def write_report_to_dir(report: VideoReport, report_dir: Path) -> ReportArtifacts:
    report_dir.mkdir(parents=True, exist_ok=True)
    _prune_missing_frame_refs(report)

    json_path = report_dir / "report.json"
    html_path = report_dir / "report.html"

    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with as_file(TEMPLATE_RESOURCE) as template_dir:
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )
        env.filters["file_uri"] = _file_uri
        env.filters["relative_asset"] = _relative_asset
        env.filters["sentences"] = _sentences
        env.filters["md_html"] = _md_html
        env.filters["md_inline"] = _md_inline
        env.filters["note_html"] = _md_html
        env.filters["clip_text"] = _clip_text
        env.filters["visual_note_text"] = _visual_note_text
        env.filters["bilibili_time_url"] = _bilibili_time_url
        env.filters["time_label"] = _time_label
        template = env.get_template("report.html.j2")
        html_path.write_text(template.render(report=report, report_dir=report_dir), encoding="utf-8")

    return ReportArtifacts(report_dir=report_dir, json_path=json_path, html_path=html_path)


def write_report(report: VideoReport, output_dir: Path) -> ReportArtifacts:
    return write_report_to_dir(report, report_dir_for(report, output_dir))
