from __future__ import annotations

import json
import re
from pathlib import Path
from importlib.resources import as_file, files
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from markupsafe import Markup, escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

from bilicourseai.models import FrameArtifact, KnowledgeBlock, ReportArtifacts, VideoPart, VideoReport, VisualAnalysis
from bilicourseai.paths import report_dir_for
from bilicourseai.transcripts.punctuation import readable_transcript_paragraphs


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


def _block_has_transcript(block: KnowledgeBlock) -> bool:
    if block.transcript:
        return True
    return any(_block_has_transcript(child) for child in block.children)


def _part_has_transcript(part: VideoPart) -> bool:
    if part.transcript:
        return True
    return any(_block_has_transcript(block) for block in part.blocks)


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


def _is_part_outline_block(block: KnowledgeBlock) -> bool:
    return (
        block.source_part_page is not None
        and block.id == f"part-p{block.source_part_page}"
    )


def _toc_action_label(block: KnowledgeBlock) -> str:
    is_structure_node = block.node_type == "branch" or _is_part_outline_block(block)
    if not is_structure_node:
        return ""
    if block.status == "skeleton":
        return "展开划分"
    if block.status == "expanded":
        return "重做划分"
    return ""


def _body_action_label(block: KnowledgeBlock) -> str:
    if _toc_action_label(block):
        return ""
    if block.node_type != "leaf":
        return ""
    if block.status == "skeleton":
        return "展开为笔记"
    if block.status == "expanded":
        return "重做笔记"
    return ""


def _block_has_body_content(block: KnowledgeBlock) -> bool:
    if block.node_type == "raw":
        return False
    has_note_content = bool(
        block.summary
        or block.key_points
        or block.sections
        or block.frames
        or block.visual_requests
        or block.visual_analyses
        or _body_action_label(block)
    )
    if has_note_content and not _toc_action_label(block):
        return True
    return any(_block_has_body_content(child) for child in block.children)


def _toc_has_link_target(block: KnowledgeBlock) -> bool:
    return _block_has_body_content(block)


def _part_has_body_content(part: VideoPart) -> bool:
    return any(_block_has_body_content(block) for block in part.blocks)


def _report_has_body_content(report: VideoReport) -> bool:
    return any(_part_has_body_content(part) for part in report.parts)


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
        r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|(?<![\w$])\$(?![\s$])[^$\n]+?(?<![\s$])\$(?![\w$]))"
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
    html = re.sub(r"&lt;br\s*/?&gt;", "<br>", html, flags=re.IGNORECASE)
    html = _restore_spans(html, spans)
    return Markup(html)


def _repair_latex_control_escapes(text: str) -> str:
    # Some LLM JSON payloads forget to double-escape LaTeX backslashes.
    # JSON then turns commands such as \rightarrow into control chars.
    replacements = {
        "\r" + "ightarrow": r"\rightarrow",
        "\r" + "angle": r"\rangle",
        "\r" + "ho": r"\rho",
        "\t" + "heta": r"\theta",
        "\t" + "imes": r"\times",
        "\t" + "o": r"\to",
        "\b" + "ar": r"\bar",
        "\f" + "rac": r"\frac",
        "\n" + "abla": r"\nabla",
    }
    for broken, fixed in replacements.items():
        text = text.replace(broken, fixed)
    return text


def _normalize_overescaped_latex(text: str) -> str:
    # Some payloads over-escape LaTeX before JSON encoding, so after JSON parsing
    # math still contains `\\sin` instead of `\sin`. MathJax treats that as a
    # line-break command plus text and may render the formula as empty/error.
    math_pattern = re.compile(
        r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|(?<![\w$])\$(?![\s$])[^$\n]+?(?<![\s$])\$(?![\w$]))"
    )

    def normalize_span(match: re.Match[str]) -> str:
        value = match.group(0)
        return re.sub(r"\\\\(?=[A-Za-z,;!])", r"\\", value)

    return math_pattern.sub(normalize_span, text)


def _escape_unmatched_math_dollars(text: str) -> str:
    result: list[str] = []
    index = 0
    in_inline_math = False
    while index < len(text):
        char = text[index]
        if char != "$":
            result.append(char)
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            result.append("$$")
            index += 2
            continue
        previous_char = text[index - 1] if index else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        can_open = not in_inline_math and next_char and not next_char.isspace() and previous_char not in "$\\"
        can_close = in_inline_math and previous_char and not previous_char.isspace() and next_char not in "$"
        if can_open or can_close:
            in_inline_math = not in_inline_math
            result.append("$")
        else:
            result.append("")
        index += 1
    if in_inline_math:
        for reverse_index in range(len(result) - 1, -1, -1):
            if result[reverse_index] == "$":
                result[reverse_index] = ""
                break
    return "".join(result)


def _simplify_standalone_latex_symbols(text: str) -> str:
    text = re.sub(r"\$\\(?:right)?arrow\$", "→", text)
    text = re.sub(r"\$\\to\$", "→", text)
    return text


def _md_html(text: str) -> Markup:
    text = _escape_unmatched_math_dollars(
        _normalize_overescaped_latex(
            _simplify_standalone_latex_symbols(_repair_latex_control_escapes(text or ""))
        )
    ).strip()
    if not text:
        return Markup("")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    def split_inline_ordered_lists(value: str) -> str:
        fixed_lines: list[str] = []
        for source_line in value.splitlines():
            if re.match(r"^\s*#{1,6}\s+\d{1,2}[.．、](?!\d)", source_line):
                fixed_lines.append(source_line)
                continue
            if re.match(r"^\s*\d{1,2}[.．、](?!\d)\s+", source_line):
                fixed_lines.append(source_line)
                continue
            fixed_lines.append(
                re.sub(
                    r"(^|[\s：:；;。！？!?])(\d{1,2}[.．、](?!\d))\s*(?=\S)",
                    lambda match: (
                        f"{match.group(2)} "
                        if match.group(1) == ""
                        else f"{match.group(1).rstrip()}\n{match.group(2)} "
                    ),
                    source_line,
                )
            )
        return "\n".join(fixed_lines)

    normalized = split_inline_ordered_lists(normalized)
    def table_cells(value: str) -> list[str]:
        stripped = value.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]

    def tsv_cells(value: str) -> list[str]:
        return [cell.strip() for cell in value.strip().split("\t")]

    def is_table_separator(value: str) -> bool:
        cells = table_cells(value)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

    def expand_compact_table_line(value: str) -> list[str]:
        line = value.strip()
        if not (
            line.startswith("|")
            and line.endswith("|")
            and re.search(r"\|\s*:?-{3,}:?\s*\|", line)
            and re.search(r"\|\s+\|", line)
        ):
            return [value]

        rows: list[str] = []
        for piece in re.split(r"\|\s+\|", line):
            piece = piece.strip()
            if not piece:
                continue
            if not piece.startswith("|"):
                piece = "| " + piece
            if not piece.endswith("|"):
                piece = piece + " |"
            rows.append(piece)

        if len(rows) >= 2 and is_table_separator(rows[1]):
            return rows
        return [value]

    expanded_lines: list[str] = []
    for raw_line in normalized.splitlines():
        expanded_lines.extend(expand_compact_table_line(raw_line))
    normalized = "\n".join(expanded_lines)
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
        if "$" in value or r"\(" in value or r"\[" in value:
            return Markup(rendered)
        return Markup(
            re.sub(
                r"^(?!<strong>)([^：:]{1,24}[：:])(.+)$",
                r"<strong>\1</strong>\2",
                rendered,
                count=1,
            )
        )

    def is_table_row(value: str) -> bool:
        return value.startswith("|") and value.endswith("|") and len(table_cells(value)) >= 2

    def is_tsv_row(value: str) -> bool:
        cells = tsv_cells(value)
        return "\t" in value and len(cells) >= 2 and sum(bool(cell) for cell in cells) >= 2

    def render_table(start_index: int) -> int:
        header = table_cells(lines[start_index].strip())
        separator = table_cells(lines[start_index + 1].strip())
        alignments = []
        for cell in separator:
            align = ""
            if cell.startswith(":") and cell.endswith(":"):
                align = ' style="text-align:center"'
            elif cell.endswith(":"):
                align = ' style="text-align:right"'
            alignments.append(align)

        html.append("<table>")
        html.append("<thead><tr>")
        for index, cell in enumerate(header):
            align = alignments[index] if index < len(alignments) else ""
            html.append(f"<th{align}>{format_inline(cell)}</th>")
        html.append("</tr></thead>")
        html.append("<tbody>")

        index = start_index + 2
        while index < len(lines):
            row_line = lines[index].strip()
            if not is_table_row(row_line):
                break
            html.append("<tr>")
            for cell_index, cell in enumerate(table_cells(row_line)):
                align = alignments[cell_index] if cell_index < len(alignments) else ""
                html.append(f"<td{align}>{format_inline(cell)}</td>")
            html.append("</tr>")
            index += 1

        html.append("</tbody>")
        html.append("</table>")
        return index

    def render_tsv_table(start_index: int) -> int:
        header = tsv_cells(lines[start_index].strip())
        column_count = len(header)

        html.append("<table>")
        html.append("<thead><tr>")
        for cell in header:
            html.append(f"<th>{format_inline(cell)}</th>")
        html.append("</tr></thead>")
        html.append("<tbody>")

        index = start_index + 1
        while index < len(lines):
            row_line = lines[index].strip()
            if not is_tsv_row(row_line):
                break
            cells = tsv_cells(row_line)
            if len(cells) != column_count:
                break
            html.append("<tr>")
            for cell in cells:
                html.append(f"<td>{format_inline(cell)}</td>")
            html.append("</tr>")
            index += 1

        html.append("</tbody>")
        html.append("</table>")
        return index

    line_index = 0
    while line_index < len(lines):
        raw_line = lines[line_index]
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
            line_index += 1
            continue
        if in_code:
            code_lines.append(raw_line)
            line_index += 1
            continue

        if not line:
            flush_paragraph()
            close_lists()
            line_index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_lists()
            level = min(6, len(heading.group(1)) + 2)
            html.append(f"<h{level}>{_md_inline(heading.group(2))}</h{level}>")
            line_index += 1
            continue

        quote = re.match(r"^>\s*(.+)$", line)
        if quote:
            flush_paragraph()
            close_lists()
            html.append(f"<blockquote><p>{format_inline(quote.group(1))}</p></blockquote>")
            line_index += 1
            continue

        if (
            line_index + 1 < len(lines)
            and is_table_row(line)
            and is_table_separator(lines[line_index + 1].strip())
        ):
            flush_paragraph()
            close_lists()
            line_index = render_table(line_index)
            continue

        if line_index + 1 < len(lines) and is_tsv_row(line) and is_tsv_row(lines[line_index + 1].strip()):
            flush_paragraph()
            close_lists()
            line_index = render_tsv_table(line_index)
            continue

        ordered = re.match(r"^(\d+)[.．、](?!\d)\s*(.+)$", line)
        unordered = re.match(r"^[-*]\s+(.+)$", line)
        if ordered:
            ensure_list("ol")
            html.append(f"<li>{format_inline(ordered.group(2))}</li>")
            line_index += 1
            continue
        if unordered:
            ensure_list("ul")
            html.append(f"<li>{format_inline(unordered.group(1))}</li>")
            line_index += 1
            continue

        if list_stack:
            close_lists()
        paragraph.append(line)
        line_index += 1

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
        env.filters["toc_action_label"] = _toc_action_label
        env.filters["body_action_label"] = _body_action_label
        env.filters["report_has_body_content"] = _report_has_body_content
        env.filters["toc_has_link_target"] = _toc_has_link_target
        env.tests["has_transcript"] = _part_has_transcript
        env.tests["has_body_content"] = _part_has_body_content
        template = env.get_template("report.html.j2")
        html_path.write_text(template.render(report=report, report_dir=report_dir), encoding="utf-8")

    return ReportArtifacts(report_dir=report_dir, json_path=json_path, html_path=html_path)


def write_report(report: VideoReport, output_dir: Path) -> ReportArtifacts:
    return write_report_to_dir(report, report_dir_for(report, output_dir))
