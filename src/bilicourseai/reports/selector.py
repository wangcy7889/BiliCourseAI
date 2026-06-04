from __future__ import annotations

import os
import sys
from pathlib import Path

import typer


def _read_selection_key() -> str:
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            key = msvcrt.getwch()
            if key == "H":
                return "up"
            if key == "P":
                return "down"
        if key in {"\r", "\n"}:
            return "enter"
        if key in {"\x03", "\x1b", "q", "Q"}:
            return "quit"
        return ""

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "quit"
        if key in {"\r", "\n"}:
            return "enter"
        if key in {"\x03", "q", "Q"}:
            return "quit"
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _choose_report_dir_interactively(matches: list[Path], raw: str) -> Path:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        names = "\n  ".join(candidate.name for candidate in matches[:12])
        suffix = "\n  ..." if len(matches) > 12 else ""
        raise typer.BadParameter(f"匹配到多个报告，请输入更具体的关键词:\n  {names}{suffix}")

    selected = 0
    typer.echo(f"匹配到 {len(matches)} 个报告: {raw}")
    typer.echo("使用 ↑/↓ 选择，Enter 确认，Esc/q 取消。")

    def render() -> None:
        print("\x1b[2K\r", end="")
        for index, candidate in enumerate(matches):
            prefix = ">" if index == selected else " "
            print(f"\x1b[2K\r{prefix} {candidate.name}")
        print(f"\x1b[{len(matches)}A", end="", flush=True)

    render()
    while True:
        key = _read_selection_key()
        if key == "up":
            selected = (selected - 1) % len(matches)
            render()
        elif key == "down":
            selected = (selected + 1) % len(matches)
            render()
        elif key == "enter":
            print(f"\x1b[{len(matches)}B", end="")
            typer.echo(f"Selected: {matches[selected].name}")
            return matches[selected].resolve()
        elif key == "quit":
            print(f"\x1b[{len(matches)}B", end="")
            raise typer.Abort()


def resolve_report_dir(value: Path, data_dir: Path) -> Path:
    if value.exists():
        report_dir = value.parent if value.is_file() and value.name == "report.json" else value
        return report_dir.resolve()

    raw = str(value).strip()
    reports_dir = data_dir / "reports"
    if not reports_dir.exists():
        raise typer.BadParameter(f"报告目录不存在: {value}；也没有找到报告根目录: {reports_dir}")

    needle = raw.casefold()
    matches = []
    for candidate in reports_dir.iterdir():
        if not candidate.is_dir():
            continue
        if not (candidate / "report.json").exists():
            continue
        name = candidate.name.casefold()
        if needle == name or needle in name:
            matches.append(candidate)

    if not matches:
        raise typer.BadParameter(
            f"没有找到匹配的报告: {raw}。可以传完整目录、report.json、BVID，或 data/reports 下目录名的一段唯一关键词。"
        )
    if len(matches) > 1:
        return _choose_report_dir_interactively(matches, raw)
    return matches[0].resolve()
