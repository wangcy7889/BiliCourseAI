from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer

from bilicourseai.auth import check_bilibili_credential, poll_qr_login, start_qr_login
from bilicourseai.bilibili import fetch_video_report
from bilicourseai.frames import capture_requested_frames
from bilicourseai.llm import (
    ai_segment_report,
    analyze_frames,
    enrich_report_text,
    expand_block,
    outline_report,
    rewrite_visual_notes,
)
from bilicourseai.models import VideoReport
from bilicourseai.punctuation import punctuate_lines
from bilicourseai.paths import report_dir_for, report_dir_from_json
from bilicourseai.question_tree import build_question_part_tree
from bilicourseai.report import write_report, write_report_to_dir
from bilicourseai.segment import segment_report_parts
from bilicourseai.server import run_report_server
from bilicourseai.settings import (
    BILIBILI_CREDENTIAL_FILE,
    BilibiliCredentialSettings,
    DEFAULT_DATA_DIR,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    LLM_SETTINGS_FILE,
    LLMSettings,
    clear_bilibili_credential_settings,
    clear_llm_settings,
    load_bilibili_credential_settings,
    load_llm_settings,
    save_bilibili_credential_settings,
    save_llm_settings,
)
from bilicourseai.visual_pipeline import run_visual_pipeline


app = typer.Typer(no_args_is_help=True)
auth_app = typer.Typer(no_args_is_help=True, help="Manage Bilibili login credentials.")
config_app = typer.Typer(no_args_is_help=True, help="Manage saved BiliCourseAI settings.")


@app.callback()
def main() -> None:
    """BiliCourseAI command line tools."""


def _mask(value: str | None, keep: int = 4) -> str:
    if not value:
        return "未设置"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def _progress(message: str) -> None:
    print(message, flush=True)


@auth_app.command("set")
def auth_set(
    sessdata: str | None = typer.Option(None, "--sessdata", help="Bilibili cookie: SESSDATA"),
    bili_jct: str | None = typer.Option(None, "--bili-jct", help="Bilibili cookie: bili_jct"),
    dedeuserid: str | None = typer.Option(None, "--dedeuserid", help="Bilibili cookie: DedeUserID"),
    buvid3: str | None = typer.Option(None, "--buvid3", help="Bilibili cookie: buvid3，可选"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="保存后验证登录凭据"),
) -> None:
    """Save Bilibili cookies for subtitle access."""

    if sessdata is None:
        sessdata = typer.prompt("SESSDATA", hide_input=True)
    if bili_jct is None:
        bili_jct = typer.prompt("bili_jct", hide_input=True)
    if dedeuserid is None:
        dedeuserid = typer.prompt("DedeUserID", hide_input=True)
    if buvid3 is None:
        buvid3 = typer.prompt("buvid3 (optional)", default="", show_default=False)

    settings = BilibiliCredentialSettings(
        sessdata=sessdata,
        bili_jct=bili_jct,
        dedeuserid=dedeuserid,
        buvid3=buvid3 or None,
    )
    path = save_bilibili_credential_settings(settings)
    typer.echo(f"Saved: {path}")

    if validate:
        ok = asyncio.run(check_bilibili_credential(settings))
        if ok:
            typer.echo("Bilibili credential: valid")
        else:
            raise typer.ClickException("Bilibili credential validation failed. 请检查 cookie 是否完整或已过期。")


@auth_app.command("qr")
def auth_qr(
    poll_interval: float = typer.Option(3.0, "--poll-interval", help="扫码状态轮询间隔，单位秒"),
    timeout_seconds: int = typer.Option(300, "--timeout", help="二维码登录超时时间，单位秒"),
    show_terminal_qr: bool = typer.Option(True, "--terminal-qr/--no-terminal-qr", help="在终端打印字符二维码"),
) -> None:
    """Login with Bilibili App QR code and save cookies."""

    async def run() -> None:
        session = await start_qr_login()
        typer.echo(f"QR image: {session.qrcode_path}")
        if show_terminal_qr:
            typer.echo(session.terminal_qrcode)
        typer.echo("请使用 B 站手机 App 扫码并确认登录。")

        deadline = session.created_at + timeout_seconds
        last_status = ""
        while True:
            status, settings = await poll_qr_login(session)
            if status != last_status:
                messages = {
                    "waiting": "等待扫码...",
                    "scan": "已扫码，等待手机确认...",
                    "confirm": "请在手机上确认登录...",
                    "expired": "二维码已过期，请重新运行 auth qr。",
                    "success": "登录成功，正在保存凭据...",
                }
                typer.echo(messages.get(status, f"状态: {status}"))
                last_status = status

            if status == "success" and settings is not None:
                path = save_bilibili_credential_settings(settings)
                typer.echo(f"Saved: {path}")
                ok = await check_bilibili_credential(settings)
                typer.echo(f"Bilibili credential: {'valid' if ok else 'saved but validation failed'}")
                return
            if status == "expired":
                raise typer.Exit(code=1)
            if time.time() >= deadline:
                raise typer.ClickException("二维码登录超时，请重新运行 auth qr。")
            await asyncio.sleep(poll_interval)

    import time

    asyncio.run(run())


@auth_app.command("status")
def auth_status(validate: bool = typer.Option(False, "--validate", help="向 Bilibili 验证当前凭据")) -> None:
    """Show whether Bilibili credentials are configured."""

    settings = load_bilibili_credential_settings()
    typer.echo(f"Credential file: {BILIBILI_CREDENTIAL_FILE}")
    typer.echo(f"SESSDATA: {_mask(settings.sessdata)}")
    typer.echo(f"bili_jct: {_mask(settings.bili_jct)}")
    typer.echo(f"DedeUserID: {_mask(settings.dedeuserid)}")
    typer.echo(f"buvid3: {_mask(settings.buvid3)}")

    if validate:
        ok = asyncio.run(check_bilibili_credential(settings))
        typer.echo(f"Bilibili credential: {'valid' if ok else 'invalid'}")


@auth_app.command("clear")
def auth_clear() -> None:
    """Delete saved Bilibili credentials."""

    deleted = clear_bilibili_credential_settings()
    typer.echo("Deleted saved credentials." if deleted else "No saved credentials found.")


app.add_typer(auth_app, name="auth")


@config_app.command("llm")
def config_llm(
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    text_model: str = typer.Option(DEFAULT_TEXT_MODEL, "--text-model", help="默认文本模型名"),
    vision_model: str = typer.Option(DEFAULT_VISION_MODEL, "--vision-model", help="默认视觉模型名"),
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking/--disable-thinking",
        help="保存默认 thinking 设置",
    ),
) -> None:
    """Save default LLM settings to the BiliCourseAI config directory."""

    if base_url is None:
        base_url = typer.prompt("base_url")
    if api_key is None:
        api_key = typer.prompt("api_key", hide_input=True)

    path = save_llm_settings(
        LLMSettings(
            base_url=base_url,
            api_key=api_key,
            text_model=text_model,
            vision_model=vision_model,
            enable_thinking=enable_thinking,
        )
    )
    typer.echo(f"Saved: {path}")
    typer.echo(f"Text model: {text_model}")
    typer.echo(f"Vision model: {vision_model}")


@config_app.command("status")
def config_status() -> None:
    """Show saved/default LLM settings."""

    settings = load_llm_settings()
    typer.echo(f"LLM config file: {LLM_SETTINGS_FILE}")
    typer.echo(f"base_url: {settings.base_url or '未设置'}")
    typer.echo(f"api_key: {_mask(settings.api_key)}")
    typer.echo(f"text_model: {settings.text_model or '未设置'}")
    typer.echo(f"vision_model: {settings.vision_model or '未设置'}")
    typer.echo(f"enable_thinking: {settings.enable_thinking}")


@config_app.command("clear")
def config_clear() -> None:
    """Delete saved LLM settings."""

    deleted = clear_llm_settings()
    typer.echo("Deleted saved LLM settings." if deleted else "No saved LLM settings found.")


app.add_typer(config_app, name="config")


def _apply_llm_overrides(
    base_url: str | None,
    api_key: str | None,
    text_model: str | None,
    vision_model: str | None = None,
    enable_thinking: bool = False,
) -> None:
    if base_url:
        os.environ["BILICOURSE_BASE_URL"] = base_url
    if api_key:
        os.environ["BILICOURSE_API_KEY"] = api_key
    if text_model:
        os.environ["BILICOURSE_TEXT_MODEL"] = text_model
    if vision_model:
        os.environ["BILICOURSE_VISION_MODEL"] = vision_model
    if enable_thinking:
        os.environ["BILICOURSE_ENABLE_THINKING"] = "true"
    else:
        os.environ.pop("BILICOURSE_ENABLE_THINKING", None)


def _require_llm_settings(need_vision: bool = False):
    settings = load_llm_settings()
    if not settings.base_url or not settings.api_key or not settings.text_model:
        raise typer.BadParameter(
            "需要先运行 `bilicourse config llm`，或传入 --base-url/--api-key/--text-model"
        )
    if need_vision and not settings.vision_model:
        raise typer.BadParameter("需要配置 vision_model")
    return settings


def _merge_fetched_report(existing: VideoReport, fetched: VideoReport, part_page: int | None) -> VideoReport:
    existing.aid = fetched.aid
    existing.title = fetched.title
    existing.owner_name = fetched.owner_name
    existing.source_url = fetched.source_url

    existing_by_page = {part.page: part for part in existing.parts}
    merged_parts = []
    for fetched_part in fetched.parts:
        old_part = existing_by_page.get(fetched_part.page)
        if old_part is not None and (part_page is None or fetched_part.page != part_page):
            merged_parts.append(old_part)
            continue
        if old_part is not None and part_page == fetched_part.page:
            fetched_part.blocks = old_part.blocks
        merged_parts.append(fetched_part)
    existing.parts = merged_parts
    return existing


@app.command()
def outline(
    source: str = typer.Argument(..., help="Bilibili 视频 URL 或 BVID"),
    output_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--output-dir", "-o", help="数据输出目录"),
    prefer_ai_subtitle: bool = typer.Option(
        True,
        "--prefer-ai-subtitle/--prefer-human-subtitle",
        help="字幕轨道优先级：AI 字幕优先或人工字幕优先",
    ),
    outline_window_seconds: int = typer.Option(
        720,
        "--outline-window-seconds",
        help="P 内软语义窗口目标秒数",
    ),
    outline_overlap_seconds: int = typer.Option(
        75,
        "--outline-overlap-seconds",
        help="传给 LLM 的窗口前后重叠上下文秒数",
    ),
    part_page: int | None = typer.Option(None, "--part-page", help="只为指定分 P 生成骨架"),
    part_tree_mode: str | None = typer.Option(
        None,
        "--part-tree-mode",
        help="按分 P 标题生成树；可选 question，适合逐题讲解合集",
    ),
    max_outline_windows: int = typer.Option(0, "--max-outline-windows", help="最多处理多少个骨架窗口，0 表示不限"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    text_model: str | None = typer.Option(None, "--text-model", help="文本模型名"),
    llm_request_delay: float = typer.Option(2.2, "--llm-request-delay", help="模型请求间隔秒数"),
    enable_thinking: bool = typer.Option(False, "--enable-thinking/--disable-thinking", help="是否开启 thinking"),
) -> None:
    """Generate only a coarse expandable outline, without screenshots or vision analysis."""

    async def run() -> None:
        _apply_llm_overrides(base_url, api_key, text_model, None, enable_thinking)
        settings = _require_llm_settings()
        typer.echo("Mode: outline skeleton")
        fetched_report = await fetch_video_report(
            source,
            prefer_ai_subtitle=prefer_ai_subtitle,
            progress=_progress,
        )
        existing_json = report_dir_for(fetched_report, output_dir) / "report.json"
        if existing_json.exists():
            report = VideoReport.model_validate(json.loads(existing_json.read_text(encoding="utf-8")))
            report = _merge_fetched_report(report, fetched_report, part_page=part_page)
            typer.echo(f"Loaded existing report: {existing_json}")
        else:
            report = fetched_report
        for part in report.parts:
            part.transcript = punctuate_lines(part.transcript)
        if part_tree_mode:
            if part_page is not None:
                raise typer.BadParameter("--part-tree-mode 不能和 --part-page 同时使用")
            if part_tree_mode.lower() not in {"question", "questions", "title-groups"}:
                raise typer.BadParameter("--part-tree-mode 目前支持 question")
            build_question_part_tree(report)
            artifacts = write_report(report, output_dir)
            typer.echo(f"Question tree: {sum(len(part.blocks) for part in report.parts)} root nodes")
            typer.echo(f"JSON: {artifacts.json_path}")
            typer.echo(f"HTML: {artifacts.html_path}")
            return
        await outline_report(
            report,
            settings,
            outline_window_seconds=outline_window_seconds,
            outline_overlap_seconds=outline_overlap_seconds,
            part_page=part_page,
            max_windows=max_outline_windows,
            request_delay=llm_request_delay,
            progress=_progress,
        )
        artifacts = write_report(report, output_dir)
        typer.echo(f"JSON: {artifacts.json_path}")
        typer.echo(f"HTML: {artifacts.html_path}")

    asyncio.run(run())


@app.command()
def probe(
    source: str = typer.Argument(..., help="Bilibili 视频 URL 或 BVID"),
    output_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--output-dir", "-o", help="数据输出目录"),
    prefer_ai_subtitle: bool = typer.Option(
        True,
        "--prefer-ai-subtitle/--prefer-human-subtitle",
        help="字幕轨道优先级：AI 字幕优先或人工字幕优先",
    ),
) -> None:
    """Fetch metadata and subtitles only, with progress output."""

    async def run() -> None:
        typer.echo("Mode: probe (metadata + subtitles only)")
        report = await fetch_video_report(
            source,
            prefer_ai_subtitle=prefer_ai_subtitle,
            progress=_progress,
        )
        for part in report.parts:
            part.transcript = punctuate_lines(part.transcript)
        artifacts = write_report(report, output_dir)
        typer.echo(f"Parts: {len(report.parts)}")
        for part in report.parts:
            typer.echo(
                f"P{part.page}: duration={part.duration}s transcript={len(part.transcript)} "
                f"tracks={len(part.subtitle_tracks)}"
            )
            for error in part.subtitle_errors:
                typer.echo(f"  subtitle error: {error}")
        typer.echo(f"JSON: {artifacts.json_path}")
        typer.echo(f"HTML: {artifacts.html_path}")

    asyncio.run(run())


@app.command()
def expand(
    report_json: Path = typer.Argument(..., help="已有 report.json 路径"),
    block_id: str = typer.Option(..., "--block-id", help="要展开的节点 ID"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="数据输出目录；默认写回 report.json 所在目录"),
    max_visual_requests: int = typer.Option(3, "--max-visual-requests", help="展开成叶子时最多请求的图片数"),
    prefer_stream_frames: bool = typer.Option(
        True,
        "--prefer-stream-frames/--prefer-storyboard-frames",
        help="优先从真实视频流截高清帧",
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    text_model: str | None = typer.Option(None, "--text-model", help="文本模型名"),
    vision_model: str | None = typer.Option(None, "--vision-model", help="视觉模型名"),
    llm_request_delay: float = typer.Option(2.2, "--llm-request-delay", help="模型请求间隔秒数"),
    enable_thinking: bool = typer.Option(False, "--enable-thinking/--disable-thinking", help="是否开启 thinking"),
) -> None:
    """Expand one selected outline node into child nodes or a leaf note."""

    async def run() -> None:
        _apply_llm_overrides(base_url, api_key, text_model, vision_model, enable_thinking)
        settings = _require_llm_settings()
        if not report_json.exists():
            raise typer.BadParameter(f"report.json 不存在: {report_json}")
        report_dir = report_dir_from_json(report_json)
        data_output_dir = output_dir or DEFAULT_DATA_DIR

        report = VideoReport.model_validate(json.loads(report_json.read_text(encoding="utf-8")))
        typer.echo(f"Expanding: {block_id}")
        visual_requests = await expand_block(
            report,
            block_id,
            settings,
            max_visual_requests=max_visual_requests,
            request_delay=llm_request_delay,
        )

        await run_visual_pipeline(
            report,
            visual_requests,
            settings,
            data_output_dir,
            prefer_stream_frames=prefer_stream_frames,
            request_delay=llm_request_delay,
            report_dir=report_dir if output_dir is None else None,
            progress=typer.echo,
        )

        artifacts = write_report(report, output_dir) if output_dir is not None else write_report_to_dir(report, report_dir)
        typer.echo(f"JSON: {artifacts.json_path}")
        typer.echo(f"HTML: {artifacts.html_path}")

    asyncio.run(run())


@app.command()
def analyze(
    source: str = typer.Argument(..., help="Bilibili 视频 URL 或 BVID"),
    output_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--output-dir", "-o", help="数据输出目录"),
    window_seconds: int = typer.Option(180, "--window-seconds", help="粗分块窗口，单位秒"),
    prefer_ai_subtitle: bool = typer.Option(
        True,
        "--prefer-ai-subtitle/--prefer-human-subtitle",
        help="字幕轨道优先级：AI 字幕优先或人工字幕优先",
    ),
    use_llm: bool = typer.Option(False, "--use-llm", help="启用 LLM 文本规划、按需截图和视觉分析。"),
    ai_segment: bool = typer.Option(False, "--ai-segment", help="让 AI 根据字幕重新分段并过滤无用口播信息"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    text_model: str | None = typer.Option(None, "--text-model", help="文本模型名"),
    vision_model: str | None = typer.Option(None, "--vision-model", help="视觉模型名"),
    max_llm_blocks: int = typer.Option(0, "--max-llm-blocks", help="本次交给文本模型处理的 block 上限，0 表示全部"),
    llm_batch_size: int = typer.Option(12, "--llm-batch-size", help="文本模型每批处理的 block 数"),
    ai_segment_window: int = typer.Option(480, "--ai-segment-window", help="AI 分段时每批字幕窗口秒数"),
    max_visual_requests: int = typer.Option(0, "--max-visual-requests", help="模型可请求的截图上限，0 表示不设总上限"),
    prefer_stream_frames: bool = typer.Option(
        True,
        "--prefer-stream-frames/--prefer-storyboard-frames",
        help="优先从真实视频流截高清帧；失败时回退到 B 站 storyboard 小图",
    ),
    llm_request_delay: float = typer.Option(
        2.2,
        "--llm-request-delay",
        help="每次模型请求后的等待秒数；2.2 秒约等于低于 30 RPM",
    ),
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking/--disable-thinking",
        help="是否为支持该参数的模型开启 thinking；默认关闭以降低延迟和 token 消耗",
    ),
) -> None:
    """Generate a local study report from Bilibili metadata, subtitles, and optional LLM vision."""

    async def run() -> None:
        if base_url:
            os.environ["BILICOURSE_BASE_URL"] = base_url
        if api_key:
            os.environ["BILICOURSE_API_KEY"] = api_key
        if text_model:
            os.environ["BILICOURSE_TEXT_MODEL"] = text_model
        if vision_model:
            os.environ["BILICOURSE_VISION_MODEL"] = vision_model
        if enable_thinking:
            os.environ["BILICOURSE_ENABLE_THINKING"] = "true"
        else:
            os.environ.pop("BILICOURSE_ENABLE_THINKING", None)

        llm_settings = load_llm_settings()
        if ai_segment and not use_llm:
            raise typer.BadParameter("--ai-segment 需要同时使用 --use-llm")
        if use_llm:
            if not llm_settings.base_url or not llm_settings.api_key or not llm_settings.text_model:
                raise typer.BadParameter(
                    "--use-llm 需要 --base-url/--api-key/--text-model "
                    "或环境变量 BILICOURSE_BASE_URL/BILICOURSE_API_KEY/BILICOURSE_TEXT_MODEL"
                )

        typer.echo("Mode: LLM vision pipeline" if use_llm else "Mode: pre-LLM probe (metadata + subtitles only)")
        report = await fetch_video_report(
            source,
            prefer_ai_subtitle=prefer_ai_subtitle,
            progress=_progress,
        )
        report.parts = segment_report_parts(report.parts, window_seconds=window_seconds)

        if use_llm:
            typer.echo(f"LLM text pass: max_blocks={max_llm_blocks}, max_visual_requests={max_visual_requests}")
            if ai_segment:
                visual_requests = await ai_segment_report(
                    report,
                    llm_settings,
                    window_seconds=ai_segment_window,
                    max_visual_requests=max_visual_requests,
                    request_delay=llm_request_delay,
                )
            else:
                visual_requests = await enrich_report_text(
                    report,
                    llm_settings,
                    max_blocks=max_llm_blocks,
                    max_visual_requests=max_visual_requests,
                    batch_size=llm_batch_size,
                    request_delay=llm_request_delay,
                )
            typer.echo(f"Visual requests: {len(visual_requests)}")
            frames = await capture_requested_frames(
                report,
                visual_requests,
                output_dir,
                prefer_stream=prefer_stream_frames,
            )
            ok_frames = [frame for frame in frames if not frame.error]
            typer.echo(f"Frames saved: {len(ok_frames)}/{len(frames)}")
            write_report(report, output_dir)
            analyses = await analyze_frames(
                report,
                llm_settings,
                ok_frames,
                request_delay=llm_request_delay,
            )
            typer.echo(f"Vision analyses: {len(analyses)}")

        artifacts = write_report(report, output_dir)
        typer.echo(f"JSON: {artifacts.json_path}")
        typer.echo(f"HTML: {artifacts.html_path}")

    asyncio.run(run())


@app.command("rewrite-visual-notes")
def rewrite_visual_notes_command(
    report_json: Path = typer.Argument(..., help="已有 report.json 路径"),
    output_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--output-dir", "-o", help="数据输出目录"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL"),
    api_key: str | None = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    text_model: str | None = typer.Option(None, "--text-model", help="文本模型名"),
    llm_request_delay: float = typer.Option(
        2.2,
        "--llm-request-delay",
        help="每次模型请求后的等待秒数；2.2 秒约等于低于 30 RPM",
    ),
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking/--disable-thinking",
        help="是否为支持该参数的模型开启 thinking；默认关闭以降低延迟和 token 消耗",
    ),
) -> None:
    """Rewrite existing visual analyses into concise Chinese study notes."""

    async def run() -> None:
        if base_url:
            os.environ["BILICOURSE_BASE_URL"] = base_url
        if api_key:
            os.environ["BILICOURSE_API_KEY"] = api_key
        if text_model:
            os.environ["BILICOURSE_TEXT_MODEL"] = text_model
        if enable_thinking:
            os.environ["BILICOURSE_ENABLE_THINKING"] = "true"
        else:
            os.environ.pop("BILICOURSE_ENABLE_THINKING", None)

        llm_settings = load_llm_settings()
        if not llm_settings.base_url or not llm_settings.api_key or not llm_settings.text_model:
            raise typer.BadParameter(
                "需要 --base-url/--api-key/--text-model "
                "或环境变量 BILICOURSE_BASE_URL/BILICOURSE_API_KEY/BILICOURSE_TEXT_MODEL"
            )
        if not report_json.exists():
            raise typer.BadParameter(f"report.json 不存在: {report_json}")

        report = VideoReport.model_validate(json.loads(report_json.read_text(encoding="utf-8")))
        count = await rewrite_visual_notes(report, llm_settings, request_delay=llm_request_delay)
        artifacts = write_report(report, output_dir)
        typer.echo(f"Rewritten visual notes: {count}")
        typer.echo(f"JSON: {artifacts.json_path}")
        typer.echo(f"HTML: {artifacts.html_path}")

    asyncio.run(run())


@app.command()
def serve(
    report_dir: Path = typer.Argument(..., help="报告目录，里面应包含 report.json 和 report.html"),
    host: str = typer.Option("127.0.0.1", "--host", help="本地服务监听地址"),
    port: int = typer.Option(8765, "--port", help="本地服务端口"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="数据输出目录；默认根据报告目录推断"),
    max_visual_requests: int = typer.Option(2, "--max-visual-requests", help="交互展开时最多请求的图片数"),
    prefer_stream_frames: bool = typer.Option(
        True,
        "--prefer-stream-frames/--prefer-storyboard-frames",
        help="优先从真实视频流截高清帧",
    ),
    llm_request_delay: float = typer.Option(2.2, "--llm-request-delay", help="模型请求间隔秒数"),
) -> None:
    """Serve one report directory with local expand/redo actions."""

    if not report_dir.exists() or not report_dir.is_dir():
        raise typer.BadParameter(f"报告目录不存在: {report_dir}")
    if not (report_dir / "report.json").exists():
        raise typer.BadParameter(f"report.json 不存在: {report_dir / 'report.json'}")
    if not (report_dir / "report.html").exists():
        raise typer.BadParameter(f"report.html 不存在: {report_dir / 'report.html'}")

    typer.echo(f"Serving: {report_dir.resolve()}")
    typer.echo(f"Open: http://{host}:{port}/")
    run_report_server(
        report_dir,
        host=host,
        port=port,
        output_dir=output_dir,
        max_visual_requests=max_visual_requests,
        prefer_stream_frames=prefer_stream_frames,
        request_delay=llm_request_delay,
    )


if __name__ == "__main__":
    app()
