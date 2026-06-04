from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from bilicourseai.llm import expand_block
from bilicourseai.models import KnowledgeBlock, VideoReport
from bilicourseai.paths import output_dir_for_report_dir
from bilicourseai.report import write_report_to_dir
from bilicourseai.settings import DEFAULT_DATA_DIR, LLMSettings, load_llm_settings
from bilicourseai.tree import find_block
from bilicourseai.visual_pipeline import run_visual_pipeline


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _safe_relative_path(root: Path, value: str) -> Path:
    root = root.resolve()
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise web.HTTPBadRequest(text="Path escapes report directory.")
    return candidate


def _reset_block_for_redo(block: KnowledgeBlock) -> None:
    was_branch = block.node_type == "branch"
    block.status = "skeleton"
    block.expandable = True
    block.node_type = "branch" if was_branch else "leaf"
    block.key_points = []
    block.children = []
    block.sections = []
    block.visual_requests = []
    block.frames = []
    block.visual_analyses = []


async def _expand_current_block(
    report: VideoReport,
    block_id: str,
    settings: LLMSettings,
    output_dir: Path,
    report_dir: Path,
    max_visual_requests: int,
    prefer_stream_frames: bool,
    request_delay: float,
) -> dict[str, Any]:
    _log(f"{block_id}: calling text LLM expand")
    visual_requests = await expand_block(
        report,
        block_id,
        settings,
        max_visual_requests=max_visual_requests,
        request_delay=request_delay,
    )
    _log(f"{block_id}: text expand done, visual_requests={len(visual_requests)}")

    return await run_visual_pipeline(
        report,
        visual_requests,
        settings,
        output_dir,
        report_dir=report_dir,
        prefer_stream_frames=prefer_stream_frames,
        request_delay=request_delay,
        progress=lambda message: _log(f"{block_id}: {message}"),
    )


def create_report_app(
    report_dir: Path,
    *,
    output_dir: Path | None = None,
    max_visual_requests: int = 2,
    prefer_stream_frames: bool = True,
    request_delay: float = 2.2,
) -> web.Application:
    report_dir = report_dir.resolve()
    output_dir = output_dir or output_dir_for_report_dir(report_dir, DEFAULT_DATA_DIR)
    lock = asyncio.Lock()

    @web.middleware
    async def json_errors(request: web.Request, handler):
        try:
            return await handler(request)
        except web.HTTPException as exc:
            if request.path.startswith("/api/"):
                return web.json_response(
                    {"ok": False, "error": exc.text or exc.reason},
                    status=exc.status,
                )
            raise
        except Exception as exc:
            if request.path.startswith("/api/"):
                return web.json_response({"ok": False, "error": str(exc)}, status=500)
            raise

    app = web.Application(middlewares=[json_errors])

    async def index(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(report_dir / "report.html")

    async def static_file(request: web.Request) -> web.StreamResponse:
        relative = request.match_info["path"]
        path = _safe_relative_path(report_dir, relative)
        if not path.exists() or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def mutate(request: web.Request, *, redo: bool) -> web.Response:
        started = time.monotonic()
        payload = await request.json()
        report_json_value = str(payload.get("report") or "report.json")
        block_id = str(payload.get("block_id") or "").strip()
        action = "redo" if redo else "expand"
        if not block_id:
            raise web.HTTPBadRequest(text="block_id is required.")

        report_json = _safe_relative_path(report_dir, report_json_value)
        if report_json.name != "report.json" or not report_json.exists():
            raise web.HTTPBadRequest(text="report must be a relative report.json path.")

        settings = load_llm_settings()
        if not settings.base_url or not settings.api_key or not settings.text_model:
            raise web.HTTPBadRequest(text="LLM settings are incomplete. Run bilicourse config llm first.")

        if lock.locked():
            _log(f"{action} {block_id}: queued, another request is running")
        else:
            _log(f"{action} {block_id}: received")

        async with lock:
            _log(f"{action} {block_id}: started")
            report = VideoReport.model_validate(json.loads(report_json.read_text(encoding="utf-8")))
            block = find_block(report, block_id)
            if block is None:
                raise web.HTTPNotFound(text=f"Block not found: {block_id}")
            if redo:
                _log(f"{action} {block_id}: reset current node state")
                _reset_block_for_redo(block)

            result = await _expand_current_block(
                report,
                block_id,
                settings,
                output_dir,
                report_dir,
                max_visual_requests=max_visual_requests,
                prefer_stream_frames=prefer_stream_frames,
                request_delay=request_delay,
            )
            artifacts = write_report_to_dir(report, report_dir)
            elapsed = time.monotonic() - started
            _log(
                f"{action} {block_id}: wrote report.html, "
                f"frames={result.get('frames', 0)}, analyses={result.get('analyses', 0)}, "
                f"elapsed={elapsed:.1f}s"
            )

        return web.json_response(
            {
                "ok": True,
                "action": "redo" if redo else "expand",
                "block_id": block_id,
                "report": artifacts.json_path.name,
                "html": artifacts.html_path.name,
                **result,
            }
        )

    async def expand(request: web.Request) -> web.Response:
        return await mutate(request, redo=False)

    async def redo(request: web.Request) -> web.Response:
        return await mutate(request, redo=True)

    app.router.add_get("/", index)
    app.router.add_post("/api/expand", expand)
    app.router.add_post("/api/redo", redo)
    app.router.add_get("/{path:.*}", static_file)
    return app


def run_report_server(
    report_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: Path | None = None,
    max_visual_requests: int = 2,
    prefer_stream_frames: bool = True,
    request_delay: float = 2.2,
) -> None:
    _log(f"Serving report: {report_dir.resolve()}")
    _log(f"Open: http://{host}:{port}/")
    app = create_report_app(
        report_dir,
        output_dir=output_dir,
        max_visual_requests=max_visual_requests,
        prefer_stream_frames=prefer_stream_frames,
        request_delay=request_delay,
    )
    web.run_app(app, host=host, port=port, print=None)
