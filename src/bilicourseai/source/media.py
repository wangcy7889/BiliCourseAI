from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import aiohttp
import imageio_ffmpeg
from PIL import Image
from bilibili_api import video

from bilicourseai.models import FrameArtifact, VideoReport, VisualRequest
from bilicourseai.paths import report_dir_for
from bilicourseai.source.auth import build_credential
from bilicourseai.tree import find_block


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.bilibili.com",
}


def _normalize_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def _find_part(report: VideoReport, page: int):
    for part in report.parts:
        if part.page == page:
            return part
    return None


def _find_block(report: VideoReport, block_id: str):
    return find_block(report, block_id)


def _find_section(block, section_id: str | None):
    if not section_id:
        return None
    for section in block.sections:
        if section.id == section_id:
            return section
    return None


async def _fetch_videoshot(bvid: str, cid: int) -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.bilibili.com/x/player/videoshot",
            params={"bvid": bvid, "cid": cid},
            headers=HEADERS,
        ) as response:
            response.raise_for_status()
            return await response.json(content_type=None)


async def _fetch_stream_url(bvid: str, cid: int) -> str:
    v = video.Video(bvid=bvid, credential=build_credential())
    data = await v.get_download_url(cid=cid, html5=True)
    streams = video.VideoDownloadURLDataDetecter(data).detect()
    if not streams:
        raise RuntimeError("No playable streams found.")
    stream = streams[0]
    url = getattr(stream, "url", None)
    if not url:
        raise RuntimeError("Playable stream has no URL.")
    return str(url)


def _capture_stream_frame(stream_url: str, timestamp: float, output_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    headers = "Referer: https://www.bilibili.com\r\nUser-Agent: Mozilla/5.0\r\n"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-headers",
        headers,
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        stream_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "ffmpeg failed").strip()
        raise RuntimeError(message[:500])
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no frame.")


async def _download_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(_normalize_url(url), headers=HEADERS) as response:
            response.raise_for_status()
            return await response.read()


def _index_for_timestamp(indexes: list[Any], timestamp: float, fallback_count: int) -> int:
    numeric = [float(item) for item in indexes if isinstance(item, (int, float))]
    if numeric:
        distances = [abs(value - timestamp) for value in numeric]
        return distances.index(min(distances))
    return max(0, min(fallback_count - 1, int(timestamp // 10)))


def _crop_sprite(sprite_path: Path, frame_index: int, meta: dict[str, Any], output_path: Path) -> None:
    img_x_len = int(meta.get("img_x_len") or 10)
    img_y_len = int(meta.get("img_y_len") or 10)

    with Image.open(sprite_path) as image:
        width, height = image.size
        cell_width = int(meta.get("img_x_size") or (width // img_x_len))
        cell_height = int(meta.get("img_y_size") or (height // img_y_len))
        per_sprite = img_x_len * img_y_len
        local_index = frame_index % per_sprite
        row = local_index // img_x_len
        col = local_index % img_x_len
        left = col * cell_width
        top = row * cell_height
        crop = image.crop((left, top, left + cell_width, top + cell_height))
        crop.convert("RGB").save(output_path, "JPEG", quality=88)


def _timestamp_slug(timestamp: float) -> str:
    return f"{timestamp:.3f}".rstrip("0").rstrip(".")


async def _capture_frame_artifact(
    report: VideoReport,
    request: VisualRequest,
    part,
    timestamp: float,
    output_path: Path,
    prefer_stream: bool,
    videoshot_cache: dict[int, dict[str, Any]],
    stream_url_cache: dict[int, str],
    source_prefix: str = "",
) -> FrameArtifact:
    source = "videoshot"
    if prefer_stream:
        source = f"{source_prefix}stream" if source_prefix else "stream"
        if part.cid not in stream_url_cache:
            stream_url_cache[part.cid] = await _fetch_stream_url(report.bvid, part.cid)
        _capture_stream_frame(stream_url_cache[part.cid], timestamp, output_path)
        return FrameArtifact(
            request_id=request.id,
            part_page=part.page,
            block_id=request.block_id,
            timestamp=timestamp,
            path=str(output_path),
            source=source,
        )

    if part.cid not in videoshot_cache:
        videoshot_cache[part.cid] = await _fetch_videoshot(report.bvid, part.cid)
    payload = videoshot_cache[part.cid]
    data = payload.get("data") or {}
    images = data.get("image") or []
    if payload.get("code") != 0 or not images:
        raise RuntimeError(f"videoshot unavailable: {payload.get('message') or payload.get('code')}")

    img_x_len = int(data.get("img_x_len") or 10)
    img_y_len = int(data.get("img_y_len") or 10)
    per_sprite = img_x_len * img_y_len
    total_count = max(1, len(images) * per_sprite)
    frame_index = _index_for_timestamp(data.get("index") or [], timestamp, total_count)
    sprite_index = min(len(images) - 1, frame_index // per_sprite)

    sprite_path = output_path.parent / f"_sprite_p{part.page}_{sprite_index}.jpg"
    if not sprite_path.exists():
        sprite_path.write_bytes(await _download_bytes(str(images[sprite_index])))
    _crop_sprite(sprite_path, frame_index, data, output_path)

    source = f"{source_prefix}videoshot" if source_prefix else "videoshot"
    return FrameArtifact(
        request_id=request.id,
        part_page=part.page,
        block_id=request.block_id,
        timestamp=timestamp,
        path=str(output_path),
        source=source,
    )


async def capture_requested_frames(
    report: VideoReport,
    requests: list[VisualRequest],
    output_dir: Path,
    prefer_stream: bool = True,
    report_dir: Path | None = None,
) -> list[FrameArtifact]:
    frames_dir = (report_dir or report_dir_for(report, output_dir)) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameArtifact] = []
    videoshot_cache: dict[int, dict[str, Any]] = {}
    stream_url_cache: dict[int, str] = {}

    for request in requests:
        part = _find_part(report, request.part_page)
        block = _find_block(report, request.block_id)
        if not part or not block:
            frames.append(
                FrameArtifact(
                    request_id=request.id,
                    part_page=request.part_page,
                    block_id=request.block_id,
                    timestamp=request.timestamp,
                    path="",
                    source="videoshot",
                    error="Part or block not found.",
                )
            )
            continue

        try:
            timestamp_slug = _timestamp_slug(request.timestamp)
            suffix = "_stream" if prefer_stream else ""
            output_path = frames_dir / f"{request.id}_p{part.page}_{timestamp_slug}s{suffix}.jpg"
            frame = await _capture_frame_artifact(
                report,
                request,
                part,
                request.timestamp,
                output_path,
                prefer_stream,
                videoshot_cache,
                stream_url_cache,
            )
            block.frames.append(frame)
            section = _find_section(block, request.section_id)
            if section is not None:
                section.frames.append(frame)
            frames.append(frame)
        except Exception as exc:
            frame = FrameArtifact(
                request_id=request.id,
                part_page=part.page,
                block_id=request.block_id,
                timestamp=request.timestamp,
                path=str(output_path),
                source="videoshot",
                error=str(exc),
            )
            block.frames.append(frame)
            frames.append(frame)

    return frames


async def capture_candidate_frames(
    report: VideoReport,
    requests: list[VisualRequest],
    output_dir: Path,
    prefer_stream: bool = True,
    report_dir: Path | None = None,
) -> dict[str, list[FrameArtifact]]:
    frames_dir = (report_dir or report_dir_for(report, output_dir)) / "frames" / "_candidates"
    frames_dir.mkdir(parents=True, exist_ok=True)

    videoshot_cache: dict[int, dict[str, Any]] = {}
    stream_url_cache: dict[int, str] = {}
    by_request: dict[str, list[FrameArtifact]] = {}

    for request in requests:
        part = _find_part(report, request.part_page)
        block = _find_block(report, request.block_id)
        timestamps = request.candidate_timestamps or [request.timestamp]
        artifacts: list[FrameArtifact] = []
        by_request[request.id] = artifacts
        if not part or not block:
            artifacts.append(
                FrameArtifact(
                    request_id=request.id,
                    part_page=request.part_page,
                    block_id=request.block_id,
                    timestamp=request.timestamp,
                    path="",
                    source="candidate",
                    error="Part or block not found.",
                )
            )
            continue
        for index, timestamp in enumerate(timestamps, start=1):
            timestamp_slug = _timestamp_slug(timestamp)
            suffix = "_stream" if prefer_stream else ""
            output_path = frames_dir / f"{request.id}_c{index}_p{part.page}_{timestamp_slug}s{suffix}.jpg"
            try:
                artifacts.append(
                    await _capture_frame_artifact(
                        report,
                        request,
                        part,
                        timestamp,
                        output_path,
                        prefer_stream,
                        videoshot_cache,
                        stream_url_cache,
                        source_prefix="candidate_",
                    )
                )
            except Exception as exc:
                artifacts.append(
                    FrameArtifact(
                        request_id=request.id,
                        part_page=part.page,
                        block_id=request.block_id,
                        timestamp=timestamp,
                        path=str(output_path),
                        source="candidate",
                        error=str(exc),
                    )
                )
    return by_request


def cleanup_candidate_frames(report: VideoReport, output_dir: Path, report_dir: Path | None = None) -> bool:
    candidates_dir = (report_dir or report_dir_for(report, output_dir)) / "frames" / "_candidates"
    if not candidates_dir.exists():
        return False
    shutil.rmtree(candidates_dir)
    return True
