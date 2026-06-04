from __future__ import annotations

import re
import asyncio
from collections.abc import Callable
from typing import Any

import aiohttp
from bilibili_api import video

from bilicourseai.models import SubtitleTrack, TranscriptLine, VideoPart, VideoReport
from bilicourseai.source.auth import build_credential


BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10,})")


def extract_bvid(value: str) -> str:
    match = BVID_RE.search(value)
    if not match:
        raise ValueError(f"无法从输入中识别 BVID: {value}")
    return match.group(1)


def _normalize_subtitle_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return "https://aisubtitle.hdslb.com" + url
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


async def _fetch_json(url: str) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.bilibili.com",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.json(content_type=None)


def _parse_transcript(payload: dict[str, Any]) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for item in payload.get("body", []):
        start = float(item.get("from", 0))
        end = float(item.get("to", start))
        text = str(item.get("content", "")).strip()
        if text:
            lines.append(TranscriptLine(start=start, end=end, text=text))
    return lines


def _is_ai_track(lan_doc: str) -> bool:
    value = lan_doc.lower()
    return any(token in value for token in ["ai", "auto", "自动", "机器", "智能", "自动生成"])


def _is_zh_track(track: SubtitleTrack) -> bool:
    lan = track.lan.lower()
    lan_doc = track.lan_doc.lower()
    return "zh" in lan or "中文" in lan_doc or "chinese" in lan_doc


def _normalize_tracks(subtitle_info: dict[str, Any] | None) -> list[SubtitleTrack]:
    if not subtitle_info:
        return []

    raw_subtitles = subtitle_info.get("subtitles") or subtitle_info.get("subtitle", {}).get("subtitles", [])
    tracks: list[SubtitleTrack] = []
    for item in raw_subtitles:
        if not isinstance(item, dict):
            continue
        lan = str(item.get("lan") or "").strip()
        lan_doc = str(item.get("lan_doc") or "").strip()
        subtitle_url = _normalize_subtitle_url(str(item.get("subtitle_url") or item.get("url") or ""))
        tracks.append(
            SubtitleTrack(
                lan=lan,
                lan_doc=lan_doc,
                is_ai=_is_ai_track(lan_doc),
                subtitle_url=subtitle_url,
            )
        )
    return tracks


def _candidate_tracks(tracks: list[SubtitleTrack], prefer_ai: bool) -> list[SubtitleTrack]:
    ai_zh = [track for track in tracks if _is_zh_track(track) and track.is_ai]
    human_zh = [track for track in tracks if _is_zh_track(track) and not track.is_ai]
    other_ai = [track for track in tracks if not _is_zh_track(track) and track.is_ai]
    other = [track for track in tracks if not track.is_ai and not _is_zh_track(track)]

    if prefer_ai:
        return ai_zh + other_ai + human_zh + other
    return human_zh + ai_zh + other + other_ai


async def _fetch_best_transcript(
    tracks: list[SubtitleTrack], prefer_ai: bool
) -> tuple[list[TranscriptLine], SubtitleTrack | None, list[str]]:
    errors: list[str] = []
    for track in _candidate_tracks(tracks, prefer_ai=prefer_ai):
        if not track.subtitle_url:
            continue
        try:
            payload = await _fetch_json(track.subtitle_url)
            transcript = _parse_transcript(payload)
            if transcript:
                return transcript, track, errors
        except Exception as exc:
            name = track.lan_doc or track.lan or track.subtitle_url
            errors.append(f"{name}: {exc}")
    return [], None, errors


async def _with_timeout(coro, label: str, timeout_seconds: int = 45):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutError(f"{label} 超时（>{timeout_seconds}s）") from exc


async def fetch_video_report(
    source: str,
    prefer_ai_subtitle: bool = True,
    part_page: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> VideoReport:
    def emit(message: str) -> None:
        if progress:
            progress(message)

    bvid = extract_bvid(source)
    emit(f"BVID: {bvid}")
    credential = build_credential()
    v = video.Video(bvid=bvid, credential=credential)
    emit("Fetching video info...")
    info = await _with_timeout(v.get_info(), "获取视频信息")
    emit("Fetching video pages...")
    pages = await _with_timeout(v.get_pages(), "获取分P列表")
    emit(f"Pages: {len(pages)}")

    parts: list[VideoPart] = []
    selected_pages = pages
    if part_page is not None:
        selected_pages = [page for page in pages if int(page.get("page", 0)) == part_page]
        if not selected_pages:
            raise ValueError(f"视频没有 P{part_page}")
        emit(f"Selected P{part_page}: fetching subtitles only for this part")

    for page in selected_pages:
        cid = int(page["cid"])
        page_no = int(page.get("page", len(parts) + 1))
        title = str(page.get("part") or page.get("title") or f"P{len(parts) + 1}")
        emit(f"P{page_no}: fetching subtitles - {title}")
        part = VideoPart(
            page=page_no,
            cid=cid,
            title=title,
            duration=page.get("duration"),
        )

        if credential is None:
            part.subtitle_errors.append("未配置 Bilibili 登录凭据，跳过字幕接口。")
        else:
            try:
                subtitle_info = await _with_timeout(v.get_subtitle(cid=cid), f"P{part.page} 获取字幕列表")
                part.subtitle_tracks = _normalize_tracks(subtitle_info)
                (
                    part.transcript,
                    part.selected_subtitle_track,
                    part.subtitle_errors,
                ) = await _fetch_best_transcript(part.subtitle_tracks, prefer_ai=prefer_ai_subtitle)
                emit(f"P{part.page}: transcript lines={len(part.transcript)} tracks={len(part.subtitle_tracks)}")
            except Exception as exc:
                part.subtitle_errors.append(f"获取字幕列表失败: {exc}")
                emit(f"P{part.page}: subtitle failed - {exc}")

        parts.append(part)

    owner = info.get("owner") or {}
    return VideoReport(
        bvid=bvid,
        aid=info.get("aid"),
        title=str(info.get("title") or bvid),
        owner_name=owner.get("name"),
        source_url=f"https://www.bilibili.com/video/{bvid}",
        parts=parts,
    )
