from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from bilibili_api import Credential, login_v2

from bilicourseai.settings import (
    AUTH_DATA_DIR,
    BilibiliCredentialSettings,
    load_bilibili_credential_settings,
)


@dataclass
class QrLoginSession:
    login: login_v2.QrCodeLogin
    qrcode_path: Path
    terminal_qrcode: str
    created_at: float


def build_credential(settings: BilibiliCredentialSettings | None = None) -> Credential | None:
    settings = settings or load_bilibili_credential_settings()
    if not (settings.sessdata and settings.bili_jct and settings.dedeuserid):
        return None
    return Credential(
        sessdata=settings.sessdata,
        bili_jct=settings.bili_jct,
        dedeuserid=settings.dedeuserid,
        buvid3=settings.buvid3 or "",
    )


async def check_bilibili_credential(settings: BilibiliCredentialSettings | None = None) -> bool:
    credential = build_credential(settings)
    if credential is None:
        return False
    return bool(await credential.check_valid())


async def start_qr_login() -> QrLoginSession:
    qr_login = login_v2.QrCodeLogin(platform=login_v2.QrCodeLoginChannel.WEB)
    await qr_login.generate_qrcode()

    AUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    qrcode_path = AUTH_DATA_DIR / "bilibili_qrcode.png"
    qrcode_path.write_bytes(qr_login.get_qrcode_picture().content)

    return QrLoginSession(
        login=qr_login,
        qrcode_path=qrcode_path,
        terminal_qrcode=qr_login.get_qrcode_terminal(),
        created_at=time.time(),
    )


async def poll_qr_login(session: QrLoginSession) -> tuple[str, BilibiliCredentialSettings | None]:
    state = await session.login.check_state()
    if state == login_v2.QrCodeLoginEvents.DONE:
        credential = session.login.get_credential()
        cookies = credential.get_cookies()
        settings = BilibiliCredentialSettings(
            sessdata=cookies.get("SESSDATA") or cookies.get("sessdata"),
            bili_jct=cookies.get("bili_jct"),
            dedeuserid=cookies.get("DedeUserID") or cookies.get("dedeuserid"),
            buvid3=cookies.get("buvid3"),
        )
        return "success", settings
    if state == login_v2.QrCodeLoginEvents.TIMEOUT:
        return "expired", None
    if state == login_v2.QrCodeLoginEvents.CONF:
        return "confirm", None
    if state == login_v2.QrCodeLoginEvents.SCAN:
        return "scan", None
    return "waiting", None
