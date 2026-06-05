from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parent


def _path_from_env(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "src" / "bilicourseai").is_dir()


def _default_home_dir() -> Path:
    explicit_home = _path_from_env("BILICOURSE_HOME")
    if explicit_home is not None:
        return explicit_home

    cwd = Path.cwd().resolve()
    if _looks_like_project_root(cwd):
        return cwd
    if _looks_like_project_root(PROJECT_ROOT):
        return PROJECT_ROOT

    return Path.home() / ".bilicourseai"


BILICOURSE_HOME = _default_home_dir()
DEFAULT_DATA_DIR = _path_from_env("BILICOURSE_DATA_DIR") or (BILICOURSE_HOME / "data")
AUTH_DATA_DIR = DEFAULT_DATA_DIR / "auth"
CONFIG_DIR = _path_from_env("BILICOURSE_CONFIG_DIR") or (BILICOURSE_HOME / "config")
BILIBILI_CREDENTIAL_FILE = CONFIG_DIR / "bilibili_credentials.json"
LLM_SETTINGS_FILE = CONFIG_DIR / "llm_settings.json"
DEFAULT_TEXT_MODEL = "SDU-AI/DeepSeek-V4-Flash"
DEFAULT_VISION_MODEL = "Ali-dashscope/Qwen3.5-Plus"


@dataclass(frozen=True)
class LLMSettings:
    base_url: str | None
    api_key: str | None
    text_base_url: str | None
    text_api_key: str | None
    vision_base_url: str | None
    vision_api_key: str | None
    text_model: str | None
    vision_model: str | None
    enable_thinking: bool = False

    @property
    def effective_text_base_url(self) -> str | None:
        return self.text_base_url or self.base_url

    @property
    def effective_text_api_key(self) -> str | None:
        return self.text_api_key or self.api_key

    @property
    def effective_vision_base_url(self) -> str | None:
        return self.vision_base_url or self.base_url

    @property
    def effective_vision_api_key(self) -> str | None:
        return self.vision_api_key or self.api_key


@dataclass(frozen=True)
class BilibiliCredentialSettings:
    sessdata: str | None
    bili_jct: str | None
    dedeuserid: str | None
    buvid3: str | None


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _load_bilibili_credentials_file() -> dict[str, str]:
    if not BILIBILI_CREDENTIAL_FILE.exists():
        return {}
    try:
        data = json.loads(BILIBILI_CREDENTIAL_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def _load_llm_settings_file() -> dict[str, str]:
    if not LLM_SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(LLM_SETTINGS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value is not None}


def _env_bool(name: str) -> bool | None:
    value = _nonempty(os.getenv(name))
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def _file_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


def load_llm_settings() -> LLMSettings:
    file_values = _load_llm_settings_file()
    enable_thinking_env = _env_bool("BILICOURSE_ENABLE_THINKING")
    base_url = _nonempty(os.getenv("BILICOURSE_BASE_URL")) or _nonempty(file_values.get("base_url"))
    api_key = _nonempty(os.getenv("BILICOURSE_API_KEY")) or _nonempty(file_values.get("api_key"))
    return LLMSettings(
        base_url=base_url,
        api_key=api_key,
        text_base_url=_nonempty(os.getenv("BILICOURSE_TEXT_BASE_URL"))
        or _nonempty(file_values.get("text_base_url"))
        or base_url,
        text_api_key=_nonempty(os.getenv("BILICOURSE_TEXT_API_KEY"))
        or _nonempty(file_values.get("text_api_key"))
        or api_key,
        vision_base_url=_nonempty(os.getenv("BILICOURSE_VISION_BASE_URL"))
        or _nonempty(file_values.get("vision_base_url"))
        or base_url,
        vision_api_key=_nonempty(os.getenv("BILICOURSE_VISION_API_KEY"))
        or _nonempty(file_values.get("vision_api_key"))
        or api_key,
        text_model=_nonempty(os.getenv("BILICOURSE_TEXT_MODEL"))
        or _nonempty(file_values.get("text_model"))
        or DEFAULT_TEXT_MODEL,
        vision_model=_nonempty(os.getenv("BILICOURSE_VISION_MODEL"))
        or _nonempty(file_values.get("vision_model"))
        or DEFAULT_VISION_MODEL,
        enable_thinking=(
            enable_thinking_env
            if enable_thinking_env is not None
            else (_file_bool(file_values.get("enable_thinking")) or False)
        ),
    )


def save_llm_settings(settings: LLMSettings) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_url": settings.base_url or "",
        "api_key": settings.api_key or "",
        "text_base_url": settings.text_base_url or "",
        "text_api_key": settings.text_api_key or "",
        "vision_base_url": settings.vision_base_url or "",
        "vision_api_key": settings.vision_api_key or "",
        "text_model": settings.text_model or DEFAULT_TEXT_MODEL,
        "vision_model": settings.vision_model or DEFAULT_VISION_MODEL,
        "enable_thinking": settings.enable_thinking,
    }
    LLM_SETTINGS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return LLM_SETTINGS_FILE


def clear_llm_settings() -> bool:
    if not LLM_SETTINGS_FILE.exists():
        return False
    LLM_SETTINGS_FILE.unlink()
    return True


def load_bilibili_credential_settings() -> BilibiliCredentialSettings:
    file_values = _load_bilibili_credentials_file()
    return BilibiliCredentialSettings(
        sessdata=_nonempty(os.getenv("BILICOURSE_SESSDATA")) or _nonempty(file_values.get("sessdata")),
        bili_jct=_nonempty(os.getenv("BILICOURSE_BILI_JCT")) or _nonempty(file_values.get("bili_jct")),
        dedeuserid=_nonempty(os.getenv("BILICOURSE_DEDEUSERID")) or _nonempty(file_values.get("dedeuserid")),
        buvid3=_nonempty(os.getenv("BILICOURSE_BUVID3")) or _nonempty(file_values.get("buvid3")),
    )


def save_bilibili_credential_settings(settings: BilibiliCredentialSettings) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "sessdata": settings.sessdata or "",
        "bili_jct": settings.bili_jct or "",
        "dedeuserid": settings.dedeuserid or "",
        "buvid3": settings.buvid3 or "",
    }
    BILIBILI_CREDENTIAL_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return BILIBILI_CREDENTIAL_FILE


def clear_bilibili_credential_settings() -> bool:
    if not BILIBILI_CREDENTIAL_FILE.exists():
        return False
    BILIBILI_CREDENTIAL_FILE.unlink()
    return True
