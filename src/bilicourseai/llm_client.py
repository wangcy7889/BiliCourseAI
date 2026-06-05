from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from bilicourseai.settings import LLMSettings


def create_client(settings: LLMSettings, *, role: str = "text") -> AsyncOpenAI:
    if role == "vision":
        base_url = settings.effective_vision_base_url
        api_key = settings.effective_vision_api_key
    else:
        base_url = settings.effective_text_base_url
        api_key = settings.effective_text_api_key
    if not base_url or not api_key:
        raise ValueError(f"LLM {role} client requires base_url and api_key.")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def extra_body(settings: LLMSettings) -> dict[str, Any]:
    return {"enable_thinking": settings.enable_thinking}
