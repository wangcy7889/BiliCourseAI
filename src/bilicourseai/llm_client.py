from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from bilicourseai.settings import LLMSettings


def create_client(settings: LLMSettings) -> AsyncOpenAI:
    if not settings.base_url or not settings.api_key:
        raise ValueError("LLM requires base_url and api_key.")
    return AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)


def extra_body(settings: LLMSettings) -> dict[str, Any]:
    return {"enable_thinking": settings.enable_thinking}
