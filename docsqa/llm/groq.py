"""Groq client (free tier, OpenAI-compatible) using the async SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..config import LlmSettings
from ..metrics import LLM_TOKENS
from .base import ChatMessage, LlmError


def _to_openai(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class GroqClient:
    name = "groq"

    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._settings.groq_api_key:
                raise LlmError("Groq API key is not configured (DOCSQA_LLM__GROQ_API_KEY)")
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self._settings.groq_api_key)
        return self._client

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_client()
        try:
            resp = await client.chat.completions.create(
                model=self._settings.groq_model,
                messages=_to_openai(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"Groq error: {exc}") from exc
        usage = getattr(resp, "usage", None)
        if usage is not None:
            LLM_TOKENS.labels(provider="groq", direction="prompt").inc(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            LLM_TOKENS.labels(provider="groq", direction="completion").inc(
                getattr(usage, "completion_tokens", 0) or 0
            )
        return resp.choices[0].message.content or ""

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        client = self._get_client()
        try:
            stream = await client.chat.completions.create(
                model=self._settings.groq_model,
                messages=_to_openai(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"Groq stream error: {exc}") from exc
