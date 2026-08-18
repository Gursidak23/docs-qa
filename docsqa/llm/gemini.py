"""Google Gemini client (free tier via AI Studio key) using google-genai async."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

from ..config import LlmSettings
from ..metrics import LLM_TOKENS
from .base import ChatMessage, LlmError


def _to_gemini(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    return system, contents


def _record_tokens(usage: Any) -> None:
    if usage is None:
        return
    prompt = getattr(usage, "prompt_token_count", None)
    completion = getattr(usage, "candidates_token_count", None)
    if prompt:
        LLM_TOKENS.labels(provider="gemini", direction="prompt").inc(prompt)
    if completion:
        LLM_TOKENS.labels(provider="gemini", direction="completion").inc(completion)


class GeminiClient:
    name = "gemini"

    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise LlmError("Gemini API key is not configured (DOCSQA_LLM__GEMINI_API_KEY)")
            from google import genai

            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _config(self, temperature: float, max_tokens: int, system: str | None) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        client = self._get_client()
        system, contents = _to_gemini(messages)
        try:
            resp = await client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=contents,
                config=self._config(temperature, max_tokens, system),
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"Gemini error: {exc}") from exc
        _record_tokens(getattr(resp, "usage_metadata", None))
        return resp.text or ""

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        client = self._get_client()
        system, contents = _to_gemini(messages)
        try:
            stream = client.aio.models.generate_content_stream(
                model=self._settings.gemini_model,
                contents=contents,
                config=self._config(temperature, max_tokens, system),
            )
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"Gemini stream error: {exc}") from exc
