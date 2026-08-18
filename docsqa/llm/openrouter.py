"""OpenRouter client using the OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import LlmSettings
from ..metrics import LLM_TOKENS
from .base import ChatMessage, LlmError


class OpenRouterClient:
    name = "openrouter"

    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings

    def _payload(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self._settings.openrouter_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _headers(self) -> dict[str, str]:
        if not self._settings.openrouter_api_key:
            raise LlmError("OpenRouter API key is not configured (DOCSQA_LLM__OPENROUTER_API_KEY)")
        return {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        url = f"{self._settings.openrouter_base_url.rstrip('/')}/chat/completions"
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        headers = self._headers()
        try:
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"OpenRouter error: {exc}") from exc

        usage = data.get("usage") or {}
        if usage:
            LLM_TOKENS.labels(provider="openrouter", direction="prompt").inc(
                int(usage.get("prompt_tokens", 0) or 0)
            )
            LLM_TOKENS.labels(provider="openrouter", direction="completion").inc(
                int(usage.get("completion_tokens", 0) or 0)
            )
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        url = f"{self._settings.openrouter_base_url.rstrip('/')}/chat/completions"
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens)
        payload["stream"] = True
        headers = self._headers()
        try:
            async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(data)
                        except Exception:
                            continue
                        choices = parsed.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"OpenRouter stream error: {exc}") from exc
