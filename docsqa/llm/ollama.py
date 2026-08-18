"""Optional local Ollama client (fully offline fallback) via the native chat API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ..config import LlmSettings
from .base import ChatMessage, LlmError


class OllamaClient:
    name = "ollama"

    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings

    def _payload(self, messages: list[ChatMessage], temperature: float, max_tokens: int) -> dict:
        return {
            "model": self._settings.ollama_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        parts = [
            token
            async for token in self.stream(
                messages, temperature=temperature, max_tokens=max_tokens
            )
        ]
        return "".join(parts)

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = self._payload(messages, temperature, max_tokens)
        try:
            async with (
                httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client,
                client.stream("POST", url, json=payload) as resp,
            ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content")
                    if content:
                        yield content
        except LlmError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise LlmError(f"Ollama error: {exc}") from exc
