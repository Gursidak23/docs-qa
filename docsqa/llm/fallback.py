"""Resilience wrapper: retry the primary LLM, then fail over to a secondary.

Free tiers are rate-limited, so the demo must degrade gracefully. ``complete``
retries the primary with exponential backoff before failing over. ``stream``
fails over only if the primary errors *before the first token* (so the user
never sees a half-answer get replaced); once tokens flow we commit to it.

``name`` reports the provider that actually served the current task's last call
(not the primary), so answers and metrics are attributed correctly after a
fail over. It is tracked in a :class:`~contextvars.ContextVar` rather than on
the instance because a single client is shared by concurrent requests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar

from ..logging_setup import get_logger
from ..metrics import LLM_FALLBACKS
from .base import ChatMessage, LlmClient, LlmError

log = get_logger(__name__)

# Provider that served the current task's most recent successful call.
_active_provider: ContextVar[str | None] = ContextVar("docsqa_active_provider", default=None)


class FallbackLlmClient:
    def __init__(
        self,
        primary: LlmClient,
        secondary: LlmClient | None = None,
        *,
        max_retries: int = 3,
    ) -> None:
        self._clients = [primary] + ([secondary] if secondary is not None else [])
        self._max_retries = max(1, max_retries)
        self._primary_name = primary.name

    @property
    def name(self) -> str:
        """Provider that served this task's last call; the primary before any call."""
        return _active_provider.get() or self._primary_name

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        last_error: Exception | None = None
        for position, client in enumerate(self._clients):
            for attempt in range(self._max_retries):
                try:
                    text = await client.complete(
                        messages, temperature=temperature, max_tokens=max_tokens
                    )
                except Exception as exc:  # noqa: BLE001 - retry/fail over on any error
                    last_error = exc
                    if attempt + 1 < self._max_retries:
                        await asyncio.sleep(0.5 * (2**attempt))
                    continue
                _active_provider.set(client.name)
                return text
            if position + 1 < len(self._clients):
                nxt = self._clients[position + 1]
                LLM_FALLBACKS.labels(from_provider=client.name, to_provider=nxt.name).inc()
                log.warning("llm_fallback", from_provider=client.name, to_provider=nxt.name)
        raise LlmError(f"All LLM providers failed; last error: {last_error}")

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        for position, client in enumerate(self._clients):
            agen = client.stream(messages, temperature=temperature, max_tokens=max_tokens)
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                return  # provider succeeded but produced no tokens
            except Exception as exc:  # noqa: BLE001 - fail over before first token
                if position + 1 < len(self._clients):
                    nxt = self._clients[position + 1]
                    LLM_FALLBACKS.labels(from_provider=client.name, to_provider=nxt.name).inc()
                    log.warning("llm_fallback", from_provider=client.name, to_provider=nxt.name)
                    continue
                raise LlmError(f"All LLM providers failed; last error: {exc}") from exc

            _active_provider.set(client.name)
            yield first
            async for token in agen:
                yield token
            return
