"""Async token-bucket rate limiting for LLM calls.

Free provider tiers cap requests-per-minute; bursting past that earns 429s. The
:class:`AsyncTokenBucket` smooths call rate to a configured budget (with a small
burst allowance), and :class:`RateLimitedLlmClient` transparently wraps any
:class:`~docsqa.llm.base.LlmClient` so both the CLI and API benefit.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic

from .base import ChatMessage, LlmClient


class AsyncTokenBucket:
    def __init__(self, rate_per_minute: float, capacity: float | None = None) -> None:
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_minute / 6.0)
        self._tokens = self.capacity
        self._updated = monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate if self.rate > 0 else 0.05
            await asyncio.sleep(wait)


class RateLimitedLlmClient:
    def __init__(self, inner: LlmClient, bucket: AsyncTokenBucket) -> None:
        self.inner = inner
        self.bucket = bucket

    @property
    def name(self) -> str:
        """Delegate so a wrapped fail-over client can report its serving provider."""
        return self.inner.name

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        await self.bucket.acquire()
        return await self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        await self.bucket.acquire()
        async for token in self.inner.stream(
            messages, temperature=temperature, max_tokens=max_tokens
        ):
            yield token
