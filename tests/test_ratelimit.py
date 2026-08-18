from __future__ import annotations

from collections.abc import AsyncIterator

import docsqa.llm.ratelimit as rl
from docsqa.llm.base import ChatMessage
from docsqa.llm.ratelimit import AsyncTokenBucket, RateLimitedLlmClient


async def test_token_bucket_throttles_after_burst(monkeypatch) -> None:
    clock = {"t": 0.0}
    sleeps: list[float] = []
    monkeypatch.setattr(rl, "monotonic", lambda: clock["t"])

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(rl.asyncio, "sleep", fake_sleep)

    bucket = AsyncTokenBucket(rate_per_minute=60, capacity=1)  # 1 token/sec, starts full
    await bucket.acquire()  # consumes the initial token, no sleep
    await bucket.acquire()  # empty -> must wait ~1s for a refill
    assert sleeps and abs(sleeps[0] - 1.0) < 1e-6


class FakeLlm:
    name = "fake"

    def __init__(self) -> None:
        self.completes = 0

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        self.completes += 1
        return "ok"

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        for token in ("a", "b"):
            yield token


async def test_rate_limited_client_delegates() -> None:
    inner = FakeLlm()
    client = RateLimitedLlmClient(inner, AsyncTokenBucket(rate_per_minute=100000, capacity=100))
    assert client.name == "fake"
    msgs = [ChatMessage("user", "hi")]
    assert await client.complete(msgs, temperature=0.0, max_tokens=10) == "ok"
    assert inner.completes == 1
    tokens = [t async for t in client.stream(msgs, temperature=0.0, max_tokens=10)]
    assert tokens == ["a", "b"]
