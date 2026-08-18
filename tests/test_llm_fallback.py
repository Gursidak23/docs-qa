from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from docsqa.config import Settings
from docsqa.factory import _build_single_llm
from docsqa.llm.base import ChatMessage, LlmError
from docsqa.llm.fallback import FallbackLlmClient

MESSAGES = [ChatMessage("user", "hi")]


class OkLlm:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        return self._text

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        for token in self._text.split():
            yield token


class FailLlm:
    def __init__(self, name: str = "fail") -> None:
        self.name = name

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        raise LlmError("boom")

    async def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        raise LlmError("boom")
        yield ""  # pragma: no cover - marks this as an async generator


async def test_complete_uses_primary_when_healthy() -> None:
    client = FallbackLlmClient(OkLlm("p", "hello"), OkLlm("s", "world"), max_retries=1)
    assert await client.complete(MESSAGES, temperature=0.0, max_tokens=10) == "hello"


async def test_complete_fails_over_to_secondary() -> None:
    client = FallbackLlmClient(FailLlm("p"), OkLlm("s", "backup"), max_retries=1)
    assert await client.complete(MESSAGES, temperature=0.0, max_tokens=10) == "backup"


async def test_complete_raises_when_all_providers_fail() -> None:
    client = FallbackLlmClient(FailLlm("p"), FailLlm("s"), max_retries=1)
    with pytest.raises(LlmError):
        await client.complete(MESSAGES, temperature=0.0, max_tokens=10)


async def test_stream_fails_over_before_first_token() -> None:
    client = FallbackLlmClient(FailLlm("p"), OkLlm("s", "a b c"), max_retries=1)
    tokens = [tok async for tok in client.stream(MESSAGES, temperature=0.0, max_tokens=10)]
    assert tokens == ["a", "b", "c"]


def test_name_reports_primary() -> None:
    assert FallbackLlmClient(OkLlm("primary", "x")).name == "primary"


def test_build_single_llm_supports_openrouter_provider() -> None:
    settings = Settings()
    settings.llm.openrouter_api_key = "test-key"
    client = _build_single_llm(settings, "openrouter")
    assert client.name == "openrouter"
