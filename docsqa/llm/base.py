"""Common LLM abstractions: chat messages, errors, and the client Protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


class LlmError(RuntimeError):
    """Raised when a provider call fails (network, auth, rate limit, timeout)."""


class LlmClient(Protocol):
    # Read-only so wrappers can compute it (e.g. the fail-over client reports
    # the provider that actually served the call) rather than store a constant.
    @property
    def name(self) -> str: ...

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str: ...

    def stream(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]: ...
