"""Embedder Protocol.

Implementations are synchronous (the default model is CPU/ONNX); callers that
need to stay responsive should offload to a thread. ``embed_documents`` and
``embed_query`` are separated because some models (e.g. BGE) use a query-side
instruction prefix for better asymmetric retrieval.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
