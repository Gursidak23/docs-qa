"""Shared test helpers."""

from __future__ import annotations

import hashlib
import math
import random

from docsqa.storage.orm import EMBED_DIM


class HashEmbedder:
    """Deterministic, network-free embedder for tests (unit-normalized vectors).

    Identical text always maps to the same vector, so cosine similarity is
    meaningful in retrieval tests without downloading a real model.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rnd = random.Random(seed)
        values = [rnd.uniform(-1.0, 1.0) for _ in range(self._dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class CountingEmbedder:
    """Wraps :class:`HashEmbedder` and records how many texts get embedded.

    Used to prove that incremental re-indexing only re-embeds changed chunks.
    """

    def __init__(self) -> None:
        self._inner = HashEmbedder()
        self.embedded_counts: list[int] = []

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def total_embedded(self) -> int:
        return sum(self.embedded_counts)

    def reset(self) -> None:
        self.embedded_counts.clear()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_counts.append(len(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)
