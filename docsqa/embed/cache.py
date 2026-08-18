"""An :class:`Embedder` decorator that memoizes query embeddings.

Query embeddings are recomputed on every search; repeated/popular questions are
common, so a small in-process LRU avoids redundant model inference. Document
embeddings (bulk ingest) are unique and handled by incremental re-indexing, so
they are passed straight through.
"""

from __future__ import annotations

from collections import OrderedDict

from ..metrics import CACHE_EVENTS
from ..text_utils import normalize_text, sha256_text
from .base import Embedder


class CachingEmbedder:
    def __init__(self, inner: Embedder, max_entries: int = 2048) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        key = sha256_text(normalize_text(text))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            CACHE_EVENTS.labels(cache="embed", event="hit").inc()
            return cached
        CACHE_EVENTS.labels(cache="embed", event="miss").inc()
        vector = self.inner.embed_query(text)
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return vector
