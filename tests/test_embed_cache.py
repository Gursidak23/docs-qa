from __future__ import annotations

from docsqa.embed.cache import CachingEmbedder


class CountingEmbedder:
    def __init__(self) -> None:
        self.query_calls = 0
        self.doc_calls = 0

    @property
    def dim(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [float(len(text))] * self.dim


def test_query_embeddings_are_memoized() -> None:
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner)
    first = embedder.embed_query("hello")
    second = embedder.embed_query("hello")
    assert first == second
    assert inner.query_calls == 1  # served from cache the second time
    embedder.embed_query("different")
    assert inner.query_calls == 2


def test_document_embeddings_pass_through() -> None:
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner)
    embedder.embed_documents(["a", "b"])
    embedder.embed_documents(["a", "b"])
    assert inner.doc_calls == 2  # documents are never cached
    assert embedder.dim == 4


def test_lru_eviction_drops_oldest() -> None:
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner, max_entries=1)
    embedder.embed_query("a")
    embedder.embed_query("b")  # evicts "a"
    embedder.embed_query("a")  # recomputed
    assert inner.query_calls == 3
