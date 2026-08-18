"""Cross-encoder reranking of fused candidates (local, free, via fastembed).

A bi-encoder (the embedder) is fast but coarse; a cross-encoder jointly reads
the query and each candidate for a much sharper relevance score. We rerank only
the small fused candidate set, so the cost stays low.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import RerankSettings
from ..models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Pass-through reranker (keeps fused order); used when reranking is disabled."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_n]


class CrossEncoderReranker:
    def __init__(self, settings: RerankSettings) -> None:
        self._settings = settings
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self._settings.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._get_model()
        scores = list(model.rerank(query, [c.text for c in candidates]))
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]
