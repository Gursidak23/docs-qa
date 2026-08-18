"""Hybrid retrieval: fuse dense + lexical candidates with Reciprocal Rank Fusion.

RRF combines ranked lists without needing comparable raw scores: each item gets
``sum(1 / (k + rank))`` across the lists it appears in. It is robust, has a
single tunable constant ``k``, and reliably beats either arm alone.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..embed.base import Embedder
from ..metrics import RETRIEVE_LATENCY
from ..models import RetrievedChunk
from .lexical import LexicalRetriever
from .vector import VectorRetriever


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedChunk]], k: int
) -> list[RetrievedChunk]:
    """Merge ranked lists by chunk id; returns items sorted by fused score desc."""
    fused: dict[int, float] = {}
    merged: dict[int, RetrievedChunk] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if item.chunk_id not in merged:
                merged[item.chunk_id] = item
            else:
                existing = merged[item.chunk_id]
                if item.vector_rank is not None:
                    existing.vector_rank = item.vector_rank
                if item.lexical_rank is not None:
                    existing.lexical_rank = item.lexical_rank

    ordered = sorted(merged.values(), key=lambda r: fused[r.chunk_id], reverse=True)
    for item in ordered:
        item.score = fused[item.chunk_id]
    return ordered


class HybridRetriever:
    def __init__(self, embedder: Embedder, settings: Settings) -> None:
        self.settings = settings
        self.vector = VectorRetriever(embedder, settings)
        self.lexical = LexicalRetriever(settings)

    async def search(
        self, session: AsyncSession, query: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        rs = self.settings.retrieve
        start = time.perf_counter()
        vector_hits = await self.vector.search(session, query, rs.vector_top_k)
        lexical_hits = await self.lexical.search(session, query, rs.lexical_top_k)
        fused = reciprocal_rank_fusion([vector_hits, lexical_hits], rs.rrf_k)
        RETRIEVE_LATENCY.labels(stage="hybrid").observe(time.perf_counter() - start)
        limit = top_k or rs.fused_top_k
        return fused[:limit]
