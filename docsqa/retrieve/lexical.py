"""Lexical retrieval over Postgres full-text search (BM25-like ts_rank_cd).

``websearch_to_tsquery`` accepts human-friendly queries (quotes, OR, -negation)
and degrades gracefully on empty/garbage input. ``ts_rank_cd`` is cover-density
ranking; for true BM25 swap in the ParadeDB ``pg_search`` extension behind this
same interface.
"""

from __future__ import annotations

import time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..metrics import RETRIEVE_LATENCY
from ..models import RetrievedChunk
from ..storage.orm import Chunk, SourceDocument


class LexicalRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(
        self, session: AsyncSession, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.text,
                Chunk.heading_path,
                Chunk.page_no,
                SourceDocument.uri,
                SourceDocument.title,
                rank,
            )
            .join(SourceDocument, Chunk.document_id == SourceDocument.id)
            .where(Chunk.tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).all()
        RETRIEVE_LATENCY.labels(stage="lexical").observe(time.perf_counter() - start)
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                text=row.text,
                heading_path=row.heading_path,
                page_no=row.page_no,
                uri=row.uri,
                doc_title=row.title,
                score=float(row.rank),
                lexical_rank=position,
            )
            for position, row in enumerate(rows)
        ]
