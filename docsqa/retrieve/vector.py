"""Dense retrieval over pgvector using cosine distance (HNSW index)."""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..embed.base import Embedder
from ..metrics import RETRIEVE_LATENCY
from ..models import RetrievedChunk
from ..storage.orm import Chunk, SourceDocument


class VectorRetriever:
    def __init__(self, embedder: Embedder, settings: Settings) -> None:
        self.embedder = embedder
        self.settings = settings

    async def search(
        self, session: AsyncSession, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Chunk.text,
                Chunk.heading_path,
                Chunk.page_no,
                SourceDocument.uri,
                SourceDocument.title,
                distance,
            )
            .join(SourceDocument, Chunk.document_id == SourceDocument.id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).all()
        RETRIEVE_LATENCY.labels(stage="vector").observe(time.perf_counter() - start)
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                text=row.text,
                heading_path=row.heading_path,
                page_no=row.page_no,
                uri=row.uri,
                doc_title=row.title,
                score=1.0 - float(row.distance),  # cosine similarity
                vector_rank=rank,
            )
            for rank, row in enumerate(rows)
        ]
