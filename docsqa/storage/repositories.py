"""Data-access helpers over the ORM models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Chunk as ChunkData
from .orm import Chunk, QueryLog, SourceDocument


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_uri(self, uri: str) -> SourceDocument | None:
        result = await self.session.execute(
            select(SourceDocument).where(SourceDocument.uri == uri)
        )
        return result.scalar_one_or_none()

    async def get(self, document_id: int) -> SourceDocument | None:
        return await self.session.get(SourceDocument, document_id)

    async def list_documents(self, limit: int = 100) -> Sequence[SourceDocument]:
        result = await self.session.execute(
            select(SourceDocument).order_by(SourceDocument.id.desc()).limit(limit)
        )
        return result.scalars().all()

    async def delete_document(self, document_id: int) -> bool:
        doc = await self.session.get(SourceDocument, document_id)
        if doc is None:
            return False
        await self.session.delete(doc)
        return True

    async def delete_chunks(self, document_id: int) -> int:
        result = await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        return cast("CursorResult[Any]", result).rowcount or 0

    async def add_chunks(
        self,
        document_id: int,
        chunks: list[ChunkData],
        embeddings: list[list[float]],
    ) -> int:
        rows = [
            Chunk(
                document_id=document_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                heading_path=chunk.heading_path,
                page_no=chunk.page_no,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_count=chunk.token_count,
                chunk_hash=chunk.chunk_hash,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.session.add_all(rows)
        return len(rows)

    async def chunk_embeddings_by_hash(self, document_id: int) -> dict[str, list[float]]:
        """Map ``chunk_hash -> embedding`` for a document (for incremental reuse)."""
        result = await self.session.execute(
            select(Chunk.chunk_hash, Chunk.embedding).where(Chunk.document_id == document_id)
        )
        out: dict[str, list[float]] = {}
        for chunk_hash, embedding in result.all():
            if embedding is not None:
                out[chunk_hash] = [float(x) for x in embedding]
        return out

    async def count_documents(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(SourceDocument))
        return int(result.scalar_one())

    async def count_chunks(self) -> int:
        result = await self.session.execute(select(func.count()).select_from(Chunk))
        return int(result.scalar_one())

    async def corpus_signature(self) -> str:
        """A cheap fingerprint of the corpus that changes on any add/update/delete.

        Folded into the answer-cache key so re-ingesting content transparently
        invalidates stale cached answers (no explicit per-key eviction needed).
        """
        result = await self.session.execute(
            select(
                func.count(SourceDocument.id),
                func.max(SourceDocument.updated_at),
                func.max(SourceDocument.id),
            )
        )
        count, max_updated, max_id = result.one()
        return f"{count}:{max_updated}:{max_id}"


class QueryLogRepository:
    """Read/write access to the ``query_log`` table (analytics + feedback)."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        question: str,
        answer: str | None,
        provider: str | None,
        latency_ms: int | None,
        retrieved_chunk_ids: list[int] | None,
        grounded: bool | None,
    ) -> int:
        row = QueryLog(
            question=question,
            answer=answer,
            provider=provider,
            latency_ms=latency_ms,
            retrieved_chunk_ids=retrieved_chunk_ids,
            grounded=grounded,
        )
        self.session.add(row)
        await self.session.flush()
        return row.id

    async def set_feedback(self, query_log_id: int, feedback: int) -> bool:
        row = await self.session.get(QueryLog, query_log_id)
        if row is None:
            return False
        row.feedback = feedback
        await self.session.flush()
        return True

    async def recent(self, limit: int = 50) -> Sequence[QueryLog]:
        result = await self.session.execute(
            select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
        )
        return result.scalars().all()
