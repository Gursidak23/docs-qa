"""Service layer behind the control-plane API.

Route handlers stay thin: they call a :class:`QaService`, which owns retrieval,
ingestion, and database access. A Protocol lets tests swap in an in-memory fake
with no Postgres, embeddings, or LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..config import Settings, get_settings
from ..ingest.parsers.registry import detect_source_type
from ..llm.base import ChatMessage
from ..models import AnswerResult, RetrievedChunk
from ..storage.orm import QueryLog, SourceDocument
from .schemas import (
    AskRequest,
    AskResponse,
    AskSource,
    DocumentOut,
    IngestResultOut,
    QueryLogOut,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsOut,
    Turn,
)

_SNIPPET_CHARS = 280


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\u2026"


def hit_from_chunk(rc: RetrievedChunk) -> SearchHit:
    return SearchHit(
        chunk_id=rc.chunk_id,
        document_id=rc.document_id,
        uri=rc.uri,
        title=rc.doc_title,
        heading_path=rc.heading_path,
        page_no=rc.page_no,
        score=round(rc.score, 6),
        snippet=_snippet(rc.text),
        vector_rank=rc.vector_rank,
        lexical_rank=rc.lexical_rank,
    )


def doc_out(doc: SourceDocument) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        uri=doc.uri,
        source_type=doc.source_type,
        title=doc.title,
        chunk_count=doc.chunk_count,
        version=doc.version,
        byte_size=doc.byte_size,
        status=doc.status,
    )


def querylog_out(row: QueryLog) -> QueryLogOut:
    return QueryLogOut(
        id=row.id,
        question=row.question,
        provider=row.provider,
        grounded=row.grounded,
        latency_ms=row.latency_ms,
        feedback=row.feedback,
        created_at=row.created_at,
    )


def to_chat_history(history: list[Turn] | None) -> list[ChatMessage] | None:
    """Convert API turns into LLM chat messages (or ``None`` when empty)."""
    if not history:
        return None
    return [ChatMessage(role=t.role, content=t.content) for t in history]


def ask_response(question: str, result: AnswerResult) -> AskResponse:
    return AskResponse(
        question=question,
        answer=result.answer,
        grounded=result.grounded,
        outcome=result.outcome,
        provider=result.provider,
        citations=result.citations,
        query_log_id=result.query_log_id,
        sources=[
            AskSource(
                index=s.index,
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                uri=s.uri,
                title=s.title,
                heading_path=s.heading_path,
                page_no=s.page_no,
                snippet=s.snippet,
            )
            for s in result.sources
        ],
    )


class QaService(Protocol):
    async def search(self, req: SearchRequest) -> SearchResponse: ...

    async def ask(self, req: AskRequest) -> AskResponse: ...

    def ask_stream(
        self, question: str, top_k: int | None, history: list[Turn] | None = None
    ) -> AsyncIterator[dict]: ...

    async def ingest_upload(
        self, filename: str, data: bytes, force: bool = False
    ) -> IngestResultOut: ...

    async def ingest_url(self, url: str, force: bool = False) -> IngestResultOut: ...

    async def stats(self) -> StatsOut: ...

    async def list_documents(self, limit: int) -> list[DocumentOut]: ...

    async def get_document(self, doc_id: int) -> DocumentOut | None: ...

    async def delete_document(self, doc_id: int) -> bool: ...

    async def submit_feedback(self, query_log_id: int, helpful: bool) -> bool: ...

    async def recent_queries(self, limit: int) -> list[QueryLogOut]: ...


class SqlQaService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._embedder: Any = None
        self._retriever: Any = None
        self._answerer: Any = None
        self._indexer: Any = None

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from ..factory import build_embedder

            self._embedder = build_embedder(self.settings)
        return self._embedder

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            from ..factory import build_retriever

            self._retriever = build_retriever(self.settings, self._get_embedder())
        return self._retriever

    def _get_indexer(self) -> Any:
        if self._indexer is None:
            from ..ingest.indexer import Indexer

            self._indexer = Indexer(embedder=self._get_embedder(), settings=self.settings)
        return self._indexer

    def _get_answerer(self) -> Any:
        if self._answerer is None:
            from ..factory import build_cache, build_llm, build_reranker
            from ..rag.answer import AnswerService

            self._answerer = AnswerService(
                retriever=self._get_retriever(),
                reranker=build_reranker(self.settings),
                llm=build_llm(self.settings),
                settings=self.settings,
                cache=build_cache(self.settings),
            )
        return self._answerer

    async def search(self, req: SearchRequest) -> SearchResponse:
        from ..storage.db import session_scope

        retriever = self._get_retriever()
        async with session_scope(self.settings) as session:
            hits = await retriever.search(session, req.query, top_k=req.top_k)
        return SearchResponse(query=req.query, results=[hit_from_chunk(h) for h in hits])

    async def ask(self, req: AskRequest) -> AskResponse:
        from ..storage.db import session_scope

        answerer = self._get_answerer()
        history = to_chat_history(req.history)
        async with session_scope(self.settings) as session:
            result = await answerer.answer(session, req.question, history)
        return ask_response(req.question, result)

    async def ask_stream(
        self, question: str, top_k: int | None, history: list[Turn] | None = None
    ) -> AsyncIterator[dict]:
        from ..storage.db import session_scope

        answerer = self._get_answerer()
        chat_history = to_chat_history(history)
        async with session_scope(self.settings) as session:
            async for event in answerer.answer_stream(session, question, chat_history):
                yield event

    async def ingest_upload(
        self, filename: str, data: bytes, force: bool = False
    ) -> IngestResultOut:
        from ..ingest.parsers.registry import detect_source_type
        from ..storage.db import session_scope

        indexer = self._get_indexer()
        source_type = detect_source_type(filename)
        async with session_scope(self.settings) as session:
            result = await indexer.ingest(
                session, uri=filename, source_type=source_type, data=data, force=force
            )
        return IngestResultOut(
            uri=result.uri,
            action=result.action,
            chunks=result.chunks,
            document_id=result.document_id,
        )

    async def ingest_url(self, url: str, force: bool = False) -> IngestResultOut:
        from ..ingest.url_loader import fetch_url
        from ..storage.db import session_scope

        indexer = self._get_indexer()
        data, content_type = await fetch_url(url)
        source_type = detect_source_type(url, content_type)
        async with session_scope(self.settings) as session:
            result = await indexer.ingest(
                session, uri=url, source_type=source_type, data=data, force=force
            )
        return IngestResultOut(
            uri=result.uri,
            action=result.action,
            chunks=result.chunks,
            document_id=result.document_id,
        )

    async def stats(self) -> StatsOut:
        from ..storage.db import session_scope
        from ..storage.repositories import DocumentRepository

        async with session_scope(self.settings) as session:
            repo = DocumentRepository(session)
            return StatsOut(
                documents=await repo.count_documents(),
                chunks=await repo.count_chunks(),
            )

    async def list_documents(self, limit: int) -> list[DocumentOut]:
        from ..storage.db import session_scope
        from ..storage.repositories import DocumentRepository

        async with session_scope(self.settings) as session:
            docs = await DocumentRepository(session).list_documents(limit)
            return [doc_out(d) for d in docs]

    async def get_document(self, doc_id: int) -> DocumentOut | None:
        from ..storage.db import session_scope
        from ..storage.repositories import DocumentRepository

        async with session_scope(self.settings) as session:
            doc = await DocumentRepository(session).get(doc_id)
            return doc_out(doc) if doc is not None else None

    async def delete_document(self, doc_id: int) -> bool:
        from ..storage.db import session_scope
        from ..storage.repositories import DocumentRepository

        async with session_scope(self.settings) as session:
            return await DocumentRepository(session).delete_document(doc_id)

    async def submit_feedback(self, query_log_id: int, helpful: bool) -> bool:
        from ..storage.db import session_scope
        from ..storage.repositories import QueryLogRepository

        async with session_scope(self.settings) as session:
            return await QueryLogRepository(session).set_feedback(
                query_log_id, 1 if helpful else -1
            )

    async def recent_queries(self, limit: int) -> list[QueryLogOut]:
        from ..storage.db import session_scope
        from ..storage.repositories import QueryLogRepository

        async with session_scope(self.settings) as session:
            rows = await QueryLogRepository(session).recent(limit)
            return [querylog_out(r) for r in rows]
