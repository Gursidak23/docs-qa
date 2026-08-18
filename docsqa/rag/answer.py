"""Answer orchestration: retrieve -> rerank -> prompt -> generate -> guard.

The groundedness guard parses ``[n]`` citations from the model output and keeps
only those that map to a real retrieved passage. If the model produced no valid
citation (or emitted the "I don't know" sentence), the answer is flagged
ungrounded so the UI/API can avoid presenting a hallucination as fact.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession

from ..cache import Cache, make_key, record_cache_event
from ..config import Settings
from ..llm.base import ChatMessage, LlmClient, LlmError
from ..llm.prompt import (
    IDK_TEXT,
    build_context,
    build_messages,
    build_retrieval_query,
    extract_citations,
)
from ..logging_setup import get_logger
from ..metrics import ANSWERS, GROUNDED, LLM_LATENCY, QUESTIONS
from ..models import AnswerResult, AnswerSource
from ..retrieve.hybrid import HybridRetriever
from ..retrieve.rerank import Reranker

log = get_logger(__name__)

# Keep at most this many prior turns (user+assistant messages) for follow-ups.
MAX_HISTORY_MESSAGES = 6


def _classify(text: str, valid_citations: list[int]) -> tuple[str, bool]:
    lowered = text.strip().lower()
    if "don't have enough information" in lowered or lowered.startswith("i don't know"):
        return "idk", False
    if valid_citations:
        return "answered", True
    return "ungrounded", False


def _serialize(result: AnswerResult) -> str:
    return json.dumps(
        {
            "answer": result.answer,
            "sources": [asdict(s) for s in result.sources],
            "citations": result.citations,
            "grounded": result.grounded,
            "provider": result.provider,
            "outcome": result.outcome,
        }
    )


def _deserialize(raw: str) -> AnswerResult:
    data = json.loads(raw)
    sources = [AnswerSource(**s) for s in data["sources"]]
    return AnswerResult(
        answer=data["answer"],
        sources=sources,
        citations=data["citations"],
        grounded=data["grounded"],
        provider=data["provider"],
        outcome=data["outcome"],
    )


def _clamp_history(history: list[ChatMessage] | None) -> list[ChatMessage] | None:
    """Bound conversation history to the most recent turns."""
    if not history:
        return None
    return history[-MAX_HISTORY_MESSAGES:]


def _history_parts(history: list[ChatMessage] | None) -> tuple[str, ...]:
    """History rendered as cache-key parts so contexts don't collide."""
    if not history:
        return ()
    return tuple(f"{m.role}:{m.content}" for m in history)


class AnswerService:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LlmClient,
        settings: Settings,
        cache: Cache | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.settings = settings
        self.cache = cache
        rs = settings.retrieve
        # Cache fingerprint: invalidate cached answers when retrieval/model knobs change.
        self._fingerprint = (
            f"{settings.embed.model_name}|{settings.llm.provider}|{llm.name}|"
            f"rerank={settings.rerank.enabled}:{settings.rerank.top_n}|fused={rs.fused_top_k}"
        )
        self._answer_ttl = settings.cache.answer_ttl_seconds

    async def _prepare(
        self,
        session: AsyncSession,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> tuple[list[AnswerSource], list]:
        rs = self.settings.retrieve
        retrieval_query = build_retrieval_query(question, history)
        candidates = await self.retriever.search(session, retrieval_query, top_k=rs.fused_top_k)
        if not candidates:
            return [], []
        top_n = self.settings.rerank.top_n if self.settings.rerank.enabled else rs.fused_top_k
        ranked = self.reranker.rerank(question, candidates, top_n)
        context, sources = build_context(ranked)
        messages = build_messages(question, context, history)
        return sources, messages

    async def _corpus_signature(self, session: AsyncSession) -> str:
        """Fingerprint the corpus so re-ingestion invalidates cached answers.

        Fails open (empty string) so a signature error never blocks answering.
        """
        from ..storage.repositories import DocumentRepository

        try:
            return await DocumentRepository(session).corpus_signature()
        except Exception as exc:  # noqa: BLE001 - cache freshness must not break answers
            log.warning("corpus_signature_failed", error=str(exc))
            return ""

    async def _log_query(
        self, session: AsyncSession, question: str, result: AnswerResult, started: float
    ) -> None:
        """Persist the turn to ``query_log`` and stamp the id onto ``result``.

        Best-effort: analytics/feedback logging must never break a request.
        """
        from ..storage.repositories import QueryLogRepository

        try:
            latency_ms = int((time.perf_counter() - started) * 1000)
            result.query_log_id = await QueryLogRepository(session).create(
                question=question,
                answer=result.answer,
                provider=result.provider,
                latency_ms=latency_ms,
                retrieved_chunk_ids=[s.chunk_id for s in result.sources] or None,
                grounded=result.grounded,
            )
        except Exception as exc:  # noqa: BLE001 - logging is non-essential
            log.warning("query_log_failed", error=str(exc))

    def _finalize(self, text: str, sources: list[AnswerSource], provider: str) -> AnswerResult:
        cited = extract_citations(text)
        valid = [i for i in cited if 1 <= i <= len(sources)]
        outcome, grounded = _classify(text, valid)
        used = [s for s in sources if s.index in valid] or sources
        ANSWERS.labels(outcome=outcome).inc()
        GROUNDED.labels(result="grounded" if grounded else "ungrounded").inc()
        return AnswerResult(
            answer=text,
            sources=used,
            citations=valid,
            grounded=grounded,
            provider=provider,
            outcome=outcome,
        )

    async def answer(
        self,
        session: AsyncSession,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> AnswerResult:
        QUESTIONS.inc()
        started = time.perf_counter()
        history = _clamp_history(history)

        key: str | None = None
        if self.cache is not None:
            signature = await self._corpus_signature(session)
            key = make_key(self._fingerprint, signature, question, *_history_parts(history))
            cached = await self.cache.get(key)
            if cached is not None:
                record_cache_event("answer", hit=True)
                result = _deserialize(cached)
                await self._log_query(session, question, result, started)
                return result
            record_cache_event("answer", hit=False)

        sources, messages = await self._prepare(session, question, history)
        if not sources:
            ANSWERS.labels(outcome="idk").inc()
            GROUNDED.labels(result="ungrounded").inc()
            result = AnswerResult(IDK_TEXT, [], [], False, self.llm.name, "idk")
            await self._log_query(session, question, result, started)
            return result

        start = time.perf_counter()
        text = await self.llm.complete(
            messages,
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_output_tokens,
        )
        LLM_LATENCY.labels(provider=self.llm.name).observe(time.perf_counter() - start)
        result = self._finalize(text, sources, self.llm.name)

        if self.cache is not None and key is not None and result.outcome == "answered":
            await self.cache.set(key, _serialize(result), self._answer_ttl)
        await self._log_query(session, question, result, started)
        return result

    async def answer_stream(
        self,
        session: AsyncSession,
        question: str,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[dict]:
        QUESTIONS.inc()
        started = time.perf_counter()
        history = _clamp_history(history)
        sources, messages = await self._prepare(session, question, history)
        yield {"type": "sources", "sources": [asdict(s) for s in sources]}

        if not sources:
            ANSWERS.labels(outcome="idk").inc()
            GROUNDED.labels(result="ungrounded").inc()
            result = AnswerResult(IDK_TEXT, [], [], False, self.llm.name, "idk")
            await self._log_query(session, question, result, started)
            yield {"type": "token", "text": IDK_TEXT}
            yield {
                "type": "done",
                "grounded": False,
                "citations": [],
                "provider": self.llm.name,
                "outcome": "idk",
                "query_log_id": result.query_log_id,
            }
            return

        buffer: list[str] = []
        start = time.perf_counter()
        try:
            async for token in self.llm.stream(
                messages,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_output_tokens,
            ):
                buffer.append(token)
                yield {"type": "token", "text": token}
        except LlmError as exc:
            ANSWERS.labels(outcome="error").inc()
            yield {"type": "error", "message": str(exc)}
            return
        LLM_LATENCY.labels(provider=self.llm.name).observe(time.perf_counter() - start)

        result = self._finalize("".join(buffer), sources, self.llm.name)
        await self._log_query(session, question, result, started)
        yield {
            "type": "done",
            "grounded": result.grounded,
            "citations": result.citations,
            "provider": result.provider,
            "outcome": result.outcome,
            "query_log_id": result.query_log_id,
        }
