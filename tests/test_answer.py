from __future__ import annotations

from collections.abc import AsyncIterator

from docsqa.cache import MemoryCache
from docsqa.config import Settings
from docsqa.llm.base import ChatMessage
from docsqa.models import RetrievedChunk
from docsqa.rag.answer import AnswerService
from docsqa.retrieve.rerank import NoopReranker


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    async def search(
        self, session: object, query: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        return self._chunks if top_k is None else self._chunks[:top_k]


class FakeLlm:
    name = "fake"

    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, messages: list, *, temperature: float, max_tokens: int) -> str:
        return self._text

    async def stream(
        self, messages: list, *, temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        for token in self._text.split(" "):
            yield token + " "


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        text=f"content {chunk_id}",
        heading_path=f"H{chunk_id}",
        page_no=None,
        uri=f"doc{chunk_id}.md",
        doc_title="Doc",
    )


def _service(chunks: list[RetrievedChunk], text: str) -> AnswerService:
    settings = Settings()
    settings.rerank.enabled = False
    return AnswerService(FakeRetriever(chunks), NoopReranker(), FakeLlm(text), settings)


async def test_answer_is_grounded_with_valid_citation() -> None:
    service = _service([_chunk(1), _chunk(2)], "The answer is here [1].")
    result = await service.answer(None, "q")
    assert result.grounded is True
    assert result.outcome == "answered"
    assert result.citations == [1]
    assert [s.index for s in result.sources] == [1]


async def test_answer_is_ungrounded_without_citation() -> None:
    service = _service([_chunk(1)], "Some confident but uncited claim.")
    result = await service.answer(None, "q")
    assert result.grounded is False
    assert result.outcome == "ungrounded"


async def test_answer_says_idk_when_no_candidates() -> None:
    service = _service([], "irrelevant")
    result = await service.answer(None, "q")
    assert result.outcome == "idk"
    assert result.grounded is False


class CountingLlm(FakeLlm):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.calls = 0

    async def complete(self, messages: list, *, temperature: float, max_tokens: int) -> str:
        self.calls += 1
        return await super().complete(messages, temperature=temperature, max_tokens=max_tokens)


async def test_answer_is_cached_on_repeat() -> None:
    settings = Settings()
    settings.rerank.enabled = False
    llm = CountingLlm("The answer is here [1].")
    service = AnswerService(
        FakeRetriever([_chunk(1)]), NoopReranker(), llm, settings, cache=MemoryCache()
    )
    first = await service.answer(None, "q")
    second = await service.answer(None, "q")
    assert first.answer == second.answer
    assert second.grounded is True
    assert llm.calls == 1  # second response served from the answer cache


async def test_answer_stream_emits_sources_tokens_and_done() -> None:
    service = _service([_chunk(1)], "Answer [1].")
    events = [event async for event in service.answer_stream(None, "q")]
    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert "token" in types
    assert types[-1] == "done"
    assert events[-1]["grounded"] is True
    assert events[-1]["citations"] == [1]
    assert "query_log_id" in events[-1]


def _cached_service(llm: CountingLlm, signature: dict[str, str]) -> AnswerService:
    settings = Settings()
    settings.rerank.enabled = False
    service = AnswerService(
        FakeRetriever([_chunk(1)]), NoopReranker(), llm, settings, cache=MemoryCache()
    )

    async def _fake_signature(_session: object) -> str:
        return signature["value"]

    service._corpus_signature = _fake_signature  # type: ignore[method-assign]
    return service


async def test_answer_cache_invalidated_when_corpus_signature_changes() -> None:
    llm = CountingLlm("The answer is here [1].")
    signature = {"value": "sig-1"}
    service = _cached_service(llm, signature)

    await service.answer(None, "q")
    await service.answer(None, "q")
    assert llm.calls == 1  # served from cache while the corpus is unchanged

    signature["value"] = "sig-2"  # simulate a re-ingest
    await service.answer(None, "q")
    assert llm.calls == 2  # stale cache entry is bypassed after re-ingest


async def test_answer_cache_separated_by_conversation_history() -> None:
    llm = CountingLlm("The answer is here [1].")
    service = _cached_service(llm, {"value": "sig-1"})

    hist_a = [ChatMessage("user", "prior A")]
    hist_b = [ChatMessage("user", "prior B")]

    await service.answer(None, "q", hist_a)
    await service.answer(None, "q", hist_b)
    assert llm.calls == 2  # different contexts must not share a cached answer

    await service.answer(None, "q", hist_a)
    assert llm.calls == 2  # identical context reuses the cache
