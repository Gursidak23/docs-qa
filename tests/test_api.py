from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from docsqa.api.app import create_app
from docsqa.api.routes import get_service
from docsqa.api.service import SqlQaService
from docsqa.api.schemas import (
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
from docsqa.models import IngestResult


class FakeService:
    async def search(self, req: SearchRequest) -> SearchResponse:
        return SearchResponse(
            query=req.query,
            results=[
                SearchHit(
                    chunk_id=1,
                    document_id=1,
                    uri="doc.md",
                    title="Doc",
                    heading_path="Intro",
                    score=0.9,
                    snippet="hello world",
                )
            ],
        )

    async def ask(self, req: AskRequest) -> AskResponse:
        return AskResponse(
            question=req.question,
            answer="The answer is here [1].",
            grounded=True,
            outcome="answered",
            provider="fake",
            citations=[1],
            query_log_id=7,
            sources=[AskSource(index=1, chunk_id=1, document_id=1, uri="doc.md", snippet="...")],
        )

    async def ask_stream(
        self, question: str, top_k: int | None, history: list[Turn] | None = None
    ) -> AsyncIterator[dict]:
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": "Hi"}
        yield {
            "type": "done",
            "grounded": True,
            "citations": [1],
            "provider": "fake",
            "outcome": "answered",
            "query_log_id": 7,
        }

    async def ingest_upload(
        self, filename: str, data: bytes, force: bool = False
    ) -> IngestResultOut:
        return IngestResultOut(uri=filename, action="indexed", chunks=2, document_id=1)

    async def ingest_url(self, url: str, force: bool = False) -> IngestResultOut:
        return IngestResultOut(uri=url, action="indexed", chunks=3, document_id=2)

    async def stats(self) -> StatsOut:
        return StatsOut(documents=2, chunks=5)

    async def list_documents(self, limit: int) -> list[DocumentOut]:
        return [DocumentOut(id=1, uri="doc.md", source_type="md", title="Doc", chunk_count=3)]

    async def get_document(self, doc_id: int) -> DocumentOut | None:
        if doc_id == 1:
            return DocumentOut(id=1, uri="doc.md", source_type="md")
        return None

    async def delete_document(self, doc_id: int) -> bool:
        return doc_id == 1

    async def submit_feedback(self, query_log_id: int, helpful: bool) -> bool:
        return query_log_id == 7

    async def recent_queries(self, limit: int) -> list[QueryLogOut]:
        return [
            QueryLogOut(
                id=7,
                question="how?",
                provider="fake",
                grounded=True,
                latency_ms=42,
                feedback=1,
            )
        ]


def _client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_service] = lambda: FakeService()
    return TestClient(app)


def test_search_endpoint_returns_hits() -> None:
    resp = _client().post("/api/v1/search", json={"query": "hello", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "hello"
    assert body["results"][0]["chunk_id"] == 1


def test_search_validation_rejects_empty_query() -> None:
    assert _client().post("/api/v1/search", json={"query": ""}).status_code == 422


def test_ask_endpoint_returns_grounded_answer() -> None:
    resp = _client().post("/api/v1/ask", json={"question": "how?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["citations"] == [1]
    assert body["sources"][0]["uri"] == "doc.md"
    assert body["query_log_id"] == 7


def test_ask_endpoint_accepts_conversation_history() -> None:
    resp = _client().post(
        "/api/v1/ask",
        json={
            "question": "what about Groq?",
            "history": [
                {"role": "user", "content": "How do I set a provider?"},
                {"role": "assistant", "content": "Use DOCSQA_LLM__PROVIDER [1]."},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["query_log_id"] == 7


def test_ask_rejects_invalid_history_role() -> None:
    resp = _client().post(
        "/api/v1/ask",
        json={"question": "hi", "history": [{"role": "system", "content": "x"}]},
    )
    assert resp.status_code == 422


def test_ask_validation_rejects_empty_question() -> None:
    assert _client().post("/api/v1/ask", json={"question": ""}).status_code == 422


def test_ask_stream_emits_sse_events() -> None:
    resp = _client().post("/api/v1/ask/stream", json={"question": "how?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"type": "done"' in resp.text


def test_stats_endpoint() -> None:
    body = _client().get("/api/v1/stats").json()
    assert body == {"documents": 2, "chunks": 5}


def test_documents_list_get_delete() -> None:
    client = _client()
    assert client.get("/api/v1/documents").json()[0]["uri"] == "doc.md"
    assert client.get("/api/v1/documents/1").status_code == 200
    assert client.get("/api/v1/documents/999").status_code == 404
    assert client.delete("/api/v1/documents/1").status_code == 204
    assert client.delete("/api/v1/documents/2").status_code == 404


def test_upload_document() -> None:
    resp = _client().post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["action"] == "indexed"


def test_ingest_url() -> None:
    resp = _client().post("/api/v1/ingest/url", json={"url": "https://example.com"})
    assert resp.status_code == 200
    assert resp.json()["document_id"] == 2


async def test_ingest_url_uses_content_type_for_source_detection(monkeypatch) -> None:
    from docsqa.ingest import url_loader
    from docsqa.storage import db as db_mod

    captured: dict[str, str] = {}

    class FakeIndexer:
        async def ingest(
            self,
            session,
            *,
            uri: str,
            source_type: str,
            data: bytes,
            force: bool = False,
        ) -> IngestResult:
            captured["source_type"] = source_type
            return IngestResult(uri=uri, action="indexed", chunks=4, document_id=99)

    @asynccontextmanager
    async def fake_session_scope(settings):
        yield object()

    async def fake_fetch_url(url: str, *, timeout_seconds: float = 30.0) -> tuple[bytes, str | None]:
        return b"%PDF-1.4 fake", "application/pdf"

    monkeypatch.setattr(db_mod, "session_scope", fake_session_scope)
    monkeypatch.setattr(url_loader, "fetch_url", fake_fetch_url)

    service = SqlQaService()
    service._indexer = FakeIndexer()

    result = await service.ingest_url("https://example.com/doc.pdf")

    assert captured["source_type"] == "pdf"
    assert result.chunks == 4


def test_feedback_endpoint() -> None:
    client = _client()
    ok = client.post("/api/v1/feedback", json={"query_log_id": 7, "helpful": True})
    assert ok.status_code == 204
    missing = client.post("/api/v1/feedback", json={"query_log_id": 999, "helpful": False})
    assert missing.status_code == 404


def test_recent_queries_endpoint() -> None:
    body = _client().get("/api/v1/queries?limit=5").json()
    assert body[0]["question"] == "how?"
    assert body[0]["grounded"] is True
    assert body[0]["feedback"] == 1


def test_chat_ui_served_at_root() -> None:
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "grounded answers, cited" in resp.text
