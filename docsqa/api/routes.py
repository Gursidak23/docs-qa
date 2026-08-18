"""Control-plane HTTP routes (mounted under ``/api/v1``).

Handlers stay thin by delegating to a :class:`QaService`; the dependency is
overridable for tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from starlette.responses import StreamingResponse

from .schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    FeedbackRequest,
    IngestResultOut,
    IngestUrlRequest,
    QueryLogOut,
    SearchRequest,
    SearchResponse,
    StatsOut,
)
from .service import QaService, SqlQaService

router = APIRouter(prefix="/api/v1", tags=["qa"])


def get_service() -> QaService:
    """Provide the QA service. Overridden in tests via ``dependency_overrides``."""
    return SqlQaService()


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest, service: QaService = Depends(get_service)
) -> SearchResponse:
    return await service.search(req)


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, service: QaService = Depends(get_service)) -> AskResponse:
    return await service.ask(req)


@router.post("/ask/stream")
async def ask_stream(
    req: AskRequest, service: QaService = Depends(get_service)
) -> StreamingResponse:
    async def event_source() -> AsyncIterator[str]:
        async for event in service.ask_stream(req.question, req.top_k, req.history):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/documents", response_model=IngestResultOut)
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Form(False),
    service: QaService = Depends(get_service),
) -> IngestResultOut:
    data = await file.read()
    return await service.ingest_upload(file.filename or "upload", data, force)


@router.post("/ingest/url", response_model=IngestResultOut)
async def ingest_url(
    req: IngestUrlRequest, service: QaService = Depends(get_service)
) -> IngestResultOut:
    return await service.ingest_url(req.url, req.force)


@router.get("/stats", response_model=StatsOut)
async def get_stats(service: QaService = Depends(get_service)) -> StatsOut:
    return await service.stats()


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    service: QaService = Depends(get_service),
) -> list[DocumentOut]:
    return await service.list_documents(limit)


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: int, service: QaService = Depends(get_service)
) -> DocumentOut:
    doc = await service.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return doc


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(
    doc_id: int, service: QaService = Depends(get_service)
) -> None:
    ok = await service.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="document not found")


@router.post("/feedback", status_code=204)
async def submit_feedback(
    req: FeedbackRequest, service: QaService = Depends(get_service)
) -> None:
    ok = await service.submit_feedback(req.query_log_id, req.helpful)
    if not ok:
        raise HTTPException(status_code=404, detail="query not found")


@router.get("/queries", response_model=list[QueryLogOut])
async def recent_queries(
    limit: int = Query(20, ge=1, le=200),
    service: QaService = Depends(get_service),
) -> list[QueryLogOut]:
    return await service.recent_queries(limit)
