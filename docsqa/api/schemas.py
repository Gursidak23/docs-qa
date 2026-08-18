"""Pydantic request/response models for the control-plane API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    uri: str
    title: str | None = None
    heading_path: str | None = None
    page_no: int | None = None
    score: float
    snippet: str
    vector_rank: int | None = None
    lexical_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class StatsOut(BaseModel):
    documents: int
    chunks: int


class DocumentOut(BaseModel):
    id: int
    uri: str
    source_type: str
    title: str | None = None
    chunk_count: int = 0
    version: int = 1
    byte_size: int = 0
    status: str = "indexed"


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=1)
    force: bool = False


class IngestResultOut(BaseModel):
    uri: str
    action: str
    chunks: int = 0
    document_id: int | None = None


class Turn(BaseModel):
    """A prior conversation turn supplied by the client for follow-up context."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    # Prior turns (oldest first) so the assistant can resolve follow-ups.
    history: list[Turn] = Field(default_factory=list)


class AskSource(BaseModel):
    index: int
    chunk_id: int
    document_id: int
    uri: str
    title: str | None = None
    heading_path: str | None = None
    page_no: int | None = None
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    outcome: str
    provider: str
    citations: list[int] = Field(default_factory=list)
    sources: list[AskSource] = Field(default_factory=list)
    query_log_id: int | None = None


class FeedbackRequest(BaseModel):
    query_log_id: int
    helpful: bool


class QueryLogOut(BaseModel):
    id: int
    question: str
    provider: str | None = None
    grounded: bool | None = None
    latency_ms: int | None = None
    feedback: int | None = None
    created_at: datetime | None = None
