"""Plain dataclasses used to pass data between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Block:
    """A structural unit from a parsed document (a heading or a text paragraph)."""

    text: str
    kind: str = "text"  # "text" | "heading"
    level: int = 0  # heading level 1..6 (0 for text)
    page_no: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    source_type: str
    title: str | None
    blocks: list[Block]
    byte_size: int = 0


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text with provenance for citations."""

    ordinal: int
    text: str
    heading_path: str | None
    page_no: int | None
    char_start: int
    char_end: int
    token_count: int
    chunk_hash: str


@dataclass(slots=True)
class IngestResult:
    uri: str
    action: str  # indexed | updated | skipped | failed
    chunks: int = 0
    document_id: int | None = None


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned from retrieval, with scores attached for ranking/citations."""

    chunk_id: int
    document_id: int
    text: str
    heading_path: str | None
    page_no: int | None
    uri: str
    doc_title: str | None
    score: float = 0.0
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class AnswerSource:
    """A numbered citation source shown alongside an answer."""

    index: int
    chunk_id: int
    document_id: int
    uri: str
    title: str | None
    heading_path: str | None
    page_no: int | None
    snippet: str


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[AnswerSource]
    citations: list[int]
    grounded: bool
    provider: str
    outcome: str  # answered | idk | ungrounded
    query_log_id: int | None = None
