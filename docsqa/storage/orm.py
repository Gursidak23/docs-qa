"""SQLAlchemy 2.0 ORM models.

The schema captures ingested source documents and their derived chunks. Each
chunk carries a dense embedding (pgvector) for semantic search and a generated
``tsvector`` (Postgres full-text) for lexical search, enabling hybrid retrieval
in a single store. ``query_log`` records questions/answers for analytics + eval.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimensionality of BAAI/bge-small-en-v1.5 (the default fastembed model).
EMBED_DIM = 384


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    __tablename__ = "source_document"
    __table_args__ = (UniqueConstraint("uri", name="uq_source_document_uri"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Canonical identifier for the source: a file path, URL, or upload name.
    uri: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(16), index=True)  # pdf|md|html|url|text
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    # SHA-256 of the normalized raw document; drives incremental re-indexing.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="indexed", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_doc_ordinal"),
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunk_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_document.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Breadcrumb of section headings ("Guide > Auth > Tokens") for nicer citations.
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    # SHA-256 of the chunk text; lets re-indexing re-embed only changed chunks.
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    # Generated lexical index column (Postgres full-text search).
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', coalesce(\"text\", ''))", persisted=True),
        nullable=True,
    )

    document: Mapped[SourceDocument] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(String(40), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieved_chunk_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), nullable=True)
    grounded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
