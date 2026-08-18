"""initial schema: source_document, chunk (pgvector + tsvector), query_log

Revision ID: 0001_init
Revises:
Create Date: 2026-06-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_init"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "source_document",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("byte_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="indexed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("uri", name="uq_source_document_uri"),
    )
    op.create_index("ix_source_document_source_type", "source_document", ["source_type"])
    op.create_index("ix_source_document_content_hash", "source_document", ["content_hash"])
    op.create_index("ix_source_document_status", "source_document", ["status"])

    op.create_table(
        "chunk",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("source_document.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading_path", sa.Text(), nullable=True),
        sa.Column("page_no", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', coalesce(\"text\", ''))", persisted=True),
            nullable=True,
        ),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunk_doc_ordinal"),
    )
    op.create_index("ix_chunk_document_id", "chunk", ["document_id"])
    op.create_index("ix_chunk_chunk_hash", "chunk", ["chunk_hash"])
    op.create_index("ix_chunk_tsv", "chunk", ["tsv"], postgresql_using="gin")
    # Approximate-nearest-neighbour index for cosine similarity over embeddings.
    op.execute(
        "CREATE INDEX ix_chunk_embedding_hnsw ON chunk "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "query_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("provider", sa.String(length=40), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column("grounded", sa.Boolean(), nullable=True),
        sa.Column("feedback", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("query_log")
    op.drop_index("ix_chunk_embedding_hnsw", table_name="chunk")
    op.drop_index("ix_chunk_tsv", table_name="chunk")
    op.drop_index("ix_chunk_chunk_hash", table_name="chunk")
    op.drop_index("ix_chunk_document_id", table_name="chunk")
    op.drop_table("chunk")
    op.drop_index("ix_source_document_status", table_name="source_document")
    op.drop_index("ix_source_document_content_hash", table_name="source_document")
    op.drop_index("ix_source_document_source_type", table_name="source_document")
    op.drop_table("source_document")
