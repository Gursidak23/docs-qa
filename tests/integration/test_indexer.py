"""Integration tests for ingestion against a real pgvector Postgres."""

from __future__ import annotations

import pytest

from docsqa.ingest.indexer import Indexer
from docsqa.storage.db import session_scope
from docsqa.storage.repositories import DocumentRepository

from ..support import CountingEmbedder, HashEmbedder
from .support import docsqa_settings

pytestmark = pytest.mark.integration


async def test_ingest_skip_and_update_cycle() -> None:
    async with docsqa_settings() as settings:
        indexer = Indexer(embedder=HashEmbedder(), settings=settings)
        data = b"# Guide\n\n" + ("Some content here. " * 200).encode()

        async with session_scope(settings) as session:
            first = await indexer.ingest(session, uri="doc1", source_type="md", data=data)
        assert first.action == "indexed"
        assert first.chunks > 0

        async with session_scope(settings) as session:
            repo = DocumentRepository(session)
            assert await repo.count_documents() == 1
            assert await repo.count_chunks() == first.chunks

        # Re-ingesting identical content is a no-op.
        async with session_scope(settings) as session:
            again = await indexer.ingest(session, uri="doc1", source_type="md", data=data)
        assert again.action == "skipped"
        async with session_scope(settings) as session:
            assert await DocumentRepository(session).count_chunks() == first.chunks

        # Changed content replaces chunks and bumps the version.
        changed = b"# Guide\n\n" + ("Totally different material. " * 220).encode()
        async with session_scope(settings) as session:
            updated = await indexer.ingest(session, uri="doc1", source_type="md", data=changed)
        assert updated.action == "updated"
        async with session_scope(settings) as session:
            repo = DocumentRepository(session)
            assert await repo.count_documents() == 1
            doc = await repo.get_by_uri("doc1")
            assert doc is not None
            assert doc.version == 2
            assert await repo.count_chunks() == updated.chunks


async def test_incremental_reembeds_only_changed_chunks() -> None:
    async with docsqa_settings() as settings:
        embedder = CountingEmbedder()
        indexer = Indexer(embedder=embedder, settings=settings)
        sections = [
            f"## Section {i}\n\nDistinct content about subject {i} with several words here."
            for i in range(8)
        ]
        body = "\n\n".join(sections).encode()

        async with session_scope(settings) as session:
            first = await indexer.ingest(session, uri="kb", source_type="md", data=body)
        assert first.chunks >= 8
        assert embedder.total_embedded == first.chunks  # everything embedded the first time

        embedder.reset()
        changed = body.replace(b"subject 3", b"a completely revised and different topic entirely")
        async with session_scope(settings) as session:
            second = await indexer.ingest(session, uri="kb", source_type="md", data=changed)
        assert second.action == "updated"
        # Only the changed chunk(s) are re-embedded; the rest reuse stored embeddings.
        assert 0 < embedder.total_embedded < first.chunks

        async with session_scope(settings) as session:
            assert await DocumentRepository(session).count_chunks() == second.chunks
