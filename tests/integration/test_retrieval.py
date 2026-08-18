"""Integration tests for hybrid retrieval against a real pgvector Postgres."""

from __future__ import annotations

import pytest

from docsqa.ingest.indexer import Indexer
from docsqa.retrieve.hybrid import HybridRetriever
from docsqa.storage.db import session_scope

from ..support import HashEmbedder
from .support import docsqa_settings

pytestmark = pytest.mark.integration


async def test_hybrid_retrieval_surfaces_keyword_match() -> None:
    async with docsqa_settings() as settings:
        embedder = HashEmbedder()
        indexer = Indexer(embedder=embedder, settings=settings)
        corpus = {
            "auth.md": b"# Authentication\n\nTo reset your password open settings and click reset "
            b"password. " + b"Additional authentication guidance. " * 40,
            "billing.md": b"# Billing\n\nInvoices are issued monthly to your account. "
            + b"General billing information. " * 40,
        }
        async with session_scope(settings) as session:
            for uri, data in corpus.items():
                await indexer.ingest(session, uri=uri, source_type="md", data=data)

        retriever = HybridRetriever(embedder, settings)
        async with session_scope(settings) as session:
            hits = await retriever.search(session, "reset password", top_k=5)

        assert hits
        # The lexical arm matches "reset password" only in auth.md, so RRF ranks it first.
        assert hits[0].uri == "auth.md"
        assert hits[0].score > 0
