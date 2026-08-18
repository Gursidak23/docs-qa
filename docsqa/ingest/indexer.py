"""Idempotent, incremental ingestion: parse -> chunk -> embed -> upsert.

Change detection happens at two levels:

* document level via ``content_hash`` (re-ingesting identical content is a no-op);
* chunk level via ``chunk_hash`` so that, when a document changes, only the
  chunks whose text actually changed are re-embedded. Unchanged chunks keep their
  existing embedding, which is the expensive part we want to avoid recomputing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..embed.base import Embedder
from ..logging_setup import get_logger
from ..metrics import CHUNKS_INDEXED, DOCS_INGESTED
from ..models import Chunk, IngestResult
from ..storage.orm import SourceDocument
from ..storage.repositories import DocumentRepository
from ..text_utils import sha256_text
from .chunker import chunk_document
from .parsers.registry import parse_document

log = get_logger(__name__)


def select_new_hash_texts(
    chunks: list[Chunk], known_hashes: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return (hashes, texts) for distinct chunk hashes not already embedded.

    Deduplicates within the document so identical chunks are embedded once.
    """
    known = set(known_hashes)
    seen: set[str] = set()
    hashes: list[str] = []
    texts: list[str] = []
    for chunk in chunks:
        if chunk.chunk_hash in known or chunk.chunk_hash in seen:
            continue
        seen.add(chunk.chunk_hash)
        hashes.append(chunk.chunk_hash)
        texts.append(chunk.text)
    return hashes, texts


@dataclass
class Indexer:
    embedder: Embedder
    settings: Settings

    async def ingest(
        self,
        session: AsyncSession,
        *,
        uri: str,
        source_type: str,
        data: bytes,
        force: bool = False,
    ) -> IngestResult:
        parsed = parse_document(data, source_type, uri=uri)
        full_text, chunks = chunk_document(parsed, self.settings.chunk)
        content_hash = sha256_text(full_text)

        repo = DocumentRepository(session)
        existing = await repo.get_by_uri(uri)
        if existing is not None and existing.content_hash == content_hash and not force:
            DOCS_INGESTED.labels(source_type=source_type, outcome="skipped").inc()
            CHUNKS_INDEXED.labels(action="skip").inc(existing.chunk_count)
            return IngestResult(
                uri=uri, action="skipped", chunks=existing.chunk_count, document_id=existing.id
            )

        now = datetime.now(UTC)
        if existing is not None:
            embeddings, embedded = await self._embeddings_incremental(repo, existing.id, chunks)
            await repo.delete_chunks(existing.id)
            existing.title = parsed.title
            existing.source_type = source_type
            existing.content_hash = content_hash
            existing.version += 1
            existing.byte_size = parsed.byte_size or len(data)
            existing.chunk_count = len(chunks)
            existing.status = "indexed"
            existing.indexed_at = now
            await session.flush()
            document_id = existing.id
            action = "updated"
            CHUNKS_INDEXED.labels(action="insert").inc(embedded)
            CHUNKS_INDEXED.labels(action="reuse").inc(len(chunks) - embedded)
        else:
            embeddings = self.embedder.embed_documents([c.text for c in chunks]) if chunks else []
            doc = SourceDocument(
                uri=uri,
                source_type=source_type,
                title=parsed.title,
                content_hash=content_hash,
                version=1,
                byte_size=parsed.byte_size or len(data),
                chunk_count=len(chunks),
                status="indexed",
                indexed_at=now,
            )
            session.add(doc)
            await session.flush()
            document_id = doc.id
            action = "indexed"
            CHUNKS_INDEXED.labels(action="insert").inc(len(chunks))

        await repo.add_chunks(document_id, chunks, embeddings)
        DOCS_INGESTED.labels(source_type=source_type, outcome=action).inc()
        log.info("ingested", uri=uri, action=action, chunks=len(chunks))
        return IngestResult(uri=uri, action=action, chunks=len(chunks), document_id=document_id)

    async def _embeddings_incremental(
        self, repo: DocumentRepository, document_id: int, chunks: list[Chunk]
    ) -> tuple[list[list[float]], int]:
        """Compute embeddings for ``chunks``, reusing unchanged ones by hash.

        Returns the per-chunk embeddings (aligned with ``chunks``) and the number
        of chunks that required a fresh embedding.
        """
        if not chunks:
            return [], 0
        known = await repo.chunk_embeddings_by_hash(document_id)
        new_hashes, new_texts = select_new_hash_texts(chunks, known.keys())
        fresh_vectors = self.embedder.embed_documents(new_texts) if new_texts else []
        fresh_map = dict(zip(new_hashes, fresh_vectors, strict=True))
        embeddings = [
            known[c.chunk_hash] if c.chunk_hash in known else fresh_map[c.chunk_hash]
            for c in chunks
        ]
        embedded = sum(1 for c in chunks if c.chunk_hash not in known)
        return embeddings, embedded
