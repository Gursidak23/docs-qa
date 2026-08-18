from __future__ import annotations

from docsqa.config import ChunkSettings
from docsqa.ingest.chunker import chunk_document
from docsqa.models import Block, ParsedDocument


def _doc(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(source_type="text", title=None, blocks=blocks, byte_size=0)


def test_chunks_do_not_cross_heading_boundaries() -> None:
    blocks = [
        Block("Intro", "heading", 1),
        Block("alpha beta gamma delta", "text"),
        Block("Details", "heading", 2),
        Block("epsilon zeta eta theta", "text"),
    ]
    settings = ChunkSettings(max_tokens=50, overlap_tokens=10, min_tokens=1)
    _full, chunks = chunk_document(_doc(blocks), settings)

    heading_paths = {c.heading_path for c in chunks}
    assert "Intro" in heading_paths
    assert "Intro > Details" in heading_paths


def test_large_section_splits_into_overlapping_windows() -> None:
    big = " ".join(f"word{i}" for i in range(400))
    settings = ChunkSettings(max_tokens=100, overlap_tokens=20, min_tokens=10)
    full_text, chunks = chunk_document(_doc([Block(big, "text")]), settings)

    assert len(chunks) >= 2
    assert all(c.token_count <= 100 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(0 <= c.char_start <= c.char_end <= len(full_text) for c in chunks)


def test_empty_document_yields_no_chunks() -> None:
    _full, chunks = chunk_document(_doc([]), ChunkSettings())
    assert chunks == []
