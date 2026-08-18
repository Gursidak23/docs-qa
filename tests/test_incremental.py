from __future__ import annotations

from docsqa.ingest.indexer import select_new_hash_texts
from docsqa.models import Chunk


def _chunk(chunk_hash: str, text: str) -> Chunk:
    return Chunk(
        ordinal=0,
        text=text,
        heading_path=None,
        page_no=None,
        char_start=0,
        char_end=0,
        token_count=0,
        chunk_hash=chunk_hash,
    )


def test_select_skips_known_hashes_and_dedups() -> None:
    chunks = [_chunk("a", "ta"), _chunk("b", "tb"), _chunk("a", "ta"), _chunk("c", "tc")]
    hashes, texts = select_new_hash_texts(chunks, {"b"})
    assert hashes == ["a", "c"]
    assert texts == ["ta", "tc"]


def test_select_returns_all_when_nothing_known() -> None:
    chunks = [_chunk("x", "tx"), _chunk("y", "ty")]
    hashes, texts = select_new_hash_texts(chunks, set())
    assert hashes == ["x", "y"]
    assert texts == ["tx", "ty"]


def test_select_handles_empty() -> None:
    assert select_new_hash_texts([], {"a"}) == ([], [])
