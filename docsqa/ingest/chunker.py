"""Structure-aware chunking.

Chunks never cross a heading boundary, so each chunk carries a coherent heading
breadcrumb (great for citations). Within a section, text is split into
token-sized windows with overlap using the same tokenizer family as the LLMs.
Character offsets are recovered from the tokenizer so the UI can highlight the
exact span inside the source document.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import tiktoken

from ..config import ChunkSettings
from ..models import Block, Chunk, ParsedDocument
from ..text_utils import sha256_text


@functools.lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True)
class _Section:
    heading_path: str | None
    page_no: int | None
    text: str
    base_offset: int


def _build_sections(blocks: list[Block]) -> tuple[str, list[_Section]]:
    """Group consecutive text blocks under their current heading breadcrumb."""
    sections: list[_Section] = []
    full_parts: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    cursor = 0

    buffer: list[str] = []
    cur_heading: str | None = None
    cur_page: int | None = None

    def flush() -> None:
        nonlocal cursor
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        sections.append(_Section(cur_heading, cur_page, text, cursor))
        full_parts.append(text)
        cursor += len(text) + 2  # mirror the "\n\n" join between sections

    for block in blocks:
        if block.kind == "heading":
            flush()
            level = block.level or 1
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text.strip()))
            cur_heading = " > ".join(title for _, title in heading_stack) or None
            cur_page = block.page_no
        else:
            if not buffer and block.page_no is not None:
                cur_page = block.page_no
            buffer.append(block.text.strip())
    flush()

    return "\n\n".join(full_parts), sections


def chunk_document(parsed: ParsedDocument, settings: ChunkSettings) -> tuple[str, list[Chunk]]:
    """Return the normalized full text and its ordered chunks."""
    enc = _encoding(settings.tokenizer_encoding)
    full_text, sections = _build_sections(parsed.blocks)

    max_tokens = max(1, settings.max_tokens)
    step = max(1, max_tokens - settings.overlap_tokens)

    chunks: list[Chunk] = []
    ordinal = 0
    for section in sections:
        tokens = enc.encode(section.text)
        n = len(tokens)
        if n == 0:
            continue
        start = 0
        while start < n:
            window = tokens[start : start + max_tokens]
            # Skip a tiny trailing remnant that only exists because of overlap.
            if start > 0 and len(window) < settings.min_tokens:
                break
            window_text = enc.decode(window)
            chunk_text = window_text.strip()
            if chunk_text:
                in_section_start = len(enc.decode(tokens[:start]))
                char_start = section.base_offset + in_section_start
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        text=chunk_text,
                        heading_path=section.heading_path,
                        page_no=section.page_no,
                        char_start=char_start,
                        char_end=char_start + len(window_text),
                        token_count=len(window),
                        chunk_hash=sha256_text(chunk_text),
                    )
                )
                ordinal += 1
            if start + max_tokens >= n:
                break
            start += step

    return full_text, chunks
