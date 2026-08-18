"""PDF parser built on PyMuPDF: per-page paragraphs with page numbers for citations."""

from __future__ import annotations

import re

import pymupdf

from ...models import Block, ParsedDocument
from ...text_utils import normalize_text

_PARA_SPLIT = re.compile(r"\n\s*\n")


def parse_pdf(data: bytes, *, uri: str) -> ParsedDocument:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        meta_title = (doc.metadata or {}).get("title") or None
        blocks: list[Block] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = normalize_text(page.get_text("text"))
            for para in _PARA_SPLIT.split(text):
                para = para.strip()
                if para:
                    blocks.append(Block(text=para, kind="text", page_no=page_index + 1))
    finally:
        doc.close()

    title = meta_title or (blocks[0].text[:200] if blocks else None)
    return ParsedDocument(source_type="pdf", title=title, blocks=blocks, byte_size=len(data))
