"""Plain-text parser: split into paragraphs on blank lines."""

from __future__ import annotations

import re

from ...models import Block, ParsedDocument
from ...text_utils import normalize_text

_PARA_SPLIT = re.compile(r"\n\s*\n")


def parse_text(data: bytes, *, uri: str) -> ParsedDocument:
    text = normalize_text(data.decode("utf-8", "replace"))
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    blocks = [Block(text=p, kind="text") for p in paragraphs]
    title = paragraphs[0][:200] if paragraphs else None
    return ParsedDocument(source_type="text", title=title, blocks=blocks, byte_size=len(data))
