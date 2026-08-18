"""Source-type detection and parser dispatch."""

from __future__ import annotations

import os

from ...models import ParsedDocument
from .html_parser import parse_html
from .markdown_parser import parse_markdown
from .pdf_parser import parse_pdf
from .text_parser import parse_text

_EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".mdown": "md",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".text": "text",
    ".rst": "text",
}

SUPPORTED_EXTENSIONS = frozenset(_EXT_TO_TYPE)


def detect_source_type(name: str, content_type: str | None = None) -> str:
    """Best-effort source type from an explicit content-type, then file extension."""
    if content_type:
        ct = content_type.lower()
        if "pdf" in ct:
            return "pdf"
        if "html" in ct or "xml" in ct:
            return "html"
        if "markdown" in ct:
            return "md"
        if "text/plain" in ct:
            return "text"
    ext = os.path.splitext(name)[1].lower()
    return _EXT_TO_TYPE.get(ext, "text")


def parse_document(data: bytes, source_type: str, *, uri: str) -> ParsedDocument:
    """Parse raw bytes into structured blocks. ``url`` is treated as HTML."""
    if source_type == "pdf":
        return parse_pdf(data, uri=uri)
    if source_type in ("html", "url"):
        return parse_html(data, uri=uri)
    if source_type == "md":
        return parse_markdown(data, uri=uri)
    return parse_text(data, uri=uri)
