"""Small text helpers shared across ingestion, retrieval, and eval."""

from __future__ import annotations

import hashlib
import re

_INLINE_WS = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def sha256_text(text: str) -> str:
    """Stable content fingerprint used for incremental (re)indexing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize newlines and collapse runs of whitespace without losing paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    # Trim trailing spaces left on individual lines.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()
