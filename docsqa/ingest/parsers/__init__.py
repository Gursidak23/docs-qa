"""Document parsers that turn raw bytes into structured blocks."""

from .registry import detect_source_type, parse_document

__all__ = ["detect_source_type", "parse_document"]
