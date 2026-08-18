from __future__ import annotations

from docsqa.text_utils import normalize_text, sha256_text


def test_sha256_is_stable_and_distinct() -> None:
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("world")


def test_normalize_collapses_inline_whitespace() -> None:
    assert normalize_text("a   b\t\tc") == "a b c"


def test_normalize_preserves_paragraph_breaks() -> None:
    assert normalize_text("Para1\n\n\n\nPara2") == "Para1\n\nPara2"
