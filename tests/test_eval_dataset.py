from __future__ import annotations

from pathlib import Path

import pytest

import docsqa.eval as eval_pkg
from docsqa.eval.dataset import GoldCase, load_gold, parse_gold


def test_parse_gold_ignores_blanks_and_comments() -> None:
    lines = [
        "# a comment",
        "",
        '{"id": "a", "question": "Q1?", "relevant_uris": ["docs/a.md"]}',
        '   ',
        '{"question": "Q2?", "relevant_chunk_ids": [3, 4]}',
    ]
    cases = parse_gold(lines)
    assert len(cases) == 2
    assert cases[0].id == "a"
    assert cases[1].id == "case-4"  # auto id from line index


def test_doc_level_relevance_and_counts() -> None:
    case = GoldCase(id="x", question="q", relevant_uris=["docs/a.md"], relevant_doc_ids=[7])
    assert not case.chunk_level
    assert case.num_relevant == 2
    assert case.is_relevant(chunk_id=1, document_id=7, uri="other")
    assert case.is_relevant(chunk_id=1, document_id=99, uri="docs/a.md")
    assert not case.is_relevant(chunk_id=1, document_id=99, uri="docs/b.md")


def test_chunk_level_relevance() -> None:
    case = GoldCase(id="x", question="q", relevant_chunk_ids=[10, 11])
    assert case.chunk_level
    assert case.num_relevant == 2
    assert case.is_relevant(chunk_id=10, document_id=1, uri="docs/a.md")
    assert not case.is_relevant(chunk_id=99, document_id=1, uri="docs/a.md")


def test_from_dict_aliases() -> None:
    case = GoldCase.from_dict(
        {"query": "Hello?", "answer": "Hi", "relevant_ids": ["docs/a.md"]}, index=2
    )
    assert case.question == "Hello?"
    assert case.reference_answer == "Hi"
    assert case.relevant_uris == ["docs/a.md"]


def test_from_dict_requires_question() -> None:
    with pytest.raises(ValueError, match="question"):
        GoldCase.from_dict({"id": "x"}, index=0)


def test_load_gold_roundtrip(tmp_path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text('{"id": "a", "question": "Q?", "relevant_uris": ["d.md"]}\n', encoding="utf-8")
    cases = load_gold(path)
    assert len(cases) == 1 and cases[0].question == "Q?"


def test_load_gold_empty_raises(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no gold cases"):
        load_gold(path)


def test_bundled_gold_dataset_is_valid() -> None:
    path = Path(eval_pkg.__file__).parent / "gold" / "support.jsonl"
    cases = load_gold(path)
    ids = {c.id for c in cases}
    assert {"pw-reset", "api-key", "rate-limit", "billing-plan", "out-of-scope"} <= ids
    # Every annotated case points at a corpus file that actually ships in the repo.
    corpus = path.parent / "corpus"
    for case in cases:
        for uri in case.relevant_uris:
            assert (Path.cwd() / uri).exists() or (corpus / Path(uri).name).exists()
