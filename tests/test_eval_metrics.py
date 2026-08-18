from __future__ import annotations

import math

from docsqa.eval.metrics import (
    citation_accuracy,
    lexical_overlap,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)


def test_recall_and_precision() -> None:
    rels = [True, False, True]
    assert recall_at_k(rels, num_relevant=2, k=2) == 0.5
    assert recall_at_k(rels, num_relevant=2, k=3) == 1.0
    assert precision_at_k(rels, 2) == 0.5
    assert recall_at_k(rels, num_relevant=0, k=3) == 0.0


def test_mrr_uses_first_relevant() -> None:
    assert mrr([False, True, True]) == 0.5
    assert mrr([True]) == 1.0
    assert mrr([False, False]) == 0.0


def test_ndcg_matches_manual_computation() -> None:
    rels = [False, True, True]
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(rels, num_relevant=2, k=3) == round(expected, 10) or math.isclose(
        ndcg_at_k(rels, num_relevant=2, k=3), expected
    )


def test_ndcg_perfect_ranking_is_one() -> None:
    assert math.isclose(ndcg_at_k([True, True, False], num_relevant=2, k=3), 1.0)


def test_citation_accuracy() -> None:
    assert citation_accuracy([1, 2, 5], num_sources=3) == 2 / 3
    assert citation_accuracy([], num_sources=3) == 0.0


def test_token_f1_and_overlap() -> None:
    assert math.isclose(token_f1("a b c", "a b"), 0.8)
    assert token_f1("", "a b") == 0.0
    assert lexical_overlap("a b c", "b c d") == 0.5
    assert lexical_overlap("x", "") == 0.0
