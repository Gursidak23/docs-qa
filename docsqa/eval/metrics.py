"""Pure metric functions for retrieval and answer quality.

Everything here is deterministic and dependency-free so it can be unit-tested in
isolation. Retrieval functions take a ranked list of booleans (``True`` where the
item at that rank is relevant) plus the total number of relevant items.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def recall_at_k(relevances: Sequence[bool], num_relevant: int, k: int) -> float:
    """Fraction of all relevant items that appear in the top-``k`` ranks."""
    if num_relevant <= 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r)
    return hits / num_relevant


def precision_at_k(relevances: Sequence[bool], k: int) -> float:
    """Fraction of the top-``k`` ranks that are relevant."""
    top = relevances[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r) / len(top)


def mrr(relevances: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item (0 if none)."""
    for index, relevant in enumerate(relevances):
        if relevant:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(relevances: Sequence[bool], num_relevant: int, k: int) -> float:
    """Binary nDCG@k using the gold relevant count for the ideal ranking."""
    gains = [1.0 if r else 0.0 for r in relevances[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def citation_accuracy(citations: Sequence[int], num_sources: int) -> float:
    """Fraction of cited indices that point at a real provided source passage."""
    if not citations:
        return 0.0
    valid = sum(1 for c in citations if 1 <= c <= num_sources)
    return valid / len(citations)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between a prediction and a reference answer."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok in pred:
        if ref_counts.get(tok, 0) - common.get(tok, 0) > 0:
            common[tok] = common.get(tok, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def lexical_overlap(text: str, reference: str) -> float:
    """Jaccard overlap of token *sets*; a cheap relevance/grounding heuristic."""
    a = set(_tokens(text))
    b = set(_tokens(reference))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
