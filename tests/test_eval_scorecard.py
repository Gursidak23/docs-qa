from __future__ import annotations

import json

from docsqa.eval.harness import CaseResult, Scorecard
from docsqa.eval.scorecard import to_dict, to_json, to_markdown


def _card(passed: bool) -> Scorecard:
    case = CaseResult(
        id="c1",
        question="q",
        num_relevant=1,
        retrieved=3,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        answered=True,
        outcome="answered",
        grounded_guard=True,
        judged_grounded=True,
        judged_relevant=True,
        faithfulness=0.9,
        citation_accuracy=1.0,
        answer_f1=0.8,
        judge="llm",
        provider="fake",
    )
    return Scorecard(
        dataset="unit",
        k=10,
        num_cases=1,
        num_answered=1,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        groundedness=1.0,
        hallucination_rate=0.0,
        answer_relevance=1.0,
        citation_accuracy=1.0,
        answer_f1=0.8,
        guard_grounded_rate=1.0,
        judge="llm",
        thresholds={
            "min_recall_at_k": 0.7,
            "min_mrr": 0.5,
            "min_groundedness": 0.8,
            "max_hallucination_rate": 0.15,
        },
        passed=passed,
        cases=[case],
    )


def test_to_json_is_valid_and_roundtrips() -> None:
    card = _card(passed=True)
    data = json.loads(to_json(card))
    assert data == to_dict(card)
    assert data["cases"][0]["id"] == "c1"


def test_to_markdown_reports_pass_and_gates() -> None:
    md = to_markdown(_card(passed=True))
    assert "Result: **PASS**" in md
    assert "Recall@10" in md
    assert "Hallucination rate" in md
    assert "| c1 |" in md


def test_to_markdown_reports_fail() -> None:
    assert "Result: **FAIL**" in to_markdown(_card(passed=False))
