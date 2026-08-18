"""Render a :class:`Scorecard` as JSON or a human-readable markdown report."""

from __future__ import annotations

import json
from dataclasses import asdict

from .harness import Scorecard


def to_dict(card: Scorecard) -> dict:
    return asdict(card)


def to_json(card: Scorecard, *, indent: int = 2) -> str:
    return json.dumps(to_dict(card), indent=indent)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _gate(value: float, floor: float, *, higher_is_better: bool = True) -> str:
    ok = value >= floor if higher_is_better else value <= floor
    return "PASS" if ok else "FAIL"


def to_markdown(card: Scorecard) -> str:
    th = card.thresholds
    halluc_gate = _gate(
        card.hallucination_rate, th["max_hallucination_rate"], higher_is_better=False
    )
    lines = [
        f"# Eval scorecard{f' — {card.dataset}' if card.dataset else ''}",
        "",
        f"- Cases: **{card.num_cases}** ({card.num_answered} answered)",
        f"- Judge: **{card.judge}**",
        f"- Result: **{'PASS' if card.passed else 'FAIL'}**",
        "",
        "## Retrieval",
        "",
        "| Metric | Value | Threshold | Gate |",
        "| --- | --- | --- | --- |",
        f"| Recall@{card.k} | {_pct(card.recall_at_k)} | >= {_pct(th['min_recall_at_k'])} "
        f"| {_gate(card.recall_at_k, th['min_recall_at_k'])} |",
        f"| MRR | {card.mrr:.3f} | >= {th['min_mrr']:.3f} "
        f"| {_gate(card.mrr, th['min_mrr'])} |",
        f"| nDCG@{card.k} | {card.ndcg_at_k:.3f} | - | - |",
        "",
        "## Answer quality",
        "",
        "| Metric | Value | Threshold | Gate |",
        "| --- | --- | --- | --- |",
        f"| Groundedness | {_pct(card.groundedness)} | >= {_pct(th['min_groundedness'])} "
        f"| {_gate(card.groundedness, th['min_groundedness'])} |",
        f"| Hallucination rate | {_pct(card.hallucination_rate)} "
        f"| <= {_pct(th['max_hallucination_rate'])} | {halluc_gate} |",
        f"| Answer relevance | {_pct(card.answer_relevance)} | - | - |",
        f"| Citation accuracy | {_pct(card.citation_accuracy)} | - | - |",
        f"| Answer token-F1 | {card.answer_f1:.3f} | - | - |",
        f"| Guard grounded rate | {_pct(card.guard_grounded_rate)} | - | - |",
        "",
        "## Per-case",
        "",
        "| id | outcome | recall@k | mrr | grounded | relevant | cite | f1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in card.cases:
        f1 = f"{c.answer_f1:.2f}" if c.answer_f1 is not None else "-"
        lines.append(
            f"| {c.id} | {c.outcome} | {c.recall_at_k:.2f} | {c.mrr:.2f} "
            f"| {'yes' if c.judged_grounded else 'no'} "
            f"| {'yes' if c.judged_relevant else 'no'} "
            f"| {c.citation_accuracy:.2f} | {f1} |"
        )
    return "\n".join(lines) + "\n"
