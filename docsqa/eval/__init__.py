"""Evaluation harness: retrieval + answer metrics, LLM-as-judge, scorecards."""

from __future__ import annotations

from .dataset import GoldCase, load_gold
from .harness import CaseResult, EvalHarness, Scorecard
from .judge import JudgeVerdict, LexicalJudge, LlmJudge

__all__ = [
    "CaseResult",
    "EvalHarness",
    "GoldCase",
    "JudgeVerdict",
    "LexicalJudge",
    "LlmJudge",
    "Scorecard",
    "load_gold",
]
