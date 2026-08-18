"""Answer judges used to estimate groundedness, relevance, and hallucination.

Two implementations sit behind the same small interface:

* :class:`LlmJudge` asks an LLM to grade the answer against the retrieved
  context and (optionally) a reference answer, returning a strict JSON verdict.
  Any provider/parse failure transparently degrades to the lexical judge.
* :class:`LexicalJudge` is a free, deterministic fallback based on token
  overlap. It needs no network access, so unit tests and offline runs still get
  meaningful (if coarser) numbers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from ..llm.base import ChatMessage, LlmClient
from ..logging_setup import get_logger
from .metrics import lexical_overlap

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

JUDGE_SYSTEM = (
    "You are a strict evaluation grader for a retrieval-augmented QA system. "
    "Given a question, the model's answer, and the retrieved context, decide:\n"
    "- grounded: true only if every factual claim in the answer is supported by "
    "the context (no invented facts).\n"
    "- relevant: true if the answer actually addresses the question.\n"
    "- faithfulness: a 0.0-1.0 estimate of how well the answer is supported.\n"
    'Respond with ONLY a JSON object: {"grounded": bool, "relevant": bool, '
    '"faithfulness": float, "rationale": string}. No prose, no code fences.'
)


@dataclass(slots=True)
class JudgeVerdict:
    grounded: bool
    relevant: bool
    faithfulness: float
    rationale: str
    judge: str  # "llm" | "lexical"


class Judge(Protocol):
    name: str

    async def judge(
        self, *, question: str, answer: str, context: str, reference: str | None
    ) -> JudgeVerdict: ...


class LexicalJudge:
    """Deterministic, offline judge based on token overlap thresholds."""

    name = "lexical"

    def __init__(self, grounded_threshold: float = 0.18, relevant_threshold: float = 0.12) -> None:
        self.grounded_threshold = grounded_threshold
        self.relevant_threshold = relevant_threshold

    async def judge(
        self, *, question: str, answer: str, context: str, reference: str | None
    ) -> JudgeVerdict:
        support = lexical_overlap(answer, context)
        target = reference or question
        relevance = lexical_overlap(answer, target)
        grounded = support >= self.grounded_threshold
        relevant = relevance >= self.relevant_threshold
        return JudgeVerdict(
            grounded=grounded,
            relevant=relevant,
            faithfulness=round(support, 3),
            rationale=f"lexical support={support:.2f}, relevance={relevance:.2f}",
            judge=self.name,
        )


def _parse_verdict(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class LlmJudge:
    """LLM-as-judge that falls back to :class:`LexicalJudge` on any failure."""

    name = "llm"

    def __init__(
        self,
        llm: LlmClient,
        *,
        fallback: LexicalJudge | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or LexicalJudge()
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def judge(
        self, *, question: str, answer: str, context: str, reference: str | None
    ) -> JudgeVerdict:
        ref_block = f"\nReference answer:\n{reference}\n" if reference else ""
        user = (
            f"Question:\n{question}\n\n"
            f"Model answer:\n{answer}\n\n"
            f"Retrieved context:\n{context}\n{ref_block}\n"
            "Grade the answer now."
        )
        messages = [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", user)]
        try:
            raw = await self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 - any provider error degrades gracefully
            log.warning("llm_judge_failed", error=str(exc))
            return await self.fallback.judge(
                question=question, answer=answer, context=context, reference=reference
            )

        data = _parse_verdict(raw)
        if data is None:
            log.warning("llm_judge_unparseable")
            return await self.fallback.judge(
                question=question, answer=answer, context=context, reference=reference
            )
        return JudgeVerdict(
            grounded=bool(data.get("grounded", False)),
            relevant=bool(data.get("relevant", False)),
            faithfulness=float(data.get("faithfulness", 0.0) or 0.0),
            rationale=str(data.get("rationale", "")),
            judge=self.name,
        )
