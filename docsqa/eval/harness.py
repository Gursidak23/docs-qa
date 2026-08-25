"""Evaluation harness: run gold cases through retrieval + answering and score them.

For each case we measure two things:

* retrieval quality (Recall@k, MRR, nDCG@k) from the hybrid retriever's ranked
  candidates against the gold relevant set; and
* answer quality (groundedness, hallucination rate, answer relevance, citation
  accuracy, token-F1) from the full RAG answer plus a judge verdict.

The collaborators are injected as small Protocols so the harness can be unit
tested with fakes and no database/LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..llm.base import LlmError
from ..logging_setup import get_logger
from ..models import AnswerResult, RetrievedChunk
from .dataset import GoldCase
from .judge import JudgeVerdict
from .metrics import citation_accuracy, mrr, ndcg_at_k, recall_at_k, token_f1

log = get_logger(__name__)


class _Retriever(Protocol):
    async def search(
        self, session: AsyncSession, query: str, *, top_k: int | None = ...
    ) -> list[RetrievedChunk]: ...


class _Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class _Answerer(Protocol):
    async def answer(self, session: AsyncSession, question: str) -> AnswerResult: ...


class _Judge(Protocol):
    async def judge(
        self, *, question: str, answer: str, context: str, reference: str | None
    ) -> JudgeVerdict: ...


@dataclass(slots=True)
class CaseResult:
    id: str
    question: str
    num_relevant: int
    retrieved: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    answered: bool
    outcome: str
    grounded_guard: bool
    judged_grounded: bool
    judged_relevant: bool
    faithfulness: float
    citation_accuracy: float
    answer_f1: float | None
    judge: str
    provider: str
    # Verdict came from the fallback judge after the LLM judge failed.
    judge_degraded: bool = False


@dataclass(slots=True)
class Scorecard:
    dataset: str
    k: int
    num_cases: int
    num_answered: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    # None when no case produced a trustworthy verdict (no LLM keys, or the
    # judge was down); the corresponding gates are skipped rather than passed.
    groundedness: float | None
    hallucination_rate: float | None
    answer_relevance: float | None
    citation_accuracy: float | None
    answer_f1: float
    guard_grounded_rate: float | None
    judge: str
    thresholds: dict[str, float]
    passed: bool
    # Answered cases whose verdict came from a degraded (fallback) judge.
    num_judge_degraded: int = 0
    judged_fraction: float = 1.0
    cases: list[CaseResult] = field(default_factory=list)


def relevance_vector(case: GoldCase, chunks: Sequence[RetrievedChunk]) -> list[bool]:
    """Boolean relevance per rank; deduped to documents unless chunk-level gold."""
    if case.chunk_level:
        return [
            case.is_relevant(chunk_id=c.chunk_id, document_id=c.document_id, uri=c.uri)
            for c in chunks
        ]
    seen: set[int] = set()
    out: list[bool] = []
    for c in chunks:
        if c.document_id in seen:
            continue
        seen.add(c.document_id)
        out.append(case.is_relevant(chunk_id=c.chunk_id, document_id=c.document_id, uri=c.uri))
    return out


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


class EvalHarness:
    def __init__(
        self,
        retriever: _Retriever,
        reranker: _Reranker,
        answerer: _Answerer,
        judge: _Judge,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.answerer = answerer
        self.judge = judge
        self.settings = settings

    async def _run_case(self, session: AsyncSession, case: GoldCase) -> CaseResult:
        k = self.settings.eval.retrieval_k
        candidates = await self.retriever.search(
            session, case.question, top_k=self.settings.retrieve.fused_top_k
        )
        relevances = relevance_vector(case, candidates)
        n_rel = case.num_relevant

        try:
            result = await self.answerer.answer(session, case.question)
        except LlmError as exc:
            # Keep going so retrieval metrics still compute (e.g. CI without LLM keys).
            log.warning("eval_answer_failed", case=case.id, error=str(exc))
            result = AnswerResult("", [], [], grounded=False, provider="error", outcome="error")

        answered = result.outcome not in ("idk", "error")
        if answered:
            top_n = (
                self.settings.rerank.top_n
                if self.settings.rerank.enabled
                else self.settings.retrieve.fused_top_k
            )
            ranked = self.reranker.rerank(case.question, candidates, top_n) if candidates else []
            context = "\n\n".join(c.text for c in ranked)
            verdict = await self.judge.judge(
                question=case.question,
                answer=result.answer,
                context=context,
                reference=case.reference_answer,
            )
        else:
            verdict = JudgeVerdict(True, False, 0.0, "not answered", "skipped")

        f1 = token_f1(result.answer, case.reference_answer) if case.reference_answer else None

        return CaseResult(
            id=case.id,
            question=case.question,
            num_relevant=n_rel,
            retrieved=len(candidates),
            recall_at_k=recall_at_k(relevances, n_rel, k),
            mrr=mrr(relevances),
            ndcg_at_k=ndcg_at_k(relevances, n_rel, k),
            answered=answered,
            outcome=result.outcome,
            grounded_guard=result.grounded,
            judged_grounded=verdict.grounded,
            judged_relevant=verdict.relevant,
            faithfulness=verdict.faithfulness,
            citation_accuracy=citation_accuracy(result.citations, len(result.sources)),
            answer_f1=f1,
            judge=verdict.judge,
            provider=result.provider,
            judge_degraded=verdict.degraded,
        )

    async def run(
        self, session: AsyncSession, cases: Sequence[GoldCase], *, dataset: str = ""
    ) -> Scorecard:
        results = [await self._run_case(session, case) for case in cases]

        retrieval = [r for r in results if r.num_relevant > 0]
        answered = [r for r in results if r.answered]
        # A degraded verdict is a fallback estimate, not a grade: counting one as
        # a hallucination turns a provider outage into a fake quality regression.
        judged = [r for r in answered if not r.judge_degraded]
        f1s = [r.answer_f1 for r in results if r.answer_f1 is not None]

        recall = _mean([r.recall_at_k for r in retrieval])
        mrr_score = _mean([r.mrr for r in retrieval])
        ndcg = _mean([r.ndcg_at_k for r in retrieval])
        groundedness = (
            _mean([1.0 if r.judged_grounded else 0.0 for r in judged]) if judged else None
        )
        hallucination = (
            _mean([0.0 if r.judged_grounded else 1.0 for r in judged]) if judged else None
        )
        relevance = _mean([1.0 if r.judged_relevant else 0.0 for r in judged]) if judged else None
        citations = _mean([r.citation_accuracy for r in answered]) if answered else None
        answer_f1 = _mean(f1s)
        guard_rate = (
            _mean([1.0 if r.grounded_guard else 0.0 for r in answered]) if answered else None
        )
        # Nothing answered at all (e.g. CI with no LLM key) leaves answer quality
        # unmeasured rather than under-covered, so coverage is vacuously fine.
        judged_fraction = (len(judged) / len(answered)) if answered else 1.0

        ev = self.settings.eval
        thresholds = {
            "min_recall_at_k": ev.min_recall_at_k,
            "min_mrr": ev.min_mrr,
            "min_groundedness": ev.min_groundedness,
            "max_hallucination_rate": ev.max_hallucination_rate,
            "min_judged_fraction": ev.min_judged_fraction,
        }
        passed = (
            recall >= ev.min_recall_at_k
            and mrr_score >= ev.min_mrr
            and judged_fraction >= ev.min_judged_fraction
            and (groundedness is None or groundedness >= ev.min_groundedness)
            and (hallucination is None or hallucination <= ev.max_hallucination_rate)
        )
        judges = sorted({r.judge for r in answered})

        return Scorecard(
            dataset=dataset,
            k=ev.retrieval_k,
            num_cases=len(results),
            num_answered=len(answered),
            recall_at_k=recall,
            mrr=mrr_score,
            ndcg_at_k=ndcg,
            groundedness=groundedness,
            hallucination_rate=hallucination,
            answer_relevance=relevance,
            citation_accuracy=citations,
            answer_f1=answer_f1,
            guard_grounded_rate=guard_rate,
            judge="+".join(judges) if judges else "none",
            thresholds=thresholds,
            passed=passed,
            num_judge_degraded=len(answered) - len(judged),
            judged_fraction=judged_fraction,
            cases=results,
        )
