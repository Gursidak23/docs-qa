from __future__ import annotations

from docsqa.config import Settings
from docsqa.eval.dataset import GoldCase
from docsqa.eval.harness import EvalHarness, relevance_vector
from docsqa.eval.judge import JudgeVerdict
from docsqa.models import AnswerResult, AnswerSource, RetrievedChunk


def _chunk(chunk_id: int, document_id: int, uri: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=f"text {chunk_id}",
        heading_path=None,
        page_no=None,
        uri=uri,
        doc_title="Doc",
    )


class FakeRetriever:
    def __init__(self, by_question: dict[str, list[RetrievedChunk]]) -> None:
        self._by_question = by_question

    async def search(
        self, session: object, query: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        return self._by_question.get(query, [])


class IdentityReranker:
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return chunks[:top_n]


class FakeAnswerer:
    def __init__(self, by_question: dict[str, AnswerResult]) -> None:
        self._by_question = by_question

    async def answer(self, session: object, question: str) -> AnswerResult:
        return self._by_question[question]


class FakeJudge:
    name = "fake"

    def __init__(self, grounded: bool, relevant: bool) -> None:
        self.grounded = grounded
        self.relevant = relevant

    async def judge(
        self, *, question: str, answer: str, context: str, reference: str | None
    ) -> JudgeVerdict:
        return JudgeVerdict(self.grounded, self.relevant, 1.0 if self.grounded else 0.0, "", "fake")


def _answer(text: str, *, sources: int, citations: list[int], outcome: str) -> AnswerResult:
    src = [
        AnswerSource(
            index=i,
            chunk_id=i,
            document_id=1,
            uri="docA",
            title="Doc",
            heading_path=None,
            page_no=None,
            snippet="s",
        )
        for i in range(1, sources + 1)
    ]
    grounded = outcome == "answered"
    return AnswerResult(text, src, citations, grounded, "fake", outcome)


def test_relevance_vector_dedupes_by_document_for_doc_level() -> None:
    case = GoldCase(id="x", question="q", relevant_uris=["docA"])
    chunks = [_chunk(1, 1, "docA"), _chunk(2, 1, "docA"), _chunk(3, 2, "docB")]
    # Two chunks share document 1, so it collapses to one ranked entry.
    assert relevance_vector(case, chunks) == [True, False]


def test_relevance_vector_per_chunk_for_chunk_level() -> None:
    case = GoldCase(id="x", question="q", relevant_chunk_ids=[2])
    chunks = [_chunk(1, 1, "docA"), _chunk(2, 1, "docA")]
    assert relevance_vector(case, chunks) == [False, True]


async def test_harness_scores_and_passes_thresholds() -> None:
    settings = Settings()
    cases = [
        GoldCase(id="ok", question="qa", reference_answer="alpha answer", relevant_uris=["docA"]),
        GoldCase(id="oos", question="qb"),  # out-of-scope: IDK, no gold
    ]
    retriever = FakeRetriever(
        {
            "qa": [_chunk(1, 1, "docA"), _chunk(2, 2, "docB")],
            "qb": [_chunk(9, 9, "docZ")],
        }
    )
    answerer = FakeAnswerer(
        {
            "qa": _answer("alpha answer [1]", sources=1, citations=[1], outcome="answered"),
            "qb": _answer("I don't know.", sources=0, citations=[], outcome="idk"),
        }
    )
    harness = EvalHarness(retriever, IdentityReranker(), answerer, FakeJudge(True, True), settings)

    card = await harness.run(None, cases, dataset="unit")

    assert card.num_cases == 2
    assert card.num_answered == 1  # the IDK case is excluded from answer metrics
    assert card.recall_at_k == 1.0  # only the annotated case counts; docA retrieved at rank 1
    assert card.mrr == 1.0
    assert card.groundedness == 1.0
    assert card.hallucination_rate == 0.0
    assert card.citation_accuracy == 1.0
    assert 0.79 <= card.answer_f1 <= 0.81
    assert card.passed is True


async def test_harness_fails_when_answer_hallucinates() -> None:
    settings = Settings()
    cases = [GoldCase(id="bad", question="qa", relevant_uris=["docA"])]
    retriever = FakeRetriever({"qa": [_chunk(1, 1, "docA")]})
    answerer = FakeAnswerer(
        {"qa": _answer("made up claim [1]", sources=1, citations=[1], outcome="answered")}
    )
    # Judge says the answer is NOT grounded -> hallucination.
    harness = EvalHarness(retriever, IdentityReranker(), answerer, FakeJudge(False, True), settings)

    card = await harness.run(None, cases, dataset="unit")

    assert card.groundedness == 0.0
    assert card.hallucination_rate == 1.0
    assert card.passed is False
