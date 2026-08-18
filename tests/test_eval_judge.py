from __future__ import annotations

from docsqa.eval.judge import LexicalJudge, LlmJudge
from docsqa.llm.base import ChatMessage, LlmError


class FakeLlm:
    name = "fake"

    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float, max_tokens: int
    ) -> str:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        assert self._reply is not None
        return self._reply


async def test_lexical_judge_grounded_when_overlap_high() -> None:
    judge = LexicalJudge()
    verdict = await judge.judge(
        question="How do I reset my password?",
        answer="Open settings and reset the password from the security page.",
        context="To reset the password, open settings and use the security page.",
        reference="Open settings, security, reset password.",
    )
    assert verdict.grounded is True
    assert verdict.judge == "lexical"


async def test_lexical_judge_not_grounded_when_disjoint() -> None:
    judge = LexicalJudge()
    verdict = await judge.judge(
        question="What is the rate limit?",
        answer="Bananas are an excellent source of potassium.",
        context="The API rate limit is sixty requests per minute.",
        reference=None,
    )
    assert verdict.grounded is False


async def test_llm_judge_parses_json_verdict() -> None:
    reply = '{"grounded": true, "relevant": true, "faithfulness": 0.9, "rationale": "ok"}'
    judge = LlmJudge(FakeLlm(reply=reply))
    verdict = await judge.judge(question="q", answer="a", context="c", reference=None)
    assert verdict.grounded is True
    assert verdict.relevant is True
    assert verdict.faithfulness == 0.9
    assert verdict.judge == "llm"


async def test_llm_judge_extracts_json_with_surrounding_text() -> None:
    reply = (
        "Here is my verdict:\n"
        '{"grounded": false, "relevant": true, "faithfulness": 0.2}\nThanks'
    )
    judge = LlmJudge(FakeLlm(reply=reply))
    verdict = await judge.judge(question="q", answer="a", context="c", reference=None)
    assert verdict.grounded is False
    assert verdict.relevant is True


async def test_llm_judge_falls_back_on_bad_json() -> None:
    judge = LlmJudge(FakeLlm(reply="not json at all"))
    verdict = await judge.judge(
        question="q", answer="shared words here", context="shared words here", reference=None
    )
    assert verdict.judge == "lexical"


async def test_llm_judge_falls_back_on_error() -> None:
    judge = LlmJudge(FakeLlm(error=LlmError("boom")))
    verdict = await judge.judge(question="q", answer="a", context="a", reference=None)
    assert verdict.judge == "lexical"
