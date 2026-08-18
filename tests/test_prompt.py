from __future__ import annotations

from docsqa.llm.base import ChatMessage
from docsqa.llm.prompt import (
    build_context,
    build_messages,
    build_retrieval_query,
    extract_citations,
)
from docsqa.models import RetrievedChunk


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        text=f"text {chunk_id}",
        heading_path=f"H{chunk_id}",
        page_no=None,
        uri=f"d{chunk_id}",
        doc_title="T",
    )


def test_build_context_numbers_blocks_and_sources() -> None:
    context, sources = build_context([_chunk(10), _chunk(11)])
    assert "[1]" in context and "[2]" in context
    assert [s.index for s in sources] == [1, 2]
    assert sources[0].chunk_id == 10


def test_build_messages_has_system_then_user() -> None:
    messages = build_messages("how do I log in?", "context here")
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "how do I log in?" in messages[1].content


def test_build_messages_inserts_history_between_system_and_current_turn() -> None:
    history = [
        ChatMessage("user", "How do I set a provider?"),
        ChatMessage("assistant", "Set DOCSQA_LLM__PROVIDER [1]."),
    ]
    messages = build_messages("what about Groq?", "ctx", history)
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "How do I set a provider?"
    assert "what about Groq?" in messages[-1].content
    assert "ctx" in messages[-1].content  # only the current turn carries context


def test_build_retrieval_query_expands_followup_with_prior_user_turns() -> None:
    history = [
        ChatMessage("user", "How do I configure the Groq provider?"),
        ChatMessage("assistant", "Set the provider and key [1]."),
    ]
    query = build_retrieval_query("what about rate limits?", history)
    assert "Groq" in query
    assert "rate limits" in query


def test_build_retrieval_query_without_history_returns_question() -> None:
    assert build_retrieval_query("standalone question", None) == "standalone question"


def test_extract_citations_distinct_in_order() -> None:
    assert extract_citations("a [2] then [1] then [2] again") == [2, 1]
    assert extract_citations("no citations here") == []
