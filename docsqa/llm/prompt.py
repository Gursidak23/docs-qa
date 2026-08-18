"""Prompt construction and citation parsing for grounded answers."""

from __future__ import annotations

import re

from ..models import AnswerSource, RetrievedChunk
from .base import ChatMessage

SYSTEM_PROMPT = (
    "You are a precise documentation assistant. Answer the user's question using "
    "ONLY the numbered context passages provided.\n"
    "- Cite every claim with bracketed numbers like [1] or [2][3] that refer to the "
    "context passages you used.\n"
    "- If the answer is not contained in the context, reply exactly with the sentence: "
    '"I don\'t have enough information in the provided documentation to answer that."\n'
    "- Be concise. Never invent facts, sources, or citation numbers."
)

IDK_TEXT = "I don't have enough information in the provided documentation to answer that."

_SNIPPET_CHARS = 240
_CITE_RE = re.compile(r"\[(\d+)\]")


def _snippet(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + "\u2026"


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[AnswerSource]]:
    """Render numbered context blocks and the matching citation sources."""
    lines: list[str] = []
    sources: list[AnswerSource] = []
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.heading_path or chunk.doc_title or chunk.uri
        lines.append(f"[{index}] (source: {location})\n{chunk.text}")
        sources.append(
            AnswerSource(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                uri=chunk.uri,
                title=chunk.doc_title,
                heading_path=chunk.heading_path,
                page_no=chunk.page_no,
                snippet=_snippet(chunk.text),
            )
        )
    return "\n\n".join(lines), sources


def build_messages(
    question: str, context: str, history: list[ChatMessage] | None = None
) -> list[ChatMessage]:
    """Assemble the chat messages.

    Prior conversation ``history`` (earlier user/assistant turns, without their
    context) is inserted between the system prompt and the current turn so the
    model can resolve follow-ups. Only the current turn carries fresh context.
    """
    messages: list[ChatMessage] = [ChatMessage("system", SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    user = (
        f"Question: {question}\n\n"
        f"Context passages:\n{context}\n\n"
        "Answer the question using only the context above, citing sources like [1]."
    )
    messages.append(ChatMessage("user", user))
    return messages


def build_retrieval_query(question: str, history: list[ChatMessage] | None = None) -> str:
    """Expand a follow-up into a self-contained retrieval query.

    A terse follow-up ("what about Groq?") retrieves poorly on its own, so we
    prepend the most recent user turns for lexical/semantic context. The LLM
    still answers the original ``question``; only retrieval sees the expansion.
    """
    if not history:
        return question
    prior_users = [m.content for m in history if m.role == "user"][-2:]
    if not prior_users:
        return question
    return " ".join([*prior_users, question])


def extract_citations(text: str) -> list[int]:
    """Return the distinct citation indices referenced in the answer, in order."""
    seen: list[int] = []
    for match in _CITE_RE.findall(text):
        value = int(match)
        if value not in seen:
            seen.append(value)
    return seen
