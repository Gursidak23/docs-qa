"""Centralized Prometheus metrics for the Q&A assistant.

The API process exposes these on ``/metrics``; Prometheus scrapes and aggregates.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

DOCS_INGESTED = Counter(
    "docsqa_documents_ingested_total",
    "Source documents ingested, labeled by source type and outcome.",
    ["source_type", "outcome"],
)

CHUNKS_INDEXED = Counter(
    "docsqa_chunks_indexed_total",
    "Chunks written to the index, labeled by action (insert|update|skip|delete).",
    ["action"],
)

EMBED_LATENCY = Histogram(
    "docsqa_embed_latency_seconds",
    "Latency of an embedding batch.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

RETRIEVE_LATENCY = Histogram(
    "docsqa_retrieve_latency_seconds",
    "Latency of a retrieval stage.",
    ["stage"],  # vector | lexical | hybrid | rerank
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

LLM_LATENCY = Histogram(
    "docsqa_llm_latency_seconds",
    "End-to-end latency of an LLM completion.",
    ["provider"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

LLM_TOKENS = Counter(
    "docsqa_llm_tokens_total",
    "Tokens consumed, labeled by provider and direction (prompt|completion).",
    ["provider", "direction"],
)

LLM_FALLBACKS = Counter(
    "docsqa_llm_fallbacks_total",
    "Times the primary LLM provider failed and a fallback was used.",
    ["from_provider", "to_provider"],
)

ANSWERS = Counter(
    "docsqa_answers_total",
    "Answers produced, labeled by outcome (answered|idk|error).",
    ["outcome"],
)

GROUNDED = Counter(
    "docsqa_grounded_total",
    "Answers labeled by groundedness (grounded|ungrounded).",
    ["result"],
)

QUESTIONS = Counter(
    "docsqa_questions_total",
    "Total questions received.",
)

CACHE_EVENTS = Counter(
    "docsqa_cache_events_total",
    "Cache hits/misses, labeled by cache (answer|embed) and event (hit|miss).",
    ["cache", "event"],
)

INDEX_SIZE = Gauge(
    "docsqa_index_chunks",
    "Approximate number of chunks currently indexed.",
)
