# Design

This document explains *why* the Docs/Support Q&A assistant is built the way it
is. The headline goal is a RAG system that is **honest** (it cites sources and
refuses to guess), **measurable** (a real eval harness, not vibes), and **free to
run** (local models + a free LLM tier), while reading like production code.

## Goals and non-goals

- **Goals:** grounded, cited answers; reproducible quality measurement;
  incremental ingestion; resilience to flaky/rate-limited free LLM tiers;
  observability; a clean, testable architecture mirroring the sibling
  `web-crawler` project (pydantic-settings, factory wiring, Protocol seams).
- **Non-goals:** multi-tenant auth, a heavyweight vector DB, agentic
  tool-use, or fine-tuning. These are deliberately out of scope to keep the
  system legible.

## Storage and retrieval in one Postgres

A single Postgres 16 with the `pgvector` extension holds both arms of retrieval,
which keeps the stack tiny and the data consistent:

- **Dense:** `chunk.embedding vector(384)` with an HNSW index
  (`vector_cosine_ops`). 384-dim matches `bge-small`, balancing quality and size.
- **Sparse:** a generated `tsvector` column with a GIN index, scored by
  `ts_rank_cd` (BM25-like). True BM25 (ParadeDB `pg_search` / `rank_bm25`) can be
  swapped in later — RRF makes the lexical scorer pluggable.

`chunk` carries provenance for citations (`heading_path`, `page_no`, char
offsets) and `chunk_hash` for incremental indexing. `source_document` tracks
`content_hash` + `version`. `query_log` is reserved for analytics/eval.

## Why hybrid + RRF

Dense retrieval captures paraphrase/semantics; lexical retrieval nails exact
terms, IDs, and rare tokens. **Reciprocal Rank Fusion** combines their ranked
lists without needing comparable raw scores: each item accrues
`sum(1 / (k + rank))`. It has a single robust constant `k`, needs no score
normalization, and reliably beats either arm alone. Fusing *ranks* (not scores)
is what makes mixing an ANN distance with a `ts_rank_cd` score sound.

## Reranking

Fusion is recall-oriented; a **cross-encoder reranker**
(`ms-marco-MiniLM-L-6-v2`, local ONNX) then reads each (query, passage) pair
jointly and reorders the top candidates for precision. It sits behind a
`Reranker` Protocol with a `NoopReranker`, so it can be disabled for latency.

## Answer generation and the groundedness guard

`AnswerService` runs retrieve → rerank → build numbered context → LLM → guard.
The prompt forces bracketed `[n]` citations and an exact "I don't have enough
information…" sentence when unsupported. Post-generation we parse citations,
keep only those mapping to a real retrieved passage, and classify the result:

- empty retrieval ⇒ `idk` (we never call the LLM);
- no valid citation ⇒ `ungrounded` (surfaced as such, not as fact);
- otherwise ⇒ `answered` + grounded.

This guard is the difference between a demo and something trustworthy: the system
would rather say "I don't know" than hallucinate.

## LLM provider abstraction (the senior signal)

A thin `LlmClient` Protocol (`complete` + `stream`) has `GeminiClient`,
`GroqClient`, and `OllamaClient` implementations. They are composed:

1. **`FallbackLlmClient`** — tries the primary, retries with backoff on
   429/5xx/timeout, then fails over to a secondary provider. Streaming uses
   *early-token* failover: if the primary errors before emitting tokens we switch;
   once tokens stream we don't switch mid-answer.
2. **`RateLimitedLlmClient`** — an async token bucket smooths calls to a
   configured requests-per-minute budget (with a small burst), the main defense
   for free quotas.

Both the CLI and API obtain their client through `build_llm`, so the resilience
behavior is identical everywhere.

## Caching

The expensive path is retrieve+rerank+LLM, so the **answer cache** keys on a
fingerprint of `(question, models, retrieval knobs)` and short-circuits repeats.
Backends sit behind a `Cache` Protocol: in-process TTL+LRU (`MemoryCache`, the
default) or `RedisCache` for multi-worker deploys; `NoopCache` disables it. Only
`answered` results are cached, and a TTL bounds staleness after re-ingest. A
separate in-process LRU memoizes **query embeddings**. The streaming path stays
live (we don't replay cached token streams) to keep the UX honest.

## Incremental re-indexing

Re-ingesting identical content is a no-op (document `content_hash`). When a
document changes, we diff at the **chunk** level: existing `chunk_hash →
embedding` is loaded, unchanged chunks reuse their stored vectors, and only
new/changed chunks are embedded before a delete-stale + insert-all upsert. This
avoids the dominant cost (embedding) on edits and keeps ids/citations stable.

## Evaluation methodology

Quality is gated, not guessed. The harness scores each gold case on two axes:

- **Retrieval:** Recall@k, MRR, nDCG@k over the hybrid ranking. Gold relevance is
  annotated by stable source URI (portable) or by ids; doc-level cases dedupe the
  ranked list to documents, chunk-level cases score per chunk.
- **Answer:** groundedness, answer relevance, citation accuracy, and SQuAD-style
  token-F1 vs. a reference. **Hallucination rate** = answered-but-not-grounded.

Judging uses **LLM-as-judge** (strict JSON verdict) with a deterministic
**lexical-overlap fallback** so runs are meaningful offline and never crash on a
bad model response. Output is a JSON + markdown scorecard; `docsqa eval --strict`
returns non-zero when a metric drops below its configured floor, which the
`Eval` GitHub workflow uses as a regression gate.

## Observability

Prometheus metrics (`/metrics`) cover ingestion counts, embed/retrieve/rerank/LLM
latency histograms, LLM fallbacks, cache hit/miss, answer outcomes, and
groundedness. Logs are structured via `structlog` (JSON in production).

## Testing strategy

- **Unit:** chunker boundaries, RRF math, citation/groundedness guard, prompt
  builder, fallback/rate-limit clients, eval metrics, caches — all dependency-free.
- **Integration (external Postgres+pgvector):** migrations, idempotent and
  incremental upsert, hybrid retrieval end-to-end. Point `DOCSQA_TEST_POSTGRES__DSN`
  at any pgvector database; these tests auto-skip when it is unset.
- **API:** `dependency_overrides` swap in a fake service (no Postgres/LLM).

## Trade-offs and future work

- `ts_rank_cd` approximates BM25; swap in `pg_search` for exact BM25 if needed.
- `bge-small` (384-dim) is the quality/footprint sweet spot; the `Embedder`
  Protocol allows larger or hosted models (note: changing dim requires a schema
  migration and re-embed).
- Streaming answers are intentionally not cached.
- Natural next steps: per-document ACLs, conversational memory, and answer-cache
  invalidation hooks on re-ingest.
```
