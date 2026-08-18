# Docs/Support Q&A Assistant (RAG done right)

A free-to-run Retrieval-Augmented-Generation assistant over your own documents
(PDF / Markdown / HTML / text / a docs URL). Ask questions and get **grounded
answers with clickable citations** through a FastAPI backend and a minimal web
chat UI.

Built to showcase production-grade RAG engineering rather than a toy demo:

- **Hybrid retrieval** — dense vectors (`pgvector` HNSW, cosine) fused with
  lexical full-text search (Postgres `tsvector`, BM25-like `ts_rank_cd`) via
  Reciprocal Rank Fusion.
- **Cross-encoder reranking** of fused candidates (local ONNX, free).
- **Structure-aware chunking** — heading breadcrumbs + token windows with overlap.
- **Incremental re-indexing** — document `content_hash` + per-chunk `chunk_hash`
  re-embed only what actually changed.
- **Grounded answers with citations** and an explicit "I don't know" guard that
  refuses to answer when retrieval is empty or the model fails to cite sources.
- **Evaluation harness** — retrieval metrics (Recall@k, MRR, nDCG) + answer
  metrics (groundedness, answer relevance, citation accuracy, token-F1) +
  hallucination rate via LLM-as-judge with a deterministic lexical fallback,
  rendered as a JSON + markdown scorecard and gated in CI.
- **Free LLMs** — Gemini or Groq behind a provider abstraction with automatic
  fallback, token-bucket rate limiting, and answer/embedding caching to stay
  inside free-tier quotas. Optional fully-offline Ollama provider.
- **Observability** — Prometheus metrics on `/metrics`, structured logging.

Everything runs locally for $0: embeddings (`BAAI/bge-small-en-v1.5`) and the
reranker (`Xenova/ms-marco-MiniLM-L-6-v2`) are local ONNX/CPU models, storage and
retrieval are a single Postgres, and the LLM uses a free Gemini or Groq key.

## Architecture

```mermaid
flowchart LR
  src["Upload file / docs URL"] --> parse["Parser (pdf/md/html/text)"]
  parse --> chunk["Structure-aware chunker"]
  chunk --> hash["content_hash + chunk_hash"]
  hash --> embed["Embedder (bge-small, ONNX)"]
  embed --> upsert["Idempotent upsert: delete stale, embed only changed"]
  upsert --> pg[("Postgres: pgvector + tsvector")]
```

```mermaid
flowchart TD
  q["Question"] --> hybrid["Hybrid retriever"]
  hybrid --> vec["pgvector ANN (HNSW)"]
  hybrid --> lex["Postgres FTS (BM25-like)"]
  vec --> rrf["Reciprocal Rank Fusion"]
  lex --> rrf
  rrf --> rerank["Cross-encoder reranker"]
  rerank --> prompt["Prompt with numbered context"]
  prompt --> llm["LLM: Gemini -> Groq fallback (SSE)"]
  llm --> guard["Citation + groundedness guard"]
  guard --> ans["Answer + clickable citations"]
```

See [DESIGN.md](DESIGN.md) for the rationale behind each decision.

## Quick start (local)

The only external dependency is Postgres 15 (14 and 16 work too) with the
`pgvector` extension (v0.5+, needed for the HNSW index). Install it natively
(`brew install postgresql@15 pgvector`,
`sudo apt-get install postgresql-15-pgvector`, the EDB installer + pgvector on
Windows) or point `DOCSQA_POSTGRES__DSN` at any managed Postgres that has
pgvector.

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

# Create a role + database matching the default DSN (run as a Postgres
# superuser); SUPERUSER lets the migration run CREATE EXTENSION vector.
psql -c "CREATE USER docsqa WITH PASSWORD 'docsqa' SUPERUSER;"
psql -c "CREATE DATABASE docsqa OWNER docsqa;"
alembic upgrade head                 # enables pgvector + creates the schema

# Point a provider key at the app (free tiers):
export DOCSQA_LLM__GEMINI_API_KEY=...      # or DOCSQA_LLM__GROQ_API_KEY=...

docsqa ingest ./docs                 # files or directories (recursive)
docsqa ask "How do I rotate an API key?"
```

The first run downloads the small ONNX embedding/reranker models once, then works
offline.

## Run the API

Easiest path — `run.sh` creates a virtualenv, installs, migrates, and serves in
the background; `stop.sh` stops it (Python 3.11+; still needs Postgres+pgvector
and, for answering, an LLM key — see Quick start above):

```bash
bash run.sh     # -> http://localhost:8000  (chat UI); logs to docsqa.log
bash stop.sh    # stop the background server
```

Or run it manually in the foreground:

```bash
export DOCSQA_LLM__GEMINI_API_KEY=...
uvicorn docsqa.api.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000  (chat UI)
```

## CLI

```bash
docsqa ingest <paths...> [--url URL] [--force]   # ingest files/dirs and/or a docs URL
docsqa reindex [--force]                          # re-read known sources, re-embed changes
docsqa search "query" [--top-k N]                 # raw hybrid retrieval results
docsqa ask "question"                             # grounded answer + citations
docsqa eval [--dataset f.jsonl] [--json-out ...] [--md-out ...] [--strict]
docsqa bench [--dataset f.jsonl] [--runs N]       # latency p50/p90/p99
```

## API (`/api/v1`)

- `POST /documents` (multipart upload) and `POST /ingest/url` — ingest.
- `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`.
- `POST /ask` — streaming (SSE) grounded answer; `POST /ask` with
  `{"stream": false}` for a JSON response (used by tests).
- `POST /search` — raw hybrid + reranked results with scores (debug/eval).
- `GET /stats`, `GET /health`, `GET /metrics`; chat UI at `/`, OpenAPI at `/docs`.

## Configuration

All settings use the `DOCSQA_` prefix with `__` for nesting (pydantic-settings),
overridable via env or a `.env` file. See [.env.example](.env.example). Key knobs:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCSQA_POSTGRES__DSN` | `...localhost:5432/docsqa` | Async Postgres DSN |
| `DOCSQA_LLM__PROVIDER` | `gemini` | Primary LLM (`gemini`/`groq`/`ollama`) |
| `DOCSQA_LLM__FALLBACK_PROVIDER` | `groq` | Secondary on error/429 (`none` to disable) |
| `DOCSQA_LLM__GEMINI_API_KEY` | – | Free Gemini key |
| `DOCSQA_LLM__GROQ_API_KEY` | – | Free Groq key |
| `DOCSQA_EMBED__PROVIDER` | `fastembed` | `fastembed` (local) or `gemini` |
| `DOCSQA_RERANK__ENABLED` | `true` | Cross-encoder reranking |
| `DOCSQA_CACHE__BACKEND` | `memory` | Answer cache backend (`memory`/`redis`) |
| `DOCSQA_CACHE__LLM_RATE_PER_MINUTE` | `25` | Token-bucket budget (0 disables) |
| `DOCSQA_EVAL__MIN_RECALL_AT_K` | `0.7` | CI gate floor |

## Evaluation

A bundled gold dataset and corpus make the harness runnable end-to-end:

```bash
docsqa ingest docsqa/eval/gold/corpus
docsqa eval --strict --md-out scorecard.md
```

Retrieval metrics (Recall@k / MRR / nDCG) are always scored. Answer-quality
metrics activate when an LLM key is configured; without one, answer generation is
skipped gracefully so retrieval gating still runs (see `.github/workflows/eval.yml`).

## Observability

`/metrics` exposes ingestion counts, embed/retrieve/rerank/LLM latency histograms,
LLM fallbacks, cache hit/miss, answer outcomes, and groundedness counters. A
starter Grafana dashboard lives in [`ops/grafana-dashboard.json`](ops/grafana-dashboard.json).

## Testing

```bash
ruff check docsqa tests && mypy docsqa
pytest -q                       # unit; integration tests skip without a test DB
```

Integration tests run against a real `pgvector` Postgres supplied via
`DOCSQA_TEST_POSTGRES__DSN` and cover migrations, idempotent/incremental upsert,
and hybrid retrieval end-to-end. They skip automatically when it is unset:

```bash
export DOCSQA_TEST_POSTGRES__DSN=postgresql+asyncpg://docsqa:docsqa@localhost:5432/docsqa
pytest -q
```

## Project layout

```
docsqa/
  ingest/    parsers (pdf/html/md/text), chunker, indexer, url_loader
  embed/     Embedder Protocol, fastembed + gemini, caching wrapper
  retrieve/  vector, lexical, hybrid (RRF), rerank
  llm/       Protocol, gemini, groq, ollama, fallback, rate limiter, prompt
  rag/       AnswerService (retrieve -> rerank -> prompt -> guard)
  eval/      dataset, metrics, judge (LLM + lexical), harness, scorecard, gold/
  storage/   db, ORM, repositories
  api/       FastAPI app, routes, service, schemas, static chat UI
  cache.py, config.py, factory.py, metrics.py, cli.py
```
