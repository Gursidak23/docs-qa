"""Command-line interface for ingestion, search, asking, and evaluation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from .eval.harness import Scorecard

from .config import get_settings
from .factory import (
    build_answer_service,
    build_embedder,
    build_judge,
    build_retriever,
)
from .ingest.indexer import Indexer
from .ingest.parsers.registry import detect_source_type
from .ingest.sources import expand_paths
from .ingest.url_loader import fetch_url
from .logging_setup import configure_logging
from .models import IngestResult
from .storage.db import dispose_engine, session_scope
from .storage.repositories import DocumentRepository

app = typer.Typer(help="Moonshot Docs/Support Q&A assistant", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    from . import __version__

    typer.echo(__version__)


async def _ingest(paths: list[str], url: str | None, force: bool) -> list[IngestResult]:
    settings = get_settings()
    embedder = build_embedder(settings)
    indexer = Indexer(embedder=embedder, settings=settings)
    files = expand_paths(paths)

    results: list[IngestResult] = []
    async with session_scope(settings) as session:
        for path in files:
            data = path.read_bytes()
            source_type = detect_source_type(path.name)
            res = await indexer.ingest(
                session, uri=str(path), source_type=source_type, data=data, force=force
            )
            results.append(res)
            typer.echo(f"  [{res.action}] {res.uri} ({res.chunks} chunks)")
        if url:
            data, _ = await fetch_url(url)
            res = await indexer.ingest(
                session, uri=url, source_type="url", data=data, force=force
            )
            results.append(res)
            typer.echo(f"  [{res.action}] {res.uri} ({res.chunks} chunks)")
    await dispose_engine()
    return results


@app.command()
def ingest(
    paths: list[str] = typer.Argument(None, help="Files or directories to ingest"),
    url: str = typer.Option(None, "--url", help="Also fetch and ingest this docs URL"),
    force: bool = typer.Option(False, "--force", help="Re-index even if content is unchanged"),
) -> None:
    """Ingest documents (PDF/Markdown/HTML/text) and/or a docs URL."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if not paths and not url:
        typer.echo("Nothing to ingest. Pass file/dir paths or --url.")
        raise typer.Exit(code=1)

    results = asyncio.run(_ingest(paths or [], url, force))
    reindexed = sum(1 for r in results if r.action in ("indexed", "updated"))
    skipped = sum(1 for r in results if r.action == "skipped")
    chunks = sum(r.chunks for r in results if r.action in ("indexed", "updated"))
    typer.echo(
        f"Done. {reindexed} document(s) (re)indexed, {skipped} unchanged, {chunks} chunks."
    )


async def _search(query: str, top_k: int) -> None:
    settings = get_settings()
    retriever = build_retriever(settings)
    async with session_scope(settings) as session:
        hits = await retriever.search(session, query, top_k=top_k)
    if not hits:
        typer.echo("No results.")
    for hit in hits:
        where = hit.heading_path or hit.uri
        snippet = hit.text[:120].replace("\n", " ")
        typer.echo(f"  {hit.score:.4f}  [{hit.uri}] {where}")
        typer.echo(f"          {snippet}...")
    await dispose_engine()


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query"),
    top_k: int = typer.Option(10, "--top-k", help="How many fused results to show"),
) -> None:
    """Run hybrid (vector + lexical) retrieval and print fused results."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    asyncio.run(_search(query, top_k))


async def _ask(question: str) -> None:
    settings = get_settings()
    service = build_answer_service(settings)
    async with session_scope(settings) as session:
        result = await service.answer(session, question)
    await dispose_engine()

    typer.echo("")
    typer.echo(result.answer)
    typer.echo("")
    if result.sources:
        typer.echo("Sources:")
        for source in result.sources:
            where = source.heading_path or source.uri
            page = f" p.{source.page_no}" if source.page_no else ""
            typer.echo(f"  [{source.index}] {where}{page}  ({source.uri})")
    status = "grounded" if result.grounded else result.outcome
    typer.echo(f"\n[{status} via {result.provider}]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from your documents"),
) -> None:
    """Answer a question with grounded citations using hybrid RAG."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    asyncio.run(_ask(question))


def _read_local_source(uri: str) -> tuple[bytes, str] | None:
    """Read a file-backed source from disk; ``None`` if it no longer exists."""
    path = Path(uri)
    if not path.is_file():
        return None
    return path.read_bytes(), detect_source_type(path.name)


async def _reindex(force: bool) -> None:
    settings = get_settings()
    indexer = Indexer(embedder=build_embedder(settings), settings=settings)
    async with session_scope(settings) as session:
        documents = await DocumentRepository(session).list_documents(limit=1_000_000)
        for doc in documents:
            try:
                if doc.source_type == "url":
                    data, _ = await fetch_url(doc.uri)
                    source_type = "url"
                else:
                    local = await asyncio.to_thread(_read_local_source, doc.uri)
                    if local is None:
                        typer.echo(f"  [skip] {doc.uri} (source not available locally)")
                        continue
                    data, source_type = local
                res = await indexer.ingest(
                    session, uri=doc.uri, source_type=source_type, data=data, force=force
                )
                typer.echo(f"  [{res.action}] {res.uri} ({res.chunks} chunks)")
            except Exception as exc:  # noqa: BLE001 - report and continue with the rest
                typer.echo(f"  [error] {doc.uri}: {exc}")
    await dispose_engine()


@app.command()
def reindex(
    force: bool = typer.Option(False, "--force", help="Re-index even if content is unchanged"),
) -> None:
    """Re-read known file/URL sources and incrementally re-index changed chunks."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    asyncio.run(_reindex(force))


async def _eval(dataset: str) -> Scorecard:
    from .eval.dataset import load_gold
    from .eval.harness import EvalHarness

    settings = get_settings()
    service = build_answer_service(settings)
    judge = build_judge(settings, service.llm)
    harness = EvalHarness(service.retriever, service.reranker, service, judge, settings)
    cases = load_gold(dataset)
    async with session_scope(settings) as session:
        card = await harness.run(session, cases, dataset=dataset)
    await dispose_engine()
    return card


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


async def _bench(dataset: str, runs: int, top_k: int | None) -> tuple[list[float], list[float]]:
    from time import perf_counter

    from .eval.dataset import load_gold

    settings = get_settings()
    service = build_answer_service(settings)
    questions = [c.question for c in load_gold(dataset)]
    retrieve_ms: list[float] = []
    answer_ms: list[float] = []
    async with session_scope(settings) as session:
        for _ in range(runs):
            for question in questions:
                t0 = perf_counter()
                await service.retriever.search(session, question, top_k=top_k)
                retrieve_ms.append((perf_counter() - t0) * 1000)
                t1 = perf_counter()
                await service.answer(session, question)
                answer_ms.append((perf_counter() - t1) * 1000)
    await dispose_engine()
    return retrieve_ms, answer_ms


@app.command()
def bench(
    dataset: str = typer.Option(None, "--dataset", help="Gold JSONL whose questions to replay"),
    runs: int = typer.Option(1, "--runs", help="How many passes over the question set"),
    top_k: int = typer.Option(None, "--top-k", help="Override fused top-k for retrieval"),
) -> None:
    """Benchmark retrieval and end-to-end answer latency (p50/p90/p99)."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    retrieve_ms, answer_ms = asyncio.run(_bench(dataset or settings.eval.dataset_path, runs, top_k))

    typer.echo(f"{'stage':<10} {'count':>6} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}  (ms)")
    for stage, samples in (("retrieve", retrieve_ms), ("answer", answer_ms)):
        typer.echo(
            f"{stage:<10} {len(samples):>6} "
            f"{_percentile(samples, 0.5):>9.1f} {_percentile(samples, 0.9):>9.1f} "
            f"{_percentile(samples, 0.99):>9.1f} {max(samples or [0.0]):>9.1f}"
        )


@app.command("eval")
def eval_cmd(
    dataset: str = typer.Option(None, "--dataset", help="Path to a gold JSONL dataset"),
    json_out: str = typer.Option(None, "--json-out", help="Write the scorecard JSON here"),
    md_out: str = typer.Option(None, "--md-out", help="Write the markdown scorecard here"),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Exit non-zero if a metric is below threshold"
    ),
) -> None:
    """Score retrieval + answer quality against a gold dataset and gate on thresholds."""
    from .eval.scorecard import to_json, to_markdown

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    card = asyncio.run(_eval(dataset or settings.eval.dataset_path))

    markdown = to_markdown(card)
    typer.echo(markdown)
    if json_out:
        Path(json_out).write_text(to_json(card), encoding="utf-8")
    if md_out:
        Path(md_out).write_text(markdown, encoding="utf-8")
    if strict and not card.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
