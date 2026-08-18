"""FastAPI application factory.

Exposes ``/health`` and ``/metrics`` plus the Q&A control-plane routes (ingest
documents, ask questions with citations, hybrid search) under ``/api/v1`` and a
minimal chat UI at ``/``. The OpenAPI schema is served at ``/docs``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import FileResponse, Response

from .. import __version__
from .. import metrics as _metrics  # noqa: F401  (ensures metrics are registered)
from ..config import get_settings
from ..logging_setup import configure_logging

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "A Docs/Support Q&A assistant: ingest documents, then ask questions "
            "and get grounded answers with citations via hybrid retrieval + reranking."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    from .routes import router

    app.include_router(router)

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
