"""Shared helpers for integration tests backed by an external Postgres+pgvector.

Integration tests need a real Postgres with the ``vector`` extension available.
Provide one via the ``DOCSQA_TEST_POSTGRES__DSN`` environment variable (an async
DSN, e.g. ``postgresql+asyncpg://docsqa:docsqa@localhost:5432/docsqa``). Any
Postgres works — a native install, a managed instance, etc. — no Docker required.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text

from docsqa.config import Settings
from docsqa.storage.db import dispose_engine, get_engine
from docsqa.storage.orm import Base

TEST_DSN_ENV = "DOCSQA_TEST_POSTGRES__DSN"


@asynccontextmanager
async def docsqa_settings() -> AsyncIterator[Settings]:
    """Yield Settings bound to the external test Postgres with a fresh schema.

    Tables are dropped and recreated around each use so tests are isolated even
    against a long-lived database.
    """
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        raise RuntimeError(
            f"{TEST_DSN_ENV} must point at a Postgres+pgvector database to run "
            "integration tests"
        )

    settings = Settings()
    settings.postgres.dsn = dsn

    await dispose_engine()  # bind the global engine to the test DSN
    engine = get_engine(settings)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
        yield settings
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await dispose_engine()
