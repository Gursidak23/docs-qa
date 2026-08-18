"""Pytest configuration: skip ``integration`` tests without a test Postgres.

Integration tests run against a real Postgres+pgvector supplied via the
``DOCSQA_TEST_POSTGRES__DSN`` environment variable. When it is unset they are
skipped so the unit suite stays runnable anywhere with no external services.
"""

from __future__ import annotations

import os

import pytest

TEST_DSN_ENV = "DOCSQA_TEST_POSTGRES__DSN"
TEST_DSN = os.environ.get(TEST_DSN_ENV)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if TEST_DSN:
        return
    skip_marker = pytest.mark.skip(
        reason=f"{TEST_DSN_ENV} not set; skipping integration tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
