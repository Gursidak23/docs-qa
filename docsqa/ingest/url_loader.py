"""Fetch a single docs URL for ingestion (treated as HTML)."""

from __future__ import annotations

import httpx

_USER_AGENT = "MoonshotDocsQA/0.1 (+https://github.com/moonshot/docs-qa)"


async def fetch_url(url: str, *, timeout_seconds: float = 30.0) -> tuple[bytes, str | None]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds) as client:
        resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        return resp.content, resp.headers.get("content-type")
