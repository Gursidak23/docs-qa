"""Pluggable async cache used for answer caching (and any keyed JSON blob).

Three backends sit behind a tiny :class:`Cache` Protocol:

* :class:`NoopCache` - disabled (always a miss);
* :class:`MemoryCache` - in-process TTL + LRU map, the default; great for a
  single-process demo and for tests;
* :class:`RedisCache` - shared across processes, for multi-worker deployments.

Caching the (expensive) answer path lets us stay inside free LLM quotas: a
repeated question skips retrieval, reranking, and the LLM call entirely.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Protocol

from .logging_setup import get_logger
from .metrics import CACHE_EVENTS

log = get_logger(__name__)


def make_key(*parts: str) -> str:
    """Stable cache key from arbitrary string parts."""
    digest = hashlib.sha256("\u0000".join(parts).encode("utf-8")).hexdigest()
    return digest[:32]


class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: int) -> None: ...


class NoopCache:
    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None


class MemoryCache:
    """In-process TTL cache with a bounded LRU eviction policy."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, str]] = OrderedDict()

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if expiry < time.monotonic():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (time.monotonic() + ttl_seconds, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


class RedisCache:
    """Cross-process cache backed by Redis (lazily connected)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self.url, decode_responses=True)
        return self._client

    async def get(self, key: str) -> str | None:
        try:
            return await self._get_client().get(key)
        except Exception as exc:  # noqa: BLE001 - cache must never break a request
            log.warning("redis_cache_get_failed", error=str(exc))
            return None

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            await self._get_client().set(key, value, ex=ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - cache must never break a request
            log.warning("redis_cache_set_failed", error=str(exc))


def record_cache_event(cache: str, hit: bool) -> None:
    CACHE_EVENTS.labels(cache=cache, event="hit" if hit else "miss").inc()
