from __future__ import annotations

import docsqa.cache as cache_mod
from docsqa.cache import MemoryCache, NoopCache, make_key


def test_make_key_is_deterministic_and_distinct() -> None:
    assert make_key("a", "b") == make_key("a", "b")
    assert make_key("a", "b") != make_key("a", "c")


async def test_noop_cache_is_always_a_miss() -> None:
    cache = NoopCache()
    await cache.set("k", "v", 100)
    assert await cache.get("k") is None


async def test_memory_cache_get_set() -> None:
    cache = MemoryCache()
    assert await cache.get("missing") is None
    await cache.set("k", "v", 100)
    assert await cache.get("k") == "v"


async def test_memory_cache_expires_by_ttl(monkeypatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: clock["t"])
    cache = MemoryCache()
    await cache.set("k", "v", ttl_seconds=10)
    assert await cache.get("k") == "v"
    clock["t"] = 1011.0
    assert await cache.get("k") is None


async def test_memory_cache_lru_eviction() -> None:
    cache = MemoryCache(max_entries=2)
    await cache.set("a", "1", 100)
    await cache.set("b", "2", 100)
    await cache.get("a")  # 'a' becomes most-recently-used
    await cache.set("c", "3", 100)  # evicts least-recently-used 'b'
    assert await cache.get("b") is None
    assert await cache.get("a") == "1"
    assert await cache.get("c") == "3"
