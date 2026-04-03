import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from services.api.app.core.cache import TTLCache


def test_ttl_cache_set_get():
    cache = TTLCache(default_ttl=5)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_ttl_cache_clear():
    cache = TTLCache(default_ttl=5)
    cache.set("k", "v")
    cache.clear()
    assert cache.get("k") is None


def test_ttl_cache_stats_and_get_or_set():
    cache = TTLCache(default_ttl=5)

    value = cache.get_or_set("a", lambda: 123)
    assert value == 123

    value2 = cache.get_or_set("a", lambda: 999)
    assert value2 == 123

    stats = cache.stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
