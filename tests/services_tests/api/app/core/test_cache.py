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
