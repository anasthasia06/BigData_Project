import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))


class _FakeElastic:
    def search_games(self, **kwargs):
        return [{"appid": 1, "name": "A", "genres": ["Action"], "price": 10.0, "positive_ratio": 0.8, "score": 1.0}]

    def close(self):
        return None


def _client(monkeypatch):
    monkeypatch.setattr("services.api.app.main.ElasticDB", lambda *args, **kwargs: _FakeElastic())
    monkeypatch.setattr(
        "services.api.app.main.load_ranking_model",
        lambda *_args, **_kwargs: {"ranking": []},
    )
    from services.api.app.main import app

    app.state.elastic = _FakeElastic()
    app.state.model = {"ranking": []}
    from services.api.app.core.cache import TTLCache
    app.state.search_cache = TTLCache(default_ttl=60)
    app.state.recommend_cache = TTLCache(default_ttl=60)
    app.state.endpoint_latency = {}

    return TestClient(app)


def test_search_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/search?q=action&size=5")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 1


def test_search_endpoint_cache(monkeypatch):
    client = _client(monkeypatch)
    _ = client.get("/search?q=action&size=5")
    _ = client.get("/search?q=action&size=5")
    from services.api.app.main import app
    assert app.state.search_cache.stats()["hits"] >= 1
