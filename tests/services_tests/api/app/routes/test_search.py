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


class _FailingElastic:
    def search_games(self, **kwargs):
        raise RuntimeError("elastic down")

    def close(self):
        return None


def test_search_endpoint_fallback_when_elastic_fails(monkeypatch):
    monkeypatch.setattr("services.api.app.main.ElasticDB", lambda *args, **kwargs: _FailingElastic())
    monkeypatch.setattr(
        "services.api.app.main.load_ranking_model",
        lambda *_args, **_kwargs: {
            "ranking": [
                {"appid": 10, "name": "Action Hero", "genres": ["Action"], "ranking_score": 0.9},
                {"appid": 20, "name": "Puzzle Land", "genres": ["Puzzle"], "ranking_score": 0.8},
            ]
        },
    )
    from services.api.app.main import app
    from services.api.app.core.cache import TTLCache

    app.state.elastic = _FailingElastic()
    app.state.model = {
        "ranking": [
            {"appid": 10, "name": "Action Hero", "genres": ["Action"], "ranking_score": 0.9},
            {"appid": 20, "name": "Puzzle Land", "genres": ["Puzzle"], "ranking_score": 0.8},
        ]
    }
    app.state.search_cache = TTLCache(default_ttl=60)
    app.state.recommend_cache = TTLCache(default_ttl=60)
    app.state.endpoint_latency = {}

    client = TestClient(app)
    response = client.get("/search?q=action&size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 10


def test_search_endpoint_normalizes_invalid_elastic_rows(monkeypatch):
    class _BadDataElastic:
        def search_games(self, **kwargs):
            return [
                {"appid": "abc", "name": "Broken", "genres": ["Action"], "price": "10", "positive_ratio": "0.8", "score": "1"},
                {"appid": "42", "name": "Good", "genres": "Action,RPG", "price": "19.99", "positive_ratio": "0.9", "score": "2.0"},
            ]

        def close(self):
            return None

    monkeypatch.setattr("services.api.app.main.ElasticDB", lambda *args, **kwargs: _BadDataElastic())
    monkeypatch.setattr(
        "services.api.app.main.load_ranking_model",
        lambda *_args, **_kwargs: {"ranking": []},
    )
    from services.api.app.main import app
    from services.api.app.core.cache import TTLCache

    app.state.elastic = _BadDataElastic()
    app.state.model = {"ranking": []}
    app.state.search_cache = TTLCache(default_ttl=60)
    app.state.recommend_cache = TTLCache(default_ttl=60)
    app.state.endpoint_latency = {}

    client = TestClient(app)
    response = client.get("/search?q=action&size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 42
