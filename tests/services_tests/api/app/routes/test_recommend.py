import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))


class _FakeElastic:
    def close(self):
        return None

    def search_games(self, **kwargs):
        return []


def _client(monkeypatch):
    monkeypatch.setattr("services.api.app.main.ElasticDB", lambda *args, **kwargs: _FakeElastic())
    monkeypatch.setattr(
        "services.api.app.main.load_ranking_model",
        lambda *_args, **_kwargs: {
            "ranking": [
                {"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0},
                {"appid": 2, "name": "B", "genres": ["RPG"], "ranking_score": 0.8},
            ]
        },
    )
    from services.api.app.main import app

    app.state.elastic = _FakeElastic()
    app.state.model = {
        "ranking": [
            {"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0},
            {"appid": 2, "name": "B", "genres": ["RPG"], "ranking_score": 0.8},
        ]
    }
    from services.api.app.core.ranking import build_ranking_engine
    from services.api.app.core.cache import TTLCache
    app.state.ranking_engine = build_ranking_engine(app.state.model)
    app.state.search_cache = TTLCache(default_ttl=60)
    app.state.recommend_cache = TTLCache(default_ttl=60)
    app.state.endpoint_latency = {}

    return TestClient(app)


def test_recommend_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/recommend?n=5&genre=Action")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 1


def test_recommend_endpoint_cache(monkeypatch):
    client = _client(monkeypatch)
    _ = client.get("/recommend?n=5&genre=Action")
    _ = client.get("/recommend?n=5&genre=Action")
    from services.api.app.main import app
    assert app.state.recommend_cache.stats()["hits"] >= 1
