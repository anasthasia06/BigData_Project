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

    return TestClient(app)


def test_search_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/search?q=action&size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 1
