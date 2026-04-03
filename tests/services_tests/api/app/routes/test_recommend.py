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

    return TestClient(app)


def test_recommend_endpoint(monkeypatch):
    client = _client(monkeypatch)
    response = client.get("/recommend?n=5&genre=Action")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["appid"] == 1
