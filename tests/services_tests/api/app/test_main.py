import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


class _FakeElastic:
    def close(self):
        return None

    def search_games(self, **kwargs):
        return [{"appid": 1, "name": "A", "genres": [], "ranking_score": 1.0}]


def _build_test_client(monkeypatch):
    monkeypatch.setattr("services.api.app.main.ElasticDB", lambda *args, **kwargs: _FakeElastic())
    monkeypatch.setattr(
        "services.api.app.main.load_ranking_model",
        lambda *_args, **_kwargs: {"ranking": [{"appid": 1, "name": "A", "genres": [], "ranking_score": 1.0}]},
    )
    from services.api.app.main import app

    return TestClient(app)


def test_healthcheck(monkeypatch):
    client = _build_test_client(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
