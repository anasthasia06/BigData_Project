import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from services.frontend.api_client import APIClient


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_search_games_calls_expected_endpoint(monkeypatch):
    called = {}

    def fake_get(url, params=None, timeout=10):
        called["url"] = url
        called["params"] = params
        return _Resp({"total": 1, "items": [{"appid": 1}]})

    monkeypatch.setattr("services.frontend.api_client.requests.get", fake_get)
    client = APIClient(base_url="http://localhost:8000")
    out = client.search_games(q="action", genres=["Action"], size=5)

    assert called["url"].endswith("/search")
    assert called["params"]["q"] == "action"
    assert called["params"]["genres"] == "Action"
    assert out["total"] == 1


def test_recommend_games_calls_expected_endpoint(monkeypatch):
    called = {}

    def fake_get(url, params=None, timeout=10):
        called["url"] = url
        called["params"] = params
        return _Resp({"total": 1, "items": [{"appid": 1}]})

    monkeypatch.setattr("services.frontend.api_client.requests.get", fake_get)
    client = APIClient(base_url="http://localhost:8000")
    out = client.recommend_games(n=7, genre="RPG")

    assert called["url"].endswith("/recommend")
    assert called["params"]["n"] == 7
    assert called["params"]["genre"] == "RPG"
    assert out["items"][0]["appid"] == 1
