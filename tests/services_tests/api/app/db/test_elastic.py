import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


class _FakeClient:
    def __init__(self):
        self.last = None

    def search(self, index, body):
        self.last = (index, body)
        return {"hits": {"hits": [{"_score": 1.0, "_source": {"appid": 1, "name": "A", "genres": []}}]}}

    def close(self):
        return None


def test_search_games_builds_query(monkeypatch):
    monkeypatch.setattr("services.api.app.db.elastic._es_client_factory", lambda _uri: _FakeClient())
    from services.api.app.db.elastic import ElasticDB

    db = ElasticDB("http://localhost:9200", "steam_games")
    out = db.search_games(query="action", genres=["Action"], min_positive_ratio=0.5, max_price=30, size=5)
    assert len(out) == 1
    assert out[0]["appid"] == 1
