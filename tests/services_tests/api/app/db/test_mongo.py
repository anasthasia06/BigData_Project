import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


class _FakeAdmin:
    def command(self, cmd):
        assert cmd == "ping"
        return {"ok": 1}


class _FakeClient:
    def __init__(self):
        self.admin = _FakeAdmin()

    def __getitem__(self, _db_name):
        return {"games": []}

    def close(self):
        return None


def test_mongo_ping(monkeypatch):
    monkeypatch.setattr("services.api.app.db.mongo._mongo_client_factory", lambda *args, **kwargs: _FakeClient())
    from services.api.app.db.mongo import MongoDB

    db = MongoDB("mongodb://localhost:27017", "steam_games", "games")
    assert db.ping() is True
