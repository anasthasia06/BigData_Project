import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.indexing.elastic_index import (
    bulk_index_documents,
    create_index,
    get_index_body,
    run_search_smoke_tests,
    search_games,
    to_index_documents,
)


class _FakeIndices:
    def __init__(self):
        self._exists = False
        self.created_body = None

    def exists(self, index):
        return self._exists

    def delete(self, index):
        self._exists = False

    def create(self, index, body):
        self._exists = True
        self.created_body = body


class _FakeClient:
    def __init__(self):
        self.indices = _FakeIndices()
        self.search_calls = []

    def search(self, index, body):
        self.search_calls.append((index, body))
        return {
            "hits": {
                "hits": [
                    {"_score": 1.0, "_source": {"appid": 1, "name": "Game A"}},
                    {"_score": 0.9, "_source": {"appid": 2, "name": "Game B"}},
                ]
            }
        }

    def close(self):
        return None


def test_get_index_body_contains_expected_fields():
    body = get_index_body()
    props = body["mappings"]["properties"]
    assert "name" in props
    assert props["name"]["type"] == "text"
    assert "genres" in props


def test_create_index_creates_mapping():
    client = _FakeClient()
    result = create_index(client=client, index_name="steam_games_test", recreate=True)
    assert result["created"] is True
    assert client.indices.created_body is not None


def test_to_index_documents_parses_list_and_types():
    df = pd.DataFrame(
        {
            "appid": [10],
            "name": ["Game X"],
            "genres": ["['Action', 'RPG']"],
            "price": [19.99],
            "positive_ratio": [0.87],
            "recommendations": [120],
            "peak_ccu": [1000],
        }
    )
    docs = to_index_documents(df)
    assert len(docs) == 1
    assert docs[0]["appid"] == 10
    assert docs[0]["genres"] == ["Action", "RPG"]


def test_bulk_index_documents_uses_helper(monkeypatch):
    calls = {"count": 0}

    def fake_bulk(_client, actions):
        calls["count"] += 1
        return len(actions), []

    monkeypatch.setattr("pipelines.indexing.elastic_index._bulk_helper", fake_bulk)

    docs = [{"appid": i, "name": f"G{i}", "genres": []} for i in range(5)]
    indexed = bulk_index_documents(docs, client=_FakeClient(), index_name="x", batch_size=2)
    assert indexed == 5
    assert calls["count"] == 3


def test_search_games_builds_advanced_query():
    client = _FakeClient()
    out = search_games(
        query="rpg",
        genres=["RPG"],
        min_positive_ratio=0.6,
        max_price=20,
        client=client,
        index_name="steam_games_test",
    )
    assert len(out) == 2
    _index, body = client.search_calls[-1]
    filters = body["query"]["bool"]["filter"]
    assert any("terms" in f for f in filters)
    assert any("range" in f and "price" in f["range"] for f in filters)


def test_run_search_smoke_tests_returns_counts(monkeypatch):
    monkeypatch.setattr(
        "pipelines.indexing.elastic_index.search_games",
        lambda **kwargs: [{"appid": 1}, {"appid": 2}],
    )
    stats = run_search_smoke_tests(client=_FakeClient(), index_name="steam_games_test")
    assert stats == {"simple_count": 2, "advanced_count": 2}
