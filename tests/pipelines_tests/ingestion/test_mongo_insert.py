import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.ingestion.mongo_insert import (
    insert_documents,
    run_mongo_ingestion,
    to_mongo_documents,
)


class _FakeInsertResult:
    def __init__(self, n):
        self.inserted_ids = list(range(n))


class _FakeCollection:
    def __init__(self):
        self.deleted = False
        self.inserted_batches = []

    def delete_many(self, _query):
        self.deleted = True

    def insert_many(self, docs, ordered=False):
        self.inserted_batches.append((docs, ordered))
        return _FakeInsertResult(len(docs))


class _FakeDB:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, _name):
        return self._collection


class _FakeAdmin:
    def command(self, _cmd):
        return {"ok": 1}


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection
        self.admin = _FakeAdmin()
        self.closed = False

    def __getitem__(self, _db_name):
        return _FakeDB(self._collection)

    def close(self):
        self.closed = True



def test_to_mongo_documents_sets_id_and_normalizes_nan():
    df = pd.DataFrame(
        {
            "appid": [1],
            "name": ["Game"],
            "price": [float("nan")],
        }
    )

    docs = to_mongo_documents(df)

    assert docs[0]["_id"] == 1
    assert docs[0]["price"] is None



def test_insert_documents_batches_and_deletes(monkeypatch):
    collection = _FakeCollection()
    fake_client = _FakeClient(collection)

    monkeypatch.setattr(
        "pipelines.ingestion.mongo_insert._create_mongo_client",
        lambda _uri: fake_client,
    )

    docs = [{"_id": i, "name": f"G{i}"} for i in range(5)]
    stats = insert_documents(docs, batch_size=2, drop_existing=True)

    assert collection.deleted is True
    assert len(collection.inserted_batches) == 3
    assert stats["inserted"] == 5
    assert stats["errors"] == 0
    assert fake_client.closed is True



def test_run_mongo_ingestion_reads_and_inserts(monkeypatch, tmp_path: Path):
    input_csv = tmp_path / "clean.csv"
    pd.DataFrame({"appid": [10, 20], "name": ["A", "B"]}).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "pipelines.ingestion.mongo_insert.insert_documents",
        lambda docs, **kwargs: {"inserted": len(docs), "errors": 0},
    )

    stats = run_mongo_ingestion(input_path=input_csv)

    assert stats == {"inserted": 2, "errors": 0}
