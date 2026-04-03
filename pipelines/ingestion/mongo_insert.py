import logging
from math import isinf, isnan
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import CLEAN_CSV, MONGO_COLL, MONGO_DB, MONGO_URI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_clean_data(input_path: Path = CLEAN_CSV) -> pd.DataFrame:
    """Charge le dataset nettoyé avant insertion MongoDB."""
    return pd.read_csv(input_path, low_memory=False)


def _normalize_scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and (isnan(value) or isinf(value)):
        return None
    return value


def to_mongo_documents(df: pd.DataFrame) -> List[Dict]:
    """Convertit un DataFrame nettoyé en documents compatibles MongoDB."""
    documents: List[Dict] = []
    for row in df.to_dict(orient="records"):
        doc = {k: _normalize_scalar(v) for k, v in row.items()}
        if "appid" in doc and doc["appid"] is not None:
            doc["_id"] = int(doc["appid"])
        documents.append(doc)
    return documents


def _create_mongo_client(mongo_uri: str):
    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pymongo n'est pas installé. Installez-le avec: pip install pymongo"
        ) from exc
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)


def insert_documents(
    documents: List[Dict],
    mongo_uri: str = MONGO_URI,
    db_name: str = MONGO_DB,
    collection_name: str = MONGO_COLL,
    batch_size: int = 1000,
    drop_existing: bool = True,
) -> Dict[str, int]:
    """Insère les documents par lot dans MongoDB."""
    client = _create_mongo_client(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]

    client.admin.command("ping")

    if drop_existing:
        coll.delete_many({})

    inserted_total = 0
    errors_total = 0

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        try:
            result = coll.insert_many(batch, ordered=False)
            inserted_total += len(result.inserted_ids)
        except Exception as exc:
            details = getattr(exc, "details", {}) or {}
            if details:
                inserted_total += int(details.get("nInserted", 0))
                errors_total += len(details.get("writeErrors", []))
            else:
                client.close()
                raise

    client.close()
    return {"inserted": inserted_total, "errors": errors_total}


def run_mongo_ingestion(input_path: Path = CLEAN_CSV) -> Dict[str, int]:
    """Exécute le chargement vers MongoDB à partir du CSV nettoyé."""
    logging.info("Chargement du dataset nettoyé depuis %s", input_path)
    clean_df = load_clean_data(input_path)
    docs = to_mongo_documents(clean_df)

    logging.info("Insertion MongoDB: %s documents", len(docs))
    stats = insert_documents(docs)
    logging.info(
        "Insertion terminée: insérés=%s erreurs=%s",
        stats["inserted"],
        stats["errors"],
    )
    return stats


if __name__ == "__main__":
    run_mongo_ingestion()
