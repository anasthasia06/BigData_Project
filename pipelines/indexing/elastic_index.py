import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import CLEAN_CSV, ELASTIC_INDEX, ELASTIC_URI

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


def _create_es_client(es_uri: str = ELASTIC_URI):
	"""Crée un client Elasticsearch à partir de l'URI de configuration."""
	try:
		from elasticsearch import Elasticsearch
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"elasticsearch n'est pas installé. Installez-le avec: pip install elasticsearch"
		) from exc
	return Elasticsearch(es_uri)


def _bulk_helper(client, actions):
	"""Wrapper du helper bulk (isolé pour faciliter les tests)."""
	from elasticsearch.helpers import bulk

	return bulk(client, actions, request_timeout=120)


def get_index_body() -> Dict:
	"""Retourne les settings + mapping de l'index Steam games."""
	return {
		"settings": {
			"number_of_shards": 1,
			"number_of_replicas": 0,
		},
		"mappings": {
			"properties": {
				"appid": {"type": "integer"},
				"name": {
					"type": "text",
					"fields": {"keyword": {"type": "keyword"}},
				},
				"genres": {"type": "keyword"},
				"price": {"type": "float"},
				"positive_ratio": {"type": "float"},
				"recommendations": {"type": "integer"},
				"peak_ccu": {"type": "integer"},
			}
		},
	}


def create_index(
	client=None,
	index_name: str = ELASTIC_INDEX,
	recreate: bool = True,
) -> Dict[str, bool]:
	"""Crée l'index Elasticsearch avec mapping/settings."""
	close_client = False
	if client is None:
		client = _create_es_client()
		close_client = True

	try:
		if client.indices.exists(index=index_name):
			if not recreate:
				return {"created": False}
			client.indices.delete(index=index_name)

		client.indices.create(index=index_name, body=get_index_body())
		return {"created": True}
	finally:
		if close_client:
			client.close()


def load_clean_data(input_path: Path = CLEAN_CSV) -> pd.DataFrame:
	"""Charge le CSV nettoyé en entrée de l'indexation."""
	return pd.read_csv(input_path, low_memory=False)


def _parse_genres(value) -> List[str]:
	if isinstance(value, list):
		return [str(v).strip() for v in value if str(v).strip()]
	if value is None:
		return []
	text = str(value).strip()
	if not text or text.lower() == "nan":
		return []
	if text.startswith("[") and text.endswith("]"):
		try:
			parsed = ast.literal_eval(text)
			if isinstance(parsed, list):
				return [str(v).strip() for v in parsed if str(v).strip()]
		except (ValueError, SyntaxError):
			pass
	return [part.strip() for part in text.split(",") if part.strip()]


def to_index_documents(df: pd.DataFrame) -> List[Dict]:
	"""Transforme le DataFrame nettoyé en documents indexables."""
	data = df.copy()
	for col in ["appid", "name", "genres", "price", "positive_ratio", "recommendations", "peak_ccu"]:
		if col not in data.columns:
			data[col] = None

	data["genres"] = data["genres"].apply(_parse_genres)

	docs: List[Dict] = []
	for _, row in data.iterrows():
		if pd.isna(row["appid"]):
			continue
		docs.append(
			{
				"appid": int(row["appid"]),
				"name": "" if pd.isna(row["name"]) else str(row["name"]),
				"genres": row["genres"],
				"price": None if pd.isna(row["price"]) else float(row["price"]),
				"positive_ratio": None
				if pd.isna(row["positive_ratio"])
				else float(row["positive_ratio"]),
				"recommendations": None
				if pd.isna(row["recommendations"])
				else int(row["recommendations"]),
				"peak_ccu": None if pd.isna(row["peak_ccu"]) else int(row["peak_ccu"]),
			}
		)
	return docs


def bulk_index_documents(
	documents: List[Dict],
	client=None,
	index_name: str = ELASTIC_INDEX,
	batch_size: int = 500,
) -> int:
	"""Indexe les documents par lots et retourne le nombre indexé."""
	close_client = False
	if client is None:
		client = _create_es_client()
		close_client = True

	indexed_total = 0
	try:
		for start in range(0, len(documents), batch_size):
			batch = documents[start : start + batch_size]
			actions = [
				{
					"_op_type": "index",
					"_index": index_name,
					"_id": doc["appid"],
					"_source": doc,
				}
				for doc in batch
			]
			success, _ = _bulk_helper(client, actions)
			indexed_total += int(success)
		return indexed_total
	finally:
		if close_client:
			client.close()


def search_games(
	query: Optional[str] = None,
	genres: Optional[List[str]] = None,
	min_positive_ratio: Optional[float] = None,
	max_price: Optional[float] = None,
	size: int = 10,
	client=None,
	index_name: str = ELASTIC_INDEX,
) -> List[Dict]:
	"""Recherche simple/avancée dans l'index Elasticsearch."""
	close_client = False
	if client is None:
		client = _create_es_client()
		close_client = True

	must = []
	filters = []

	if query:
		must.append({"multi_match": {"query": query, "fields": ["name^3", "genres"]}})
	else:
		must.append({"match_all": {}})

	if genres:
		filters.append({"terms": {"genres": genres}})
	if min_positive_ratio is not None:
		filters.append({"range": {"positive_ratio": {"gte": float(min_positive_ratio)}}})
	if max_price is not None:
		filters.append({"range": {"price": {"lte": float(max_price)}}})

	body = {
		"size": size,
		"query": {
			"bool": {
				"must": must,
				"filter": filters,
			}
		},
	}

	try:
		response = client.search(index=index_name, body=body)
		hits = response.get("hits", {}).get("hits", [])
		return [
			{
				"score": hit.get("_score"),
				**hit.get("_source", {}),
			}
			for hit in hits
		]
	finally:
		if close_client:
			client.close()


def run_search_smoke_tests(client=None, index_name: str = ELASTIC_INDEX) -> Dict[str, int]:
	"""Exécute quelques requêtes simples/avancées pour valider l'index."""
	simple_results = search_games(query="action", size=5, client=client, index_name=index_name)
	advanced_results = search_games(
		query="rpg",
		genres=["RPG"],
		min_positive_ratio=0.6,
		max_price=30,
		size=5,
		client=client,
		index_name=index_name,
	)
	return {
		"simple_count": len(simple_results),
		"advanced_count": len(advanced_results),
	}


def run_elasticsearch_pipeline(
	input_path: Path = CLEAN_CSV,
	index_name: str = ELASTIC_INDEX,
	recreate_index: bool = True,
) -> Dict[str, int]:
	"""Exécute le pipeline: create index -> bulk index -> smoke tests."""
	client = _create_es_client()
	try:
		logging.info("Création index Elasticsearch: %s", index_name)
		create_index(client=client, index_name=index_name, recreate=recreate_index)

		logging.info("Chargement du dataset nettoyé: %s", input_path)
		clean_df = load_clean_data(input_path)
		docs = to_index_documents(clean_df)

		logging.info("Indexation de %s documents", len(docs))
		indexed_count = bulk_index_documents(docs, client=client, index_name=index_name)

		smoke = run_search_smoke_tests(client=client, index_name=index_name)
		return {
			"indexed_count": indexed_count,
			"simple_count": smoke["simple_count"],
			"advanced_count": smoke["advanced_count"],
		}
	finally:
		client.close()


if __name__ == "__main__":
	stats = run_elasticsearch_pipeline()
	logging.info("Pipeline Elasticsearch terminé: %s", stats)

