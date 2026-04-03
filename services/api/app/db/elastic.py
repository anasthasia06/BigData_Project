from typing import Dict, List, Optional


def _es_client_factory(uri: str):
	from elasticsearch import Elasticsearch

	return Elasticsearch(uri)


class ElasticDB:
	def __init__(self, uri: str, index_name: str):
		try:
			self.client = _es_client_factory(uri)
		except ModuleNotFoundError as exc:
			raise RuntimeError("elasticsearch package manquant") from exc
		self.index_name = index_name

	def search_games(
		self,
		query: Optional[str] = None,
		genres: Optional[List[str]] = None,
		min_positive_ratio: Optional[float] = None,
		max_price: Optional[float] = None,
		size: int = 10,
	) -> List[Dict]:
		must = [{"match_all": {}}]
		filters = []

		if query:
			must = [{"multi_match": {"query": query, "fields": ["name^3", "genres"]}}]
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
		response = self.client.search(index=self.index_name, body=body)
		hits = response.get("hits", {}).get("hits", [])
		return [{"score": h.get("_score"), **h.get("_source", {})} for h in hits]

	def close(self) -> None:
		self.client.close()

