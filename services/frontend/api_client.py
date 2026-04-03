import os
from typing import Dict, List, Optional

import requests


class APIClient:
	def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
		self.base_url = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
		self.timeout = timeout

	def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
		url = f"{self.base_url}{path}"
		response = requests.get(url, params=params, timeout=self.timeout)
		response.raise_for_status()
		return response.json()

	def health(self) -> Dict:
		return self._get("/")

	def search_games(
		self,
		q: Optional[str] = None,
		genres: Optional[List[str]] = None,
		min_positive_ratio: Optional[float] = None,
		max_price: Optional[float] = None,
		size: int = 10,
	) -> Dict:
		params: Dict = {"size": size}
		if q:
			params["q"] = q
		if genres:
			params["genres"] = ",".join(genres)
		if min_positive_ratio is not None:
			params["min_positive_ratio"] = min_positive_ratio
		if max_price is not None:
			params["max_price"] = max_price

		return self._get("/search", params=params)

	def recommend_games(self, n: int = 10, genre: Optional[str] = None) -> Dict:
		params: Dict = {"n": n}
		if genre:
			params["genre"] = genre
		return self._get("/recommend", params=params)

