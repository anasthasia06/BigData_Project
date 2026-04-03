from pathlib import Path
from typing import Dict, List, Optional

from pipelines.training.train_model import load_model, recommend_top_n


class RankingEngine:
	"""Moteur de recommandation en RAM avec index genre pour accès rapide."""

	def __init__(self, model: Dict):
		self.model = model
		self.ranking: List[Dict] = model.get("ranking", [])
		self.by_genre: Dict[str, List[Dict]] = {}
		for item in self.ranking:
			for genre in item.get("genres", []):
				g = str(genre).strip().lower()
				if not g:
					continue
				self.by_genre.setdefault(g, []).append(item)

	def recommend(self, n: int = 10, genre: Optional[str] = None) -> List[Dict]:
		if not genre:
			return self.ranking[:n]
		return self.by_genre.get(genre.strip().lower(), [])[:n]


def load_ranking_model(model_path: Path) -> Dict:
	return load_model(model_path)


def build_ranking_engine(model: Dict) -> RankingEngine:
	return RankingEngine(model)


def get_recommendations(model: Dict, n: int = 10, genre: Optional[str] = None) -> List[Dict]:
	return recommend_top_n(model=model, n=n, genre=genre)


def get_recommendations_fast(engine: RankingEngine, n: int = 10, genre: Optional[str] = None) -> List[Dict]:
	return engine.recommend(n=n, genre=genre)

