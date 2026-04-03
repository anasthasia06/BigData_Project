from pathlib import Path
from typing import Dict, List, Optional

from pipelines.training.train_model import load_model, recommend_top_n


def load_ranking_model(model_path: Path) -> Dict:
	return load_model(model_path)


def get_recommendations(model: Dict, n: int = 10, genre: Optional[str] = None) -> List[Dict]:
	return recommend_top_n(model=model, n=n, genre=genre)

