import ast
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config import FEATURES_DATA_DIR, MODEL_PATH

FEATURES_CSV = FEATURES_DATA_DIR / "games_features.csv"

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_features_data(input_path: Path = FEATURES_CSV) -> pd.DataFrame:
	"""Charge la table de features."""
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


def train_baseline_model(df: pd.DataFrame, top_k: int = 1000) -> Dict:
	"""Entraîne un baseline de recommandation par ranking global."""
	data = df.copy()

	if "ranking_score" not in data.columns:
		raise ValueError("La colonne 'ranking_score' est requise pour l'entraînement.")

	for col in ["appid", "name", "genres", "ranking_score"]:
		if col not in data.columns:
			raise ValueError(f"Colonne manquante: {col}")

	data["genres"] = data["genres"].apply(_parse_genres)
	data["ranking_score"] = pd.to_numeric(data["ranking_score"], errors="coerce").fillna(0)
	data["appid"] = pd.to_numeric(data["appid"], errors="coerce")
	data = data[np.isfinite(data["appid"])]
	data["appid"] = data["appid"].astype(int)

	ranked = (
		data.sort_values("ranking_score", ascending=False)
		.drop_duplicates(subset=["appid"])
		.head(top_k)
	)

	ranking = [
		{
			"appid": int(row["appid"]),
			"name": str(row["name"]),
			"genres": row["genres"],
			"ranking_score": float(row["ranking_score"]),
		}
		for _, row in ranked.iterrows()
	]

	model = {
		"model_type": "baseline_popularity",
		"created_at": datetime.now(timezone.utc).isoformat(),
		"top_k": int(top_k),
		"ranking": ranking,
	}
	return model


def save_model(model: Dict, output_path: Path = MODEL_PATH) -> Path:
	"""Sauvegarde le modèle baseline en pickle."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("wb") as f:
		pickle.dump(model, f)
	return output_path


def load_model(model_path: Path = MODEL_PATH) -> Dict:
	"""Charge un modèle baseline pickle."""
	with model_path.open("rb") as f:
		return pickle.load(f)


def recommend_top_n(model: Dict, n: int = 10, genre: Optional[str] = None) -> List[Dict]:
	"""Renvoie un top-N global ou filtré par genre."""
	ranking = model.get("ranking", [])
	if not genre:
		return ranking[:n]

	target = genre.strip().lower()
	filtered = [
		item
		for item in ranking
		if any(str(g).strip().lower() == target for g in item.get("genres", []))
	]
	return filtered[:n]


def run_training_pipeline(
	input_path: Path = FEATURES_CSV,
	output_path: Path = MODEL_PATH,
	top_k: int = 1000,
) -> Path:
	"""Exécute entraînement + sauvegarde du modèle baseline."""
	logging.info("Chargement des features depuis %s", input_path)
	feat_df = load_features_data(input_path)

	logging.info("Entraînement baseline (top_k=%s)", top_k)
	model = train_baseline_model(feat_df, top_k=top_k)

	logging.info("Sauvegarde modèle dans %s", output_path)
	saved = save_model(model, output_path)
	return saved


if __name__ == "__main__":
	model_file = run_training_pipeline()
	logging.info("Training terminé: %s", model_file)
