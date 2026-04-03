import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLEAN_CSV, FEATURES_DATA_DIR

FEATURES_CSV = FEATURES_DATA_DIR / "games_features.csv"

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_clean_data(input_path: Path = CLEAN_CSV) -> pd.DataFrame:
	"""Charge le dataset nettoyé en entrée du feature engineering."""
	return pd.read_csv(input_path, low_memory=False)


def _parse_list_like(value):
	"""Convertit une valeur potentiellement sérialisée en liste Python."""
	if isinstance(value, list):
		return [str(v).strip() for v in value if str(v).strip()]
	if value is None:
		return []

	text = str(value).strip()
	if not text or text.lower() == "nan":
		return []

	# Cas d'une liste sérialisée via to_csv: "['Action', 'RPG']"
	if text.startswith("[") and text.endswith("]"):
		try:
			parsed = ast.literal_eval(text)
			if isinstance(parsed, list):
				return [str(v).strip() for v in parsed if str(v).strip()]
		except (ValueError, SyntaxError):
			pass

	# Fallback: split par virgule
	return [part.strip() for part in text.split(",") if part.strip()]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Crée des features simples orientées ranking/recommandation baseline."""
	feat = df.copy()

	for col in ["positive_ratio", "recommendations", "peak_ccu", "price"]:
		if col not in feat.columns:
			feat[col] = 0

	feat["positive_ratio"] = pd.to_numeric(feat["positive_ratio"], errors="coerce").fillna(0)
	feat["recommendations"] = pd.to_numeric(feat["recommendations"], errors="coerce").fillna(0)
	feat["peak_ccu"] = pd.to_numeric(feat["peak_ccu"], errors="coerce").fillna(0)
	feat["price"] = pd.to_numeric(feat["price"], errors="coerce").fillna(0)

	if "genres" in feat.columns:
		feat["genres"] = feat["genres"].apply(_parse_list_like)
	else:
		feat["genres"] = [[] for _ in range(len(feat))]

	feat["genre_count"] = feat["genres"].apply(len)
	feat["recommendations_log"] = feat["recommendations"].clip(lower=0).apply(np.log1p)
	feat["peak_ccu_log"] = feat["peak_ccu"].clip(lower=0).apply(np.log1p)

	# Score baseline simple: qualité + popularité - pénalité prix
	feat["ranking_score"] = (
		0.50 * feat["positive_ratio"].clip(lower=0, upper=1)
		+ 0.25 * feat["recommendations_log"]
		+ 0.20 * feat["peak_ccu_log"]
		- 0.05 * feat["price"].clip(lower=0)
	)

	keep_cols = [
		col
		for col in [
			"appid",
			"name",
			"genres",
			"genre_count",
			"price",
			"positive_ratio",
			"recommendations",
			"peak_ccu",
			"recommendations_log",
			"peak_ccu_log",
			"ranking_score",
		]
		if col in feat.columns
	]

	return feat[keep_cols].sort_values("ranking_score", ascending=False).reset_index(drop=True)


def save_features(df: pd.DataFrame, output_path: Path = FEATURES_CSV) -> Path:
	"""Sauvegarde les features calculées."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	return output_path


def run_feature_pipeline(
	input_path: Path = CLEAN_CSV,
	output_path: Path = FEATURES_CSV,
) -> Path:
	"""Exécute le pipeline complet de feature engineering."""
	logging.info("Chargement des données nettoyées: %s", input_path)
	clean_df = load_clean_data(input_path)

	logging.info("Construction des features (%s lignes)", len(clean_df))
	feat_df = build_features(clean_df)

	logging.info("Sauvegarde des features: %s", output_path)
	saved = save_features(feat_df, output_path)
	return saved


if __name__ == "__main__":
	path = run_feature_pipeline()
	logging.info("Feature engineering terminé: %s", path)
