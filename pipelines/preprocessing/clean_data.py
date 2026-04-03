import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import CLEAN_CSV, GAMES_CSV, PROCESSED_DATA_DIR

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_raw_data(input_path: Path) -> pd.DataFrame:
	"""Charge le dataset brut CSV depuis le chemin donné."""
	return pd.read_csv(input_path, low_memory=False, index_col=False)


def _to_snake_case(columns: pd.Index) -> pd.Index:
	return (
		columns.str.strip()
		.str.lower()
		.str.replace(r"[\s\-/]+", "_", regex=True)
	)


def clean_games_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Applique un nettoyage de base reproductible pour le dataset Steam."""
	cleaned = df.copy()

	cleaned.columns = _to_snake_case(cleaned.columns)

	if "appid" in cleaned.columns:
		cleaned = cleaned.drop_duplicates(subset=["appid"])
	else:
		cleaned = cleaned.drop_duplicates()

	if "name" in cleaned.columns:
		cleaned["name"] = cleaned["name"].astype(str).str.strip()
		cleaned = cleaned[cleaned["name"] != ""]
		cleaned = cleaned[cleaned["name"].str.lower() != "nan"]

	numeric_cols = [
		"price",
		"positive",
		"negative",
		"recommendations",
		"peak_ccu",
	]
	for col in numeric_cols:
		if col in cleaned.columns:
			cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

	if "price" in cleaned.columns:
		cleaned = cleaned[(cleaned["price"].isna()) | (cleaned["price"] >= 0)]

	if {"positive", "negative"}.issubset(cleaned.columns):
		total_reviews = cleaned["positive"].fillna(0) + cleaned["negative"].fillna(0)
		ratio = cleaned["positive"].fillna(0) / total_reviews.replace(0, np.nan)
		cleaned["positive_ratio"] = ratio.fillna(0).round(4)

	list_like_cols = ["genres", "categories", "tags"]
	for col in list_like_cols:
		if col in cleaned.columns:
			cleaned[col] = cleaned[col].fillna("").apply(
				lambda v: [item.strip() for item in str(v).split(",") if item.strip()]
			)

	return cleaned.reset_index(drop=True)


def save_clean_data(df: pd.DataFrame, output_path: Path) -> Path:
	"""Sauvegarde le dataset nettoyé en CSV."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	return output_path


def run_cleaning_pipeline(
	input_path: Path = GAMES_CSV,
	output_path: Path = CLEAN_CSV,
) -> Path:
	"""Exécute le pipeline de nettoyage complet."""
	logging.info("Chargement des données brutes depuis %s", input_path)
	raw_df = load_raw_data(input_path)

	logging.info("Nettoyage des données (%s lignes)", len(raw_df))
	clean_df = clean_games_dataframe(raw_df)

	removed = len(raw_df) - len(clean_df)
	logging.info("Nettoyage terminé: %s lignes supprimées", removed)

	logging.info("Sauvegarde du dataset nettoyé dans %s", output_path)
	saved_path = save_clean_data(clean_df, output_path)
	return saved_path


if __name__ == "__main__":
	PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
	final_path = run_cleaning_pipeline()
	logging.info("Pipeline de nettoyage terminé: %s", final_path)
