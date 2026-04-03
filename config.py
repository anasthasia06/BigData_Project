"""
config.py — Configuration centrale du projet BigData_Project.
Toutes les variables d'environnement sont lues ici avec des valeurs par défaut locales.
"""
import os
from pathlib import Path

# ─── Racines ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = Path(os.getenv("DATA_ROOT", "/data/BigData_Project/data"))
MODELS_ROOT  = Path(os.getenv("MODELS_ROOT", "/data/BigData_Project/models"))

# ─── Chemins données ──────────────────────────────────────────────────────────
RAW_DATA_DIR       = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
FEATURES_DATA_DIR  = DATA_ROOT / "features"

# Fichiers principaux
GAMES_CSV  = RAW_DATA_DIR / "games.csv"
GAMES_JSON = RAW_DATA_DIR / "games.json"
CLEAN_CSV  = PROCESSED_DATA_DIR / "games_clean.csv"

# ─── Kaggle ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "fronkongames/steam-games-dataset")

# ─── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI    = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB     = os.getenv("MONGO_DB", "steam_games")
MONGO_COLL   = os.getenv("MONGO_COLL", "games")

# ─── Elasticsearch ────────────────────────────────────────────────────────────
ELASTIC_URI   = os.getenv("ELASTIC_URI", "http://localhost:9200")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "steam_games")

# ─── API / Frontend ───────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL  = os.getenv("API_URL", f"http://localhost:{API_PORT}")

# ─── Modèle ───────────────────────────────────────────────────────────────────
MODEL_PATH = MODELS_ROOT / "recommender.pkl"
