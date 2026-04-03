import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	api_host: str = os.getenv("API_HOST", "0.0.0.0")
	api_port: int = int(os.getenv("API_PORT", "8000"))

	elastic_uri: str = os.getenv("ELASTIC_URI", "http://localhost:9200")
	elastic_index: str = os.getenv("ELASTIC_INDEX", "steam_games")

	mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
	mongo_db: str = os.getenv("MONGO_DB", "steam_games")
	mongo_coll: str = os.getenv("MONGO_COLL", "games")

	model_path: str = os.getenv("MODEL_PATH", str(Path(__file__).resolve().parents[3] / "models" / "recommender.pkl"))


def get_settings() -> Settings:
	return Settings()


def get_model_path() -> Path:
	return Path(get_settings().model_path)

