import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from services.api.app.core.config import get_model_path, get_settings


def test_get_settings_defaults():
    settings = get_settings()
    assert settings.api_port == 8000
    assert settings.elastic_index == "steam_games"


def test_get_model_path_returns_path_instance():
    model_path = get_model_path()
    assert isinstance(model_path, Path)
