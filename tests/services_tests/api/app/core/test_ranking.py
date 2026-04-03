import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from services.api.app.core.ranking import get_recommendations, load_ranking_model
from services.api.app.core.ranking import build_ranking_engine, get_recommendations_fast


def test_load_ranking_model(tmp_path: Path):
    model_file = tmp_path / "m.pkl"
    with model_file.open("wb") as f:
        pickle.dump({"ranking": []}, f)
    model = load_ranking_model(model_file)
    assert "ranking" in model


def test_get_recommendations_with_genre_filter():
    model = {
        "ranking": [
            {"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0},
            {"appid": 2, "name": "B", "genres": ["RPG"], "ranking_score": 0.9},
        ]
    }
    out = get_recommendations(model, n=10, genre="Action")
    assert len(out) == 1
    assert out[0]["appid"] == 1


def test_ranking_engine_fast_path():
    model = {
        "ranking": [
            {"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0},
            {"appid": 2, "name": "B", "genres": ["RPG"], "ranking_score": 0.9},
        ]
    }
    engine = build_ranking_engine(model)
    out = get_recommendations_fast(engine, n=10, genre="Action")
    assert len(out) == 1
    assert out[0]["appid"] == 1
