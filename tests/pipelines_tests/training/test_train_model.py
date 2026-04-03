import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.training.train_model import (
    load_model,
    recommend_top_n,
    run_training_pipeline,
    train_baseline_model,
)


def test_train_baseline_model_builds_sorted_ranking():
    df = pd.DataFrame(
        {
            "appid": [1, 2, 3],
            "name": ["A", "B", "C"],
            "genres": ["['Action']", "['RPG']", "['Action']"],
            "ranking_score": [0.2, 0.9, 0.5],
        }
    )

    model = train_baseline_model(df, top_k=2)

    assert model["model_type"] == "baseline_popularity"
    assert len(model["ranking"]) == 2
    assert model["ranking"][0]["appid"] == 2


def test_recommend_top_n_with_genre_filter():
    model = {
        "ranking": [
            {"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0},
            {"appid": 2, "name": "B", "genres": ["RPG"], "ranking_score": 0.9},
            {"appid": 3, "name": "C", "genres": ["Action"], "ranking_score": 0.8},
        ]
    }

    recos = recommend_top_n(model, n=5, genre="Action")
    assert [r["appid"] for r in recos] == [1, 3]


def test_run_training_pipeline_saves_pickle(tmp_path: Path):
    feat_csv = tmp_path / "games_features.csv"
    model_file = tmp_path / "models" / "recommender.pkl"

    pd.DataFrame(
        {
            "appid": [10, 20],
            "name": ["X", "Y"],
            "genres": ["['Action']", "['Indie']"],
            "ranking_score": [0.7, 0.4],
        }
    ).to_csv(feat_csv, index=False)

    saved = run_training_pipeline(input_path=feat_csv, output_path=model_file, top_k=50)
    assert saved == model_file
    assert model_file.exists()

    model = load_model(model_file)
    assert model["model_type"] == "baseline_popularity"
    assert len(model["ranking"]) == 2
