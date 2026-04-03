import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.features.build_features import build_features, run_feature_pipeline


def test_build_features_adds_expected_columns_and_sorts():
    df = pd.DataFrame(
        {
            "appid": [1, 2],
            "name": ["A", "B"],
            "genres": ["['Action']", "['RPG']"],
            "positive_ratio": [0.9, 0.5],
            "recommendations": [100, 10],
            "peak_ccu": [1000, 100],
            "price": [10, 20],
        }
    )

    out = build_features(df)

    assert "ranking_score" in out.columns
    assert "genre_count" in out.columns
    assert out.iloc[0]["appid"] == 1


def test_build_features_parses_genres_to_list():
    df = pd.DataFrame(
        {
            "appid": [1],
            "name": ["A"],
            "genres": ["['Action', 'Indie']"],
            "positive_ratio": [0.8],
            "recommendations": [50],
            "peak_ccu": [500],
            "price": [0],
        }
    )

    out = build_features(df)
    assert out.iloc[0]["genres"] == ["Action", "Indie"]
    assert out.iloc[0]["genre_count"] == 2


def test_run_feature_pipeline_writes_output(tmp_path: Path):
    clean_csv = tmp_path / "games_clean.csv"
    feat_csv = tmp_path / "features" / "games_features.csv"

    pd.DataFrame(
        {
            "appid": [1, 2],
            "name": ["A", "B"],
            "genres": ["Action", "RPG"],
            "positive_ratio": [0.7, 0.6],
            "recommendations": [30, 20],
            "peak_ccu": [300, 200],
            "price": [5, 10],
        }
    ).to_csv(clean_csv, index=False)

    saved = run_feature_pipeline(input_path=clean_csv, output_path=feat_csv)

    assert saved == feat_csv
    assert feat_csv.exists()
    out = pd.read_csv(feat_csv)
    assert len(out) == 2
