import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.preprocessing.clean_data import clean_games_dataframe, run_cleaning_pipeline


def test_clean_games_dataframe_normalizes_columns_and_removes_duplicates():
    df = pd.DataFrame(
        {
            "AppID": [10, 10, 20],
            "Name": [" Game A ", " Game A ", "Game B"],
            "Price": ["1.99", "1.99", "2.49"],
        }
    )

    cleaned = clean_games_dataframe(df)

    assert list(cleaned.columns) == ["appid", "name", "price"]
    assert len(cleaned) == 2
    assert cleaned["name"].tolist() == ["Game A", "Game B"]


def test_clean_games_dataframe_creates_positive_ratio_and_filters_negative_price():
    df = pd.DataFrame(
        {
            "AppID": [1, 2],
            "Name": ["A", "B"],
            "Price": ["-1", "4.99"],
            "Positive": [10, 3],
            "Negative": [5, 1],
        }
    )

    cleaned = clean_games_dataframe(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["appid"] == 2
    assert "positive_ratio" in cleaned.columns
    assert float(cleaned.iloc[0]["positive_ratio"]) == 0.75


def test_clean_games_dataframe_splits_list_like_columns():
    df = pd.DataFrame(
        {
            "AppID": [1],
            "Name": ["A"],
            "Genres": ["Action, RPG , Indie"],
        }
    )

    cleaned = clean_games_dataframe(df)

    assert cleaned.iloc[0]["genres"] == ["Action", "RPG", "Indie"]


def test_run_cleaning_pipeline_writes_output_file(tmp_path: Path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "clean.csv"

    pd.DataFrame(
        {
            "AppID": [1, 1, 2],
            "Name": ["X", "X", "Y"],
            "Price": ["0", "0", "9.99"],
        }
    ).to_csv(input_path, index=False)

    saved_path = run_cleaning_pipeline(input_path=input_path, output_path=output_path)

    assert saved_path == output_path
    assert output_path.exists()

    out_df = pd.read_csv(output_path)
    assert len(out_df) == 2
