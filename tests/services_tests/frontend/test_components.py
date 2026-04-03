import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from services.frontend.components import format_genres, game_card_html


def test_format_genres():
    assert format_genres(["Action", "RPG"]) == "Action • RPG"
    assert format_genres([]) == "-"


def test_game_card_html_contains_name_and_appid():
    html = game_card_html(
        {
            "appid": 10,
            "name": "Game X",
            "genres": ["Action"],
            "score": 1.23,
            "price": 9.99,
            "positive_ratio": 0.8,
        },
        mode="search",
    )
    assert "Game X" in html
    assert "AppID: 10" in html
