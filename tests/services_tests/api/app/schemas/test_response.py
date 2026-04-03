import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from services.api.app.schemas.response import RecommendResponse, SearchResponse


def test_search_response_schema():
    payload = SearchResponse(
        total=1,
        items=[
            {
                "appid": 1,
                "name": "A",
                "genres": ["Action"],
                "price": 10.0,
                "positive_ratio": 0.8,
                "score": 1.0,
            }
        ],
    )
    assert payload.total == 1


def test_recommend_response_schema():
    payload = RecommendResponse(
        total=1,
        items=[{"appid": 1, "name": "A", "genres": ["Action"], "ranking_score": 1.0}],
    )
    assert payload.items[0].name == "A"
