import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from services.frontend.app import create_api_client


def test_create_api_client():
    client = create_api_client()
    assert hasattr(client, "search_games")
    assert hasattr(client, "recommend_games")
