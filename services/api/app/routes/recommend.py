from typing import Optional

from fastapi import APIRouter, Query, Request

from services.api.app.core.ranking import get_recommendations
from services.api.app.schemas.response import RecommendResponse

router = APIRouter(prefix="", tags=["recommend"])


@router.get("/recommend", response_model=RecommendResponse)
def recommend_games(
	request: Request,
	n: int = Query(default=10, ge=1, le=100),
	genre: Optional[str] = Query(default=None),
):
	model = request.app.state.model
	items = get_recommendations(model=model, n=n, genre=genre)
	return {"total": len(items), "items": items}

