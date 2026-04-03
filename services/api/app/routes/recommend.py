from typing import Optional

from fastapi import APIRouter, Query, Request

from services.api.app.core.ranking import get_recommendations, get_recommendations_fast
from services.api.app.schemas.response import RecommendResponse

router = APIRouter(prefix="", tags=["recommend"])


@router.get("/recommend", response_model=RecommendResponse)
def recommend_games(
	request: Request,
	n: int = Query(default=10, ge=1, le=100),
	genre: Optional[str] = Query(default=None),
):
	cache_key = f"n={n}|genre={genre}"
	cache = getattr(request.app.state, "recommend_cache", None)
	if cache is not None:
		cached = cache.get(cache_key)
		if cached is not None:
			return cached

	engine = getattr(request.app.state, "ranking_engine", None)
	if engine is not None:
		items = get_recommendations_fast(engine=engine, n=n, genre=genre)
	else:
		model = request.app.state.model
		items = get_recommendations(model=model, n=n, genre=genre)

	payload = {"total": len(items), "items": items}
	if cache is not None:
		cache.set(cache_key, payload)
	return payload

