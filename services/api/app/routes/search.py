from typing import Optional

from fastapi import APIRouter, Query, Request

from services.api.app.schemas.response import SearchResponse

router = APIRouter(prefix="", tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_games(
	request: Request,
	q: Optional[str] = Query(default=None),
	genres: Optional[str] = Query(default=None),
	min_positive_ratio: Optional[float] = Query(default=None),
	max_price: Optional[float] = Query(default=None),
	size: int = Query(default=10, ge=1, le=100),
):
	genre_list = [g.strip() for g in genres.split(",")] if genres else None
	cache_key = f"q={q}|genres={genre_list}|min_pr={min_positive_ratio}|max_p={max_price}|size={size}"
	cache = getattr(request.app.state, "search_cache", None)

	if cache is not None:
		cached = cache.get(cache_key)
		if cached is not None:
			return cached

	results = request.app.state.elastic.search_games(
		query=q,
		genres=genre_list,
		min_positive_ratio=min_positive_ratio,
		max_price=max_price,
		size=size,
	)
	payload = {"total": len(results), "items": results}
	if cache is not None:
		cache.set(cache_key, payload)

	return payload

