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

	results = request.app.state.elastic.search_games(
		query=q,
		genres=genre_list,
		min_positive_ratio=min_positive_ratio,
		max_price=max_price,
		size=size,
	)

	return {"total": len(results), "items": results}

