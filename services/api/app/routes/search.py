import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Query, Request

from services.api.app.schemas.response import SearchResponse

router = APIRouter(prefix="", tags=["search"])


def _fallback_search_from_model(
	model: Dict,
	query: Optional[str],
	genres: Optional[List[str]],
	size: int,
) -> List[Dict]:
	ranking = model.get("ranking", []) if isinstance(model, dict) else []
	q = (query or "").strip().lower()
	genre_set = {g.strip().lower() for g in (genres or []) if g.strip()}

	items: List[Dict] = []
	for entry in ranking:
		name = str(entry.get("name", ""))
		entry_genres = [str(g).strip() for g in entry.get("genres", [])]
		name_l = name.lower()
		entry_genres_l = [g.lower() for g in entry_genres]

		if q and (q not in name_l) and (not any(q in g for g in entry_genres_l)):
			continue
		if genre_set and not genre_set.intersection(entry_genres_l):
			continue

		score = 1.0 if (q and q in name_l) else 0.5
		items.append(
			{
				"appid": int(entry.get("appid", 0)),
				"name": name,
				"genres": entry_genres,
				"price": None,
				"positive_ratio": None,
				"score": score,
			}
		)

	if q:
		items.sort(key=lambda x: (x.get("score", 0), x.get("name", "")), reverse=True)

	return items[:size]


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

	try:
		results = request.app.state.elastic.search_games(
			query=q,
			genres=genre_list,
			min_positive_ratio=min_positive_ratio,
			max_price=max_price,
			size=size,
		)
	except Exception as exc:
		logging.warning("Elasticsearch indisponible sur /search (%s). Fallback local active.", exc)
		results = _fallback_search_from_model(
			model=getattr(request.app.state, "model", {}),
			query=q,
			genres=genre_list,
			size=size,
		)
	payload = {"total": len(results), "items": results}
	if cache is not None:
		cache.set(cache_key, payload)

	return payload

