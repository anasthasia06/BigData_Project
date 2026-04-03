from typing import List, Optional

from pydantic import BaseModel


class GameSearchItem(BaseModel):
	appid: int
	name: str
	genres: List[str] = []
	price: Optional[float] = None
	positive_ratio: Optional[float] = None
	score: Optional[float] = None


class SearchResponse(BaseModel):
	total: int
	items: List[GameSearchItem]


class GameRecommendItem(BaseModel):
	appid: int
	name: str
	genres: List[str] = []
	ranking_score: float


class RecommendResponse(BaseModel):
	total: int
	items: List[GameRecommendItem]

