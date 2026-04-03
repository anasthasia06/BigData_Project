from typing import Dict, List, Optional

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import RamStorage
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.qparser import MultifieldParser


class WhooshSearchEngine:
    """Moteur de recherche local base sur Whoosh en memoire."""

    def __init__(self, model: Dict):
        self._storage = RamStorage()
        self._schema = Schema(
            appid=ID(stored=True, unique=True),
            name=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            genres=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            price=NUMERIC(stored=True, numtype=float),
            positive_ratio=NUMERIC(stored=True, numtype=float),
            ranking_score=NUMERIC(stored=True, numtype=float),
        )
        self._index = self._storage.create_index(self._schema)
        self._build_index(model)

    def _build_index(self, model: Dict) -> None:
        ranking = model.get("ranking", []) if isinstance(model, dict) else []
        writer = self._index.writer()
        for item in ranking:
            try:
                appid = int(float(item.get("appid")))
            except (TypeError, ValueError):
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                continue

            raw_genres = item.get("genres", [])
            if isinstance(raw_genres, list):
                genres = " ".join(str(g).strip() for g in raw_genres if str(g).strip())
            elif isinstance(raw_genres, str):
                genres = raw_genres
            else:
                genres = ""

            ranking_score = item.get("ranking_score")
            try:
                ranking_score = float(ranking_score) if ranking_score is not None else 0.0
            except (TypeError, ValueError):
                ranking_score = 0.0

            price = item.get("price")
            try:
                price = float(price) if price is not None else None
            except (TypeError, ValueError):
                price = None

            positive_ratio = item.get("positive_ratio")
            try:
                positive_ratio = float(positive_ratio) if positive_ratio is not None else None
            except (TypeError, ValueError):
                positive_ratio = None

            writer.update_document(
                appid=str(appid),
                name=name,
                genres=genres,
                price=price,
                positive_ratio=positive_ratio,
                ranking_score=ranking_score,
            )
        writer.commit()

    def search_games(
        self,
        query: Optional[str] = None,
        genres: Optional[List[str]] = None,
        min_positive_ratio: Optional[float] = None,
        max_price: Optional[float] = None,
        size: int = 10,
    ) -> List[Dict]:
        with self._index.searcher() as searcher:
            parser = MultifieldParser(["name", "genres"], schema=self._schema)
            q = parser.parse(query) if query else parser.parse("*")
            hits = searcher.search(q, limit=max(size * 5, size))

            results: List[Dict] = []
            genre_set = {g.strip().lower() for g in (genres or []) if g.strip()}
            for hit in hits:
                item_genres_str = hit.get("genres") or ""
                item_genres = [g for g in item_genres_str.split() if g]
                item_genres_lower = [g.lower() for g in item_genres]

                if genre_set and not genre_set.intersection(item_genres_lower):
                    continue

                price = hit.get("price")
                positive_ratio = hit.get("positive_ratio")
                if min_positive_ratio is not None and positive_ratio is not None and float(positive_ratio) < float(min_positive_ratio):
                    continue
                if max_price is not None and price is not None and float(price) > float(max_price):
                    continue

                score = float(hit.score) if hit.score is not None else 0.0
                results.append(
                    {
                        "appid": int(hit["appid"]),
                        "name": hit.get("name"),
                        "genres": item_genres,
                        "price": price,
                        "positive_ratio": positive_ratio,
                        "score": score,
                    }
                )
                if len(results) >= size:
                    break

            if not query and results:
                results.sort(key=lambda x: x.get("score", 0), reverse=True)

            return results[:size]

    def close(self) -> None:
        return None
