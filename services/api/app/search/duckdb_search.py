from typing import Dict, List, Optional

import duckdb


class DuckDBSearchEngine:
    """Moteur de recherche local base sur DuckDB en memoire."""

    def __init__(self, model: Dict):
        self.conn = duckdb.connect(database=":memory:")
        self._create_schema()
        self._build_index(model)

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS games (
                appid BIGINT,
                name VARCHAR,
                genres VARCHAR,
                price DOUBLE,
                positive_ratio DOUBLE,
                ranking_score DOUBLE
            )
            """
        )

    def _build_index(self, model: Dict) -> None:
        ranking = model.get("ranking", []) if isinstance(model, dict) else []
        rows: List[tuple] = []

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
                genres_list = [str(g).strip() for g in raw_genres if str(g).strip()]
            elif isinstance(raw_genres, str):
                genres_list = [g.strip() for g in raw_genres.split(",") if g.strip()]
            else:
                genres_list = []

            genres_joined = " ".join(genres_list)

            ranking_score = item.get("ranking_score")
            try:
                ranking_score = float(ranking_score) if ranking_score is not None else 0.0
            except (TypeError, ValueError):
                ranking_score = 0.0

            rows.append((appid, name, genres_joined, None, None, ranking_score))

        if rows:
            self.conn.executemany(
                "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def search_games(
        self,
        query: Optional[str] = None,
        genres: Optional[List[str]] = None,
        min_positive_ratio: Optional[float] = None,
        max_price: Optional[float] = None,
        size: int = 10,
    ) -> List[Dict]:
        where_parts = []
        params: List = []

        if query:
            where_parts.append("(LOWER(name) LIKE ? OR LOWER(genres) LIKE ?)")
            like_query = f"%{query.strip().lower()}%"
            params.extend([like_query, like_query])

        if genres:
            for g in genres:
                where_parts.append("LOWER(genres) LIKE ?")
                params.append(f"%{g.strip().lower()}%")

        if min_positive_ratio is not None:
            where_parts.append("(positive_ratio IS NULL OR positive_ratio >= ?)")
            params.append(float(min_positive_ratio))

        if max_price is not None:
            where_parts.append("(price IS NULL OR price <= ?)")
            params.append(float(max_price))

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = f"""
            SELECT
                appid,
                name,
                genres,
                price,
                positive_ratio,
                CASE
                    WHEN ? IS NOT NULL AND (LOWER(name) LIKE ? OR LOWER(genres) LIKE ?) THEN 2.0
                    ELSE COALESCE(ranking_score, 0.0)
                END AS score
            FROM games
            {where_sql}
            ORDER BY score DESC
            LIMIT ?
        """

        query_marker = query.strip().lower() if query else None
        scoring_like = f"%{query_marker}%" if query_marker else None
        full_params = [query_marker, scoring_like, scoring_like] + params + [int(size)]
        rows = self.conn.execute(sql, full_params).fetchall()

        payload: List[Dict] = []
        for row in rows:
            genres_out = [g for g in str(row[2]).split(" ") if g]
            payload.append(
                {
                    "appid": int(row[0]),
                    "name": str(row[1]),
                    "genres": genres_out,
                    "price": row[3],
                    "positive_ratio": row[4],
                    "score": float(row[5]) if row[5] is not None else None,
                }
            )
        return payload

    def close(self) -> None:
        self.conn.close()
