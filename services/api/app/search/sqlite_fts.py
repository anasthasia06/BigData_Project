import sqlite3
from threading import RLock
from typing import Dict, List, Optional


class SQLiteFTSSearchEngine:
    """Moteur de recherche local base sur SQLite FTS5 en memoire."""

    def __init__(self, model: Dict):
        self._lock = RLock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        self._build_index(model)

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS games (
                    appid INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    genres TEXT NOT NULL,
                    price REAL,
                    positive_ratio REAL,
                    ranking_score REAL
                )
                """
            )
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS games_fts
                USING fts5(appid UNINDEXED, name, genres)
                """
            )

    def _build_index(self, model: Dict) -> None:
        ranking = model.get("ranking", []) if isinstance(model, dict) else []
        rows: List[tuple] = []
        fts_rows: List[tuple] = []

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
                ranking_score = float(ranking_score) if ranking_score is not None else None
            except (TypeError, ValueError):
                ranking_score = None

            rows.append((appid, name, genres_joined, None, None, ranking_score))
            fts_rows.append((appid, name, genres_joined))

        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO games(appid, name, genres, price, positive_ratio, ranking_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.executemany(
                "INSERT INTO games_fts(appid, name, genres) VALUES (?, ?, ?)",
                fts_rows,
            )

    def search_games(
        self,
        query: Optional[str] = None,
        genres: Optional[List[str]] = None,
        min_positive_ratio: Optional[float] = None,
        max_price: Optional[float] = None,
        size: int = 10,
    ) -> List[Dict]:
        params: List = []
        where_clauses: List[str] = []

        if min_positive_ratio is not None:
            where_clauses.append("(g.positive_ratio IS NULL OR g.positive_ratio >= ?)")
            params.append(float(min_positive_ratio))
        if max_price is not None:
            where_clauses.append("(g.price IS NULL OR g.price <= ?)")
            params.append(float(max_price))
        if genres:
            for genre in genres:
                where_clauses.append("LOWER(g.genres) LIKE ?")
                params.append(f"%{genre.strip().lower()}%")

        if query:
            sql = """
                SELECT g.appid, g.name, g.genres, g.price, g.positive_ratio,
                       g.ranking_score, (-bm25(games_fts)) AS score
                FROM games_fts
                JOIN games g ON CAST(games_fts.appid AS INTEGER) = g.appid
                WHERE games_fts MATCH ?
            """
            params = [query] + params
        else:
            sql = """
                SELECT g.appid, g.name, g.genres, g.price, g.positive_ratio,
                       g.ranking_score, COALESCE(g.ranking_score, 0) AS score
                FROM games g
            """

        if where_clauses:
            if "WHERE" in sql:
                sql += " AND " + " AND ".join(where_clauses)
            else:
                sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY score DESC LIMIT ?"
        params.append(int(size))

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()

        payload: List[Dict] = []
        for row in rows:
            genres_out = [g for g in str(row["genres"]).split(" ") if g]
            payload.append(
                {
                    "appid": int(row["appid"]),
                    "name": str(row["name"]),
                    "genres": genres_out,
                    "price": row["price"],
                    "positive_ratio": row["positive_ratio"],
                    "score": float(row["score"]) if row["score"] is not None else None,
                }
            )

        return payload

    def close(self) -> None:
        self.conn.close()
