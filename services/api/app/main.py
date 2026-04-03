from contextlib import asynccontextmanager
import logging
import os
from time import perf_counter

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from services.api.app.core.cache import TTLCache
from services.api.app.core.config import get_model_path, get_settings
from services.api.app.core.ranking import build_ranking_engine, load_ranking_model
from services.api.app.db.elastic import ElasticDB
from services.api.app.routes.recommend import router as recommend_router
from services.api.app.routes.search import router as search_router
from services.api.app.search.sqlite_fts import SQLiteFTSSearchEngine


def _build_minimal_fallback_model() -> dict:
	return {
		"model_type": "bootstrap_minimal",
		"created_at": "runtime-fallback",
		"top_k": 0,
		"ranking": [],
	}


@asynccontextmanager
async def lifespan(app: FastAPI):
	settings = get_settings()

	app.state.elastic = ElasticDB(settings.elastic_uri, settings.elastic_index)
	model_path = get_model_path()
	try:
		app.state.model = load_ranking_model(model_path)
	except FileNotFoundError:
		logging.warning("Modele introuvable au demarrage (%s). Fallback minimal active.", model_path)
		app.state.model = _build_minimal_fallback_model()

	if settings.search_backend == "duckdb":
		from services.api.app.search.duckdb_search import DuckDBSearchEngine

		app.state.search_engine = DuckDBSearchEngine(app.state.model)
	elif settings.search_backend == "sqlite_fts5":
		app.state.search_engine = SQLiteFTSSearchEngine(app.state.model)
	else:
		app.state.search_engine = app.state.elastic
	app.state.ranking_engine = build_ranking_engine(app.state.model)
	app.state.search_cache = TTLCache(default_ttl=30)
	app.state.recommend_cache = TTLCache(default_ttl=30)
	app.state.endpoint_latency = {"/search": [], "/recommend": [], "/": []}

	yield

	if hasattr(app.state, "search_engine") and hasattr(app.state.search_engine, "close"):
		app.state.search_engine.close()
	if getattr(app.state, "search_engine", None) is not app.state.elastic:
		app.state.elastic.close()


app = FastAPI(title="Steam Reco API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
	start = perf_counter()
	response: Response = await call_next(request)
	process_ms = (perf_counter() - start) * 1000
	response.headers["X-Process-Time-Ms"] = f"{process_ms:.3f}"

	latency_store = getattr(request.app.state, "endpoint_latency", None)
	if isinstance(latency_store, dict):
		path = request.url.path
		latency_store.setdefault(path, []).append(process_ms)
	return response


@app.get("/")
def healthcheck():
	return {"status": "ok", "service": "steam-api"}


@app.get("/metrics-json")
def metrics_json(request: Request):
	search_cache = getattr(request.app.state, "search_cache", None)
	recommend_cache = getattr(request.app.state, "recommend_cache", None)

	latency = getattr(request.app.state, "endpoint_latency", {})
	latency_summary = {
		path: {
			"count": len(values),
			"avg_ms": (sum(values) / len(values)) if values else 0.0,
		}
		for path, values in latency.items()
	}

	# Standard Linux metric available in container, avoids extra dependency.
	rss_bytes = None
	try:
		with open("/proc/self/status", "r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("VmRSS:"):
					parts = line.split()
					if len(parts) >= 2:
						rss_bytes = int(parts[1]) * 1024
					break
	except OSError:
		rss_bytes = None

	return {
		"pid": os.getpid(),
		"memory_rss_bytes": rss_bytes,
		"search_cache": search_cache.stats() if search_cache else {},
		"recommend_cache": recommend_cache.stats() if recommend_cache else {},
		"latency": latency_summary,
	}


app.include_router(search_router)
app.include_router(recommend_router)

