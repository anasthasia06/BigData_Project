from contextlib import asynccontextmanager
import logging
import os
from time import perf_counter

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

from services.api.app.core.cache import TTLCache
from services.api.app.core.config import get_model_path, get_settings
from services.api.app.core.ranking import build_ranking_engine, load_ranking_model
from services.api.app.db.elastic import ElasticDB
from services.api.app.routes.recommend import router as recommend_router
from services.api.app.routes.search import router as search_router
from services.api.app.search.sqlite_fts import SQLiteFTSSearchEngine


REQUESTS_TOTAL = Counter(
	"api_requests_total",
	"Nombre total de requetes HTTP",
	["endpoint", "method", "status"],
)
REQUEST_LATENCY_SECONDS = Histogram(
	"api_request_latency_seconds",
	"Latence des requetes HTTP en secondes",
	["endpoint", "method", "status"],
	buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
SEARCH_CACHE_HITS = Gauge("api_search_cache_hits", "Hits observes dans le cache search")
SEARCH_CACHE_MISSES = Gauge("api_search_cache_misses", "Misses observes dans le cache search")
RECOMMEND_CACHE_HITS = Gauge("api_recommend_cache_hits", "Hits observes dans le cache recommend")
RECOMMEND_CACHE_MISSES = Gauge("api_recommend_cache_misses", "Misses observes dans le cache recommend")
CACHE_SIZE = Gauge("api_cache_size", "Taille du cache en nombre d'entrees", ["cache_name"])
MEMORY_RSS_BYTES = Gauge("api_memory_rss_bytes", "Resident Set Size en bytes")


def _read_rss_bytes() -> int:
	try:
		with open("/proc/self/status", "r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("VmRSS:"):
					parts = line.split()
					if len(parts) >= 2:
						return int(parts[1]) * 1024
	except OSError:
		return 0
	return 0


def _update_runtime_metrics(request: Request) -> None:
	search_cache = getattr(request.app.state, "search_cache", None)
	recommend_cache = getattr(request.app.state, "recommend_cache", None)

	if search_cache is not None:
		stats = search_cache.stats()
		CACHE_SIZE.labels(cache_name="search").set(stats.get("size", 0))
		SEARCH_CACHE_HITS.set(stats.get("hits", 0))
		SEARCH_CACHE_MISSES.set(stats.get("misses", 0))

	if recommend_cache is not None:
		stats = recommend_cache.stats()
		CACHE_SIZE.labels(cache_name="recommend").set(stats.get("size", 0))
		RECOMMEND_CACHE_HITS.set(stats.get("hits", 0))
		RECOMMEND_CACHE_MISSES.set(stats.get("misses", 0))

	MEMORY_RSS_BYTES.set(_read_rss_bytes())


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

	if settings.search_backend == "whoosh":
		from services.api.app.search.whoosh_search import WhooshSearchEngine

		app.state.search_engine = WhooshSearchEngine(app.state.model)
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

	status = str(response.status_code)
	endpoint = request.url.path
	method = request.method
	REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status=status).inc()
	REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method, status=status).observe(process_ms / 1000.0)

	latency_store = getattr(request.app.state, "endpoint_latency", None)
	if isinstance(latency_store, dict):
		latency_store.setdefault(endpoint, []).append(process_ms)
	return response


@app.get("/")
def healthcheck():
	return {"status": "ok", "service": "steam-api"}


@app.get("/metrics-json")
def metrics_json(request: Request):
	_update_runtime_metrics(request)
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


@app.get("/metrics")
def metrics(request: Request):
	_update_runtime_metrics(request)
	return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(search_router)
app.include_router(recommend_router)

