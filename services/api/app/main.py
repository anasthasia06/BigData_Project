from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
	settings = get_settings()

	app.state.elastic = ElasticDB(settings.elastic_uri, settings.elastic_index)
	app.state.model = load_ranking_model(get_model_path())
	app.state.ranking_engine = build_ranking_engine(app.state.model)
	app.state.search_cache = TTLCache(default_ttl=30)
	app.state.recommend_cache = TTLCache(default_ttl=30)
	app.state.endpoint_latency = {"/search": [], "/recommend": [], "/": []}

	yield

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


app.include_router(search_router)
app.include_router(recommend_router)

