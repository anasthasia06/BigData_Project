from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.api.app.core.config import get_model_path, get_settings
from services.api.app.core.ranking import load_ranking_model
from services.api.app.db.elastic import ElasticDB
from services.api.app.routes.recommend import router as recommend_router
from services.api.app.routes.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
	settings = get_settings()

	app.state.elastic = ElasticDB(settings.elastic_uri, settings.elastic_index)
	app.state.model = load_ranking_model(get_model_path())

	yield

	app.state.elastic.close()


app = FastAPI(title="Steam Reco API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def healthcheck():
	return {"status": "ok", "service": "steam-api"}


app.include_router(search_router)
app.include_router(recommend_router)

