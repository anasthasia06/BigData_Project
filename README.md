
# Steam Reco System (Low Latency + Search)

## Overview
End-to-end recommendation system inspired by Steam.

Goals:
- Serve recommendations under 100ms
- Integrate search + ranking
- Simulate production architecture

## Architecture
- MongoDB: storage
- Elasticsearch: search
- FastAPI: ranking API
- Streamlit: frontend

## Data Flow
Kaggle -> Preprocessing -> MongoDB -> Feature pipeline -> Elasticsearch -> API -> UI

## Branch Roadmap
See [docs/plan_branches.md](docs/plan_branches.md).

## feat/data-ingestion Status
Done:
- Step 1: Kaggle download script
- Step 2: cleaning pipeline
- Step 3: MongoDB insertion pipeline
- Step 4: documentation and runnable scripts

## feat/elasticsearch Status
Done:
- Step 1: index creation script with settings/mapping
- Step 2: cleaned games bulk indexing
- Step 3: simple and advanced search smoke tests
- Step 4: search endpoint documentation

## Data-Ingestion Scripts
- Download raw dataset: `pipelines/ingestion/kaggle_ingest.py`
- Clean CSV data: `pipelines/preprocessing/clean_data.py`
- Insert cleaned data into MongoDB: `pipelines/ingestion/mongo_insert.py`
- Orchestrator script: `scripts/run_pipeline.sh`

## Elasticsearch Scripts
- Build index + mapping + bulk indexing + smoke tests: `pipelines/indexing/elastic_index.py`

Run Elasticsearch pipeline:
```bash
PYTHONPATH=$(pwd) .venv/bin/python pipelines/indexing/elastic_index.py
```

## Search Endpoints (API)
Target endpoint for search integration in API branch:

`GET /search`

Query parameters (planned):
- `q`: free-text query on game name/genres
- `genres`: optional comma-separated genre filters
- `min_positive_ratio`: optional float filter (>=)
- `max_price`: optional float filter (<=)
- `size`: max number of hits (default 10)

Example simple search:
```http
GET /search?q=action&size=10
```

Example advanced search:
```http
GET /search?q=rpg&genres=RPG,Action&min_positive_ratio=0.7&max_price=30&size=20
```

## Configuration
Centralized in `config.py`.
Default data storage is outside repo to avoid disk pressure:
- `DATA_ROOT=/data/BigData_Project/data`
- Raw files: `/data/BigData_Project/data/raw`
- Processed files: `/data/BigData_Project/data/processed`

## Environment (uv)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r pipelines/ingestion/requirements.txt
```

## Run Data-Ingestion End-to-End
Prerequisites:
- MongoDB running locally on `localhost:27017` (or update `config.py`)
- Kaggle credentials available in `~/.kaggle/kaggle.json`

```bash
bash scripts/run_pipeline.sh
```

## Run Tests (ingestion/preprocessing)
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest \
	tests/pipelines_tests/ingestion/test_kaggle_ingest.py \
	tests/pipelines_tests/ingestion/test_mongo_insert.py \
	tests/pipelines_tests/preprocessing/test_clean_data.py -v
```

## Run Tests (elasticsearch)
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest \
  tests/pipelines_tests/indexing/test_elastic_index.py -v
```

## Run Full Stack Locally
```bash
docker compose -f infra/docker-compose.yml up --build
```

## Deployment

### Docker
- API image: services/api/Dockerfile
- Frontend image: services/frontend/Dockerfile
- Compose local: infra/docker-compose.yml

### Render
- Blueprint file: render.yaml
- Configure env vars in Render dashboard after import
- Validate API healthcheck on /

### CI/CD
- Workflow: .github/workflows/deploy.yml
- On PR to main: tests
- On push to main: tests + docker build + optional Render auto-deploy hooks

Required GitHub Secrets for auto-deploy:
- RENDER_API_DEPLOY_HOOK
- RENDER_FRONTEND_DEPLOY_HOOK

### Full procedure
See docs/deployment.md

