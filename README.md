
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

## Data-Ingestion Scripts
- Download raw dataset: `pipelines/ingestion/kaggle_ingest.py`
- Clean CSV data: `pipelines/preprocessing/clean_data.py`
- Insert cleaned data into MongoDB: `pipelines/ingestion/mongo_insert.py`
- Orchestrator script: `scripts/run_pipeline.sh`

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

## Run Full Stack Locally
```bash
docker compose -f infra/docker-compose.yml up --build
```

