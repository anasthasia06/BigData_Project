

# Steam Reco System (Low Latency + Search)

## Sommaire
- [Présentation](#présentation)
- [Architecture & Choix de base de données](#architecture--choix-de-base-de-données)
- [Scripts & Pipelines](#scripts--pipelines)
- [Benchmarks & Métriques](#benchmarks--métriques)
- [Observabilité](#observabilité)
- [Tests](#tests)
- [Déploiement & CI/CD](#déploiement--cicd)


## Présentation
Système de recommandation et recherche temps réel inspiré de Steam.
Objectifs :
- Recommandations < 100ms
- Recherche + ranking intégrés
- Simulation d’architecture production (API, search, observabilité)


## Architecture & Choix de base de données
- **MongoDB** : stockage principal (jeux, features)
- **Elasticsearch** : moteur de recherche principal (full-text, filtres)
- **SQLite FTS5** : moteur de recherche local (fallback/dev, rapide, RAM)
- **Whoosh** : moteur de recherche Python pur (fallback/dev, RAM)
- **FastAPI** : API de ranking/recherche
- **Streamlit** : frontend utilisateur

**Pourquoi ces choix ?**
- **MongoDB** : insertion rapide, souple pour features, facile à dockeriser
- **Elasticsearch** : recherche full-text scalable, filtres avancés, support production
- **SQLite FTS5**/**Whoosh** : alternatives légères pour dev/tests, pas besoin de service externe

**Métriques collectées (campagne 1000 requêtes)** :
- Latence (avg, médiane, p95, max)
- Mémoire RSS avant/après
- Statut du cache RAM (search/recommend)
- Résumé des latences par endpoint
→ Voir scripts/benchmark_search.py et docs/performance.md


## Data Flow
Kaggle → Préprocessing → MongoDB → Pipeline features → Elasticsearch → API → UI


## Roadmap & Features de dev
Voir [docs/plan_branches.md](docs/plan_branches.md).
Features principales :
- Ingestion & nettoyage de données (Kaggle, MongoDB)
- Indexation Elasticsearch
- Recherche avancée (multi-moteurs)
- Recommandation rapide (RAM)
- Observabilité (Prometheus, Grafana)
- Benchmarks automatisés (1000 requêtes, scripts dédiés)


## Scripts & Pipelines
**Ingestion & Prétraitement :**
- `pipelines/ingestion/kaggle_ingest.py` : téléchargement Kaggle
- `pipelines/preprocessing/clean_data.py` : nettoyage CSV
- `pipelines/ingestion/mongo_insert.py` : insertion MongoDB
- `pipelines/features/build_features.py` : features avancées
- Orchestration : `scripts/run_pipeline.sh`

**Indexation & Recherche :**
- `pipelines/indexing/elastic_index.py` : indexation Elasticsearch
- `services/api/app/search/sqlite_fts.py` : moteur SQLite FTS5 (RAM)
- `services/api/app/search/whoosh_search.py` : moteur Whoosh (RAM)

**API & Frontend :**
- `services/api/app/main.py` : FastAPI
- `services/frontend/app.py` : Streamlit

**Benchmarks & Métriques :**
- `scripts/benchmark_api.py` : bench endpoints API (latence, p95, etc.)
- `scripts/benchmark_search.py` : campagne 1000 requêtes, collecte mémoire/cache/latence
- `scripts/run_campaign_1000.sh` : lance 1000 requêtes search (simulateur charge)

**Observabilité :**
- `infra/observability/` : stack Prometheus + Grafana (docker-compose)
- Dashboards Grafana : `infra/observability/grafana/dashboards/`

**Notebooks :**
- `notebooks/data_ingestion.ipynb` : exploration/ingestion

**Modèles :**
- `models/recommender.pkl` : modèle de ranking


## Endpoints API principaux
**Recherche** :
- `GET /search` (params : q, genres, min_positive_ratio, max_price, size)
	- Ex : `/search?q=action&size=10`
	- Ex avancé : `/search?q=rpg&genres=RPG,Action&min_positive_ratio=0.7&max_price=30&size=20`
**Recommandation** :
- `GET /recommend` (params : n, genre)

**Métriques Prometheus** :
- `GET /metrics` (Prometheus)
- `GET /metrics-json` (pour scripts de bench)


## Configuration
Centralisée dans `config.py`.
Stockage des données par défaut hors repo :
- `DATA_ROOT=/data/BigData_Project/data`
- Fichiers bruts : `/data/BigData_Project/data/raw`
- Fichiers traités : `/data/BigData_Project/data/processed`


## Environnement (uv recommandé)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r pipelines/ingestion/requirements.txt
```


## Exécution pipeline ingestion complète
Pré-requis :
- MongoDB local (`localhost:27017`) ou config dans `config.py`
- Credentials Kaggle dans `~/.kaggle/kaggle.json`

```bash
bash scripts/run_pipeline.sh
```


## Tests
**Unitaires & intégration :**
- `tests/pipelines_tests/` : ingestion, features, indexing, preprocessing, training
- `tests/services_tests/` : API (core, db, routes, schemas), frontend

**Exemples :**
```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/pipelines_tests/ingestion/test_kaggle_ingest.py -v
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/services_tests/api/app/routes/test_search.py -v
```


## Run Full Stack Local
```bash
docker compose -f infra/docker-compose.yml up --build
```


## Observabilité
- Stack Prometheus + Grafana (voir `infra/observability/`)
- Dashboard custom : `infra/observability/grafana/dashboards/steam-search-observability.json`
- Script de test : `scripts/test_observability.sh`
- Métriques API : latence, mémoire, cache, etc. (voir `/metrics`)

## Déploiement & CI/CD
- Docker :
	- API : `services/api/Dockerfile`
	- Frontend : `services/frontend/Dockerfile`
	- Compose : `infra/docker-compose.yml`
- Render :
	- Blueprint : `render.yaml`
	- Config env à faire sur dashboard Render
- CI/CD :
	- Workflow : `.github/workflows/deploy.yml`
	- Secrets requis : `RENDER_API_DEPLOY_HOOK`, `RENDER_FRONTEND_DEPLOY_HOOK`
	- Voir procédure complète : [docs/deployment.md](docs/deployment.md)

