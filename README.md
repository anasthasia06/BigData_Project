
# 🎮 Steam Reco System (Low Latency + Search)

## Overview
End-to-end recommendation system inspired by Steam.

Goal:
- Serve recommendations < 100ms
- Integrate search + ranking
- Simulate production architecture

## Architecture
- MongoDB → storage
- Elasticsearch → search
- FastAPI → ranking API
- Streamlit → frontend

## Data Flow
Kaggle → MongoDB → Feature pipeline → Elasticsearch → API → UI

## Services

### API (FastAPI)
- /recommend → ranking
- /search → Elasticsearch query

### Frontend (Streamlit)
- search interface
- recommendations display

## Tech Stack
- Python
- FastAPI
- Streamlit
- MongoDB
- Elasticsearch
- Docker
- Render

## Latency Strategy
- precomputed features
- in-memory model
- minimal DB calls
- optional cache

## Run locally
```bash
docker-compose up

