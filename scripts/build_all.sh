#!/bin/bash
set -e

# Build API Docker image
cd services/api
if [ -f pyproject.toml ]; then
  uv sync --system --no-cache
fi
docker build -t steam_api .
cd -

# Build Frontend Docker image
cd services/frontend
if [ -f pyproject.toml ]; then
  uv sync --system --no-cache
fi
docker build -t steam_frontend .
cd -

echo "Build complet des images Docker terminé."
