#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
(docker rm -f steam_api >/dev/null 2>&1 || true)
docker compose -f infra/docker-compose.yml up --build -d --no-deps api
sleep 4

python scripts/benchmark_search.py \
  --base-url http://localhost:8000 \
  --requests 120 \
  --concurrency 20 \
  --q action \
  --genres RPG \
  --min-positive-ratio 0.3 \
  --max-price 46 \
  --size 10
