#!/bin/bash
set -e

# Génère les fichiers de dépendances pour chaque service si besoin
for svc in services/api services/frontend; do
  cd $svc
  if [ -f pyproject.toml ]; then
    uv pip compile --all-extras
    uv sync --system --no-cache
  fi
  cd -
done

echo "Fichiers de dépendances générés (requirements.txt, uv.lock) pour chaque service."
