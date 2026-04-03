#!/bin/bash
set -e

# Lint et tests pour chaque service
for svc in services/api services/frontend; do
  echo "\n--- $svc ---"
  cd $svc
  if [ -f requirements.txt ]; then
    pip install --no-cache-dir -r requirements.txt
  fi
  if [ -f pyproject.toml ]; then
    uv sync --system --no-cache
  fi
  # Lint
  if command -v pylint &> /dev/null; then
    pylint $(find . -name '*.py') || true
  fi
  if command -v flake8 &> /dev/null; then
    flake8 . || true
  fi
  # Tests unitaires
  if [ -d tests ]; then
    pytest tests || true
  fi
  cd -
done

echo "Lint et tests terminés pour tous les services."
