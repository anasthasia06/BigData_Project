#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
	echo "Erreur: ${PYTHON_BIN} introuvable. Créez d'abord le venv .venv." >&2
	exit 1
fi

echo "[1/3] Téléchargement dataset Kaggle"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/pipelines/ingestion/kaggle_ingest.py"

echo "[2/3] Nettoyage des données"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/pipelines/preprocessing/clean_data.py"

echo "[3/3] Insertion MongoDB"
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/pipelines/ingestion/mongo_insert.py"

echo "Pipeline data-ingestion terminé avec succès."
