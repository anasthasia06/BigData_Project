# Dockerfile pour la pipeline d'ingestion
FROM python:3.11-slim

WORKDIR /app

# Copie des dépendances spécifiques à l'ingestion
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copie du code source
COPY . .

# Commande par défaut
CMD ["python", "kaggle_ingest.py"]
