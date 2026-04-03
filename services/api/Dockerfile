FROM python:3.11-slim
WORKDIR /app
# Installer uv
RUN pip install --no-cache-dir uv
# Copier les fichiers de dépendances
COPY requirements.txt ./
COPY pyproject.toml ./
COPY uv.lock ./
# Installer les dépendances avec uv
RUN uv sync --system --no-cache
# Copier le code
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
