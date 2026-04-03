FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONPATH=/app

# Installer les dépendances frontend
COPY services/frontend/requirements.txt /app/services/frontend/requirements.txt
RUN pip install --no-cache-dir -r /app/services/frontend/requirements.txt

# Copier tout le repo pour conserver les imports services.frontend.*
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "services/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
