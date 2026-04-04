# Observabilité Production sur Render

## Architecture Actuelle
- **API** : FastAPI avec métriques Prometheus intégrées (`/metrics`)
- **Frontend** : Streamlit (sans métriques actuellement)
- **Local** : Prometheus + Grafana via Docker Compose

## Options pour Production sur Render

### **OPTION 1 : Stack Observabilité Dédiée sur Render** (Recommandée)
Déployer Prometheus + Grafana comme **service séparé** payant sur Render

**Avantages :**
- Contrôle total + sécurité
- Métriques centralisées pour tous les services
- Alertes natives Prometheus

**Inconvénients :**
- Coût additionnel (service payant Render)
- Gestion supplémentaire

---

### **OPTION 2 : Grafana Cloud** (Simple & Gratuit)
Utiliser Grafana Cloud (gratuit jusqu'à certaines limites)

**Avantages :**
- Gratuit (limites généreuses)
- Setup rapide
- Alerting intégré

**Inconvénients :**
- Dépendance externe
- Configuration Prometheus obligatoire côté serveur

---

### **OPTION 3 : Service Monitoring Jaeger + Prometheus Exporters** (Avancé)
Héberger Prometheus + Jaeger sur une infra légère externe

**Avantages :**
- Trace distribuée
- Métriques + Logs centralisés

**Inconvénients :**
- Complexité accrue

---

## Plan d'Action Recommandé

### Étape 1 : Préparer l'API pour Production
L'API a déjà Prometheus intégré ✓

**Vérifier l'endpoint /metrics** :
```bash
curl https://steam-api.onrender.com/metrics
```

### Étape 2 : Ajouter Métriques au Frontend Streamlit
Streamlit n'a pas Prometheus natif, ajouter une couche de proxy.

**Solution rapide** :
- Wrapper HTTP avec Prometheus middleware
- OU déployer frontend avec Gunicorn + Prometheus middleware

### Étape 3 : Déployer Prometheus en Production

**Sur Render, créer un nouveau service** :
```yaml
- type: web
  name: steam-prometheus
  env: docker
  dockerfilePath: infra/observability/Dockerfile.prometheus
  plan: starter  # Minimum $10/mois
  envVars:
    - key: PROMETHEUS_REMOTE_WRITE_URL
      value: https://prom-blocks-prod-us-central1.grafana.net/api/prom/push
```

### Étape 4 : Déployer Grafana en Production

**Option A : Grafana Cloud** (Préféré pour MVP)
1. Créer compte Grafana Cloud gratuit
2. Récupérer API key
3. Configurer Prometheus pour envoyer les métriques
4. Créer dashboards dans Grafana Cloud

**Option B : Grafana Self-hosted sur Render**
```yaml
- type: web
  name: steam-grafana
  env: docker
  dockerfilePath: infra/observability/Dockerfile.grafana
  plan: starter  # Minimum $10/mois
```

---

## Scénario Simple : Grafana Cloud + Prometheus Remote Write

### Configuration Prometheus Production

```yaml
# infra/observability/prometheus.production.yml
global:
  scrape_interval: 30s  # Plus long qu'en local pour économiser
  evaluation_interval: 30s

remote_write:
  - url: https://prometheus-blocks-prod-eu-west-0.grafana.net/api/prom/push
    basic_auth:
      username: YOUR_USER_ID
      password: YOUR_API_TOKEN

scrape_configs:
  - job_name: steam_api
    metrics_path: /metrics
    static_configs:
      - targets: ["steam-api.onrender.com"]
    scheme: https

  - job_name: steam_frontend
    metrics_path: /metrics
    static_configs:
      - targets: ["steam-frontend.onrender.com"]
    scheme: https
```

### Stack Docker Render (Prometheus seul avec Remote Write)

```yaml
# infra/observability/Dockerfile.prometheus.render
FROM prom/prometheus:v2.54.1

COPY prometheus.production.yml /etc/prometheus/prometheus.yml
COPY alerts.yml /etc/prometheus/alerts.yml

EXPOSE 9090

CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.retention.time=7d"]
```

### Render.yaml Mise à Jour

```yaml
services:
  - type: web
    name: steam-api
    env: docker
    dockerfilePath: services/api/Dockerfile
    plan: free
    healthCheckPath: /metrics
    autoDeploy: true

  - type: web
    name: steam-frontend
    env: docker
    dockerfilePath: services/frontend/Dockerfile
    plan: free
    autoDeploy: true

  # Nouveau service Prometheus
  - type: background-worker
    name: steam-prometheus
    env: docker
    dockerfilePath: infra/observability/Dockerfile.prometheus.render
    plan: starter
    envVars:
      - key: PROMETHEUS_API_URL
        value: https://prometheus-blocks-prod-eu-west-0.grafana.net/api/prom/push
      - key: PROMETHEUS_USER_ID
        value: YOUR_USER_ID
      - key: PROMETHEUS_API_TOKEN
        value: YOUR_API_TOKEN
        isSecret: true
```

---

## Coûts Estimés (Render)

| Service | Plan | Coût |
|---------|------|------|
| API | Free | $0 |
| Frontend | Free | $0 |
| Prometheus | Starter | $10/mois |
| **Total** | | **$10/mois** |

**Alternative** : Grafana Cloud gracieux = $0 (mais moins de contrôle)

---

## Étapes d'Implémentation

### 1️⃣ **Court terme (2h)** : Vérifier Prometheus sur l'API
```bash
# En production
curl https://steam-api.onrender.com/metrics

# Doit afficher les métriques Prometheus
api_requests_total{...}
api_request_latency_seconds{...}
```

### 2️⃣ **Moyen terme (4h)** : Ajouter Prometheus to Render

1. Créer `infra/observability/Dockerfile.prometheus.render`
2. Créer `prometheus.production.yml` avec remote_write
3. Ajouter le service dans `render.yaml`
4. Push et déployer

### 3️⃣ **Long terme (6h)** : Grafana Cloud + Dashboards

1. S'inscrire Grafana Cloud (gratuit)
2. Configurer Prometheus pour envoyer les métriques
3. Créer dashboards Grafana
4. Configurer alertes

---

## Résumé Comparatif

| Approche | Setup | Coût | Complexité |
|----------|-------|------|-----------|
| **Grafana Cloud** | 1h | $0 | ⭐ |
| **Prometheus Semi-prod** | 3h | $0-10 | ⭐⭐ |
| **Stack Complète Render** | 5h | $20+ | ⭐⭐⭐ |

**Recommandation pour MVP** : **Grafana Cloud + monitoring local** = Gratuit + Simple

---

## Prochaines Actions ?

1. Vérifier endpoint `/metrics` en production
2. Créer compte Grafana Cloud (freemium)
3. Configurer remote_write Prometheus
4. Tester les métriques en live

Voulez-vous que je configure une de ces solutions ? 🚀
