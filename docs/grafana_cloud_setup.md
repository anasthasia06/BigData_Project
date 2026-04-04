# Guide Grafana Cloud - Setup Gratuit

## Vue d'ensemble
**Grafana Cloud** = Monitoring et alertes **gratuites** pour les petits projets

**Coûts** : $0-50/mois (freemium)
**Setup** : 30 minutes
**Avantage** : Aucun service à payer sur Render

---

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   PRODUCTION (Render)                    │
│                                                           │
│  ┌──────────────────┐    ┌──────────────────┐           │
│  │   steam-api      │    │ steam-frontend   │           │
│  │  (expose /metrics)    │ (sans métriques) │           │
│  └──────────┬───────┘    └──────────────────┘           │
│             │                                            │
│             └─────────────────────────────────┐          │
│                                               │          │
│                                        ┌──────▼──────┐   │
│                                        │ Prometheus  │   │
│                                        │  (local)    │   │
│                                        └──────┬──────┘   │
│                                               │          │
│                                    (remote_write)        │
│                                               │          │
│              ┌────────────────────────────────┘          │
│              │                                            │
│              ▼                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │          Grafana Cloud (HTTP)                │        │
│  │  - Dashboards                               │        │
│  │  - Alertes                                  │        │
│  │  - API                                      │        │
│  └──────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## Étape 1 : Créer Compte Grafana Cloud

1. Aller à https://grafana.com/auth/sign-up/create-user
2. S'inscrire (email, mot de passe)
3. **Stack créée automatiquement** ✓

---

## Étape 2 : Obtenir les Credentials

1. Aller à https://grafana.com/my/org
2. **Datasources** → **Prometheus** → Copier `URL`
3. **API Tokens** → Créer nouveau token `role: Editor`
4. **Copier token généré** (affiché une seule fois)

**Résultat** :
- `PROMETHEUS_REMOTE_URL` : `https://prometheus-blocks-prod-us-central1.grafana.net/api/prom/push`
- `GRAFANA_USER_ID` : `123456` (numéro)
- `GRAFANA_API_TOKEN` : `glc_xxxxx...` (secret)

---

## Étape 3 : Configurer Prometheus pour Remote Write

Modifier `infra/observability/prometheus.yml` (local) :

```yaml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

remote_write:
  - url: https://prometheus-blocks-prod-us-central1.grafana.net/api/prom/push
    basic_auth:
      username: YOUR_GRAFANA_USER_ID
      password: YOUR_GRAFANA_API_TOKEN

scrape_configs:
  - job_name: steam_api
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]  # Local dev
```

---

## Étape 4 : Configuration Production sur Render

**Prometheus n'a BESOIN que de local** ! Créer un service "proxy" HTTP :

### A. Créer un service léger sur Render (optionnel)

Ou simplement : **Prometheus tourne en local, envoie à Grafana Cloud**

```bash
# En développement
cd infra/observability
docker compose up -d prometheus

# Les métriques sont collectées localement ET envoyées à Grafana Cloud
curl http://localhost:9090/api/v1/query?query=up
```

### B. Vérifier dans Grafana Cloud

1. Aller à https://grafana.com/my/dashboards
2. **Explore** → Explorer les métriques
3. Requête PromQL : `up{job="steam_api"}`
4. Vous voyez les métriques live depuis votre Prometheus local ! 🎉

---

## Étape 5 : Créer Dashboards Grafana Cloud

### Dashboard Prédéfini (Rapide)

1. Grafana Cloud → **Dashboards** → **Import**
2. Importer depuis Grafana.com
   - ID: `3662` (Prometheus 2.0)
   - ID: `1860` (Node Exporter)

### Dashboard Personnalisé

1. **Dashboards** → **Create** → **New dashboard**
2. Ajouter panels avec requêtes PromQL :

**Panel 1 : Uptime API**
```
up{job="steam_api"}
```

**Panel 2 : Requêtes par minute**
```
rate(api_requests_total[1m])
```

**Panel 3 : Latence moyenne**
```
rate(api_request_latency_seconds_sum[5m]) / rate(api_request_latency_seconds_count[5m])
```

**Panel 4 : Cache Hit Ratio**
```
api_search_cache_hits / (api_search_cache_hits + api_search_cache_misses)
```

---

## Étape 6 : Créer Alertes

### Alerte : API Down
1. **Alerting** → **New alert rule**
2. Condition : `up{job="steam_api"} < 1`
3. Duration: `1m`
4. Notification Channel : Email

### Alerte : Latence Élevée
1. **New alert rule**
2. Condition : `api_request_latency_seconds_p99 > 1`
3. Duration: `5m`

---

## Coûts Réels

| Métrique | Freemium | Payant |
|----------|----------|--------|
| Rétention | 14 jours | 1+ ans |
| Serie actives | 10k | 100k+ |
| Alertes | 5 | illimitées |
| Coût/mois | FREE | $23+ |

**Pour MVP** : Freemium suffit complètement !

---

## Architecture Final (Production)

```
Render (Gratuit)        →  Prometheus Local  →  Grafana Cloud
│                               │
├─ steam-api ($0)          Remote write ────>  (web.grafana.com)
│  ├─ endpoint /metrics                       │
│  └─ métriques Prometheus                    ├─ Dashboards
│                                             ├─ Alertes
├─ steam-frontend ($0)                        ├─ API
│
└─ Coût total : $0 ✓
```

---

## Déploiement Étapes

### Local (You)
```bash
# 1. Ajouter credentials Grafana Cloud
export GRAFANA_USER_ID=xxxxx
export GRAFANA_API_TOKEN=glc_xxxxx

# 2. Relancer Prometheus avec remote write
docker compose -f infra/observability/docker-compose.observability.yml down
docker compose -f infra/observability/docker-compose.observability.yml up -d

# 3. Vérifier les métriques arrivent
curl https://grafana.com/my/datasources  # Check Prometheus datasource
```

### Production (Render) - Pas de changement nécessaire !
Prometheus reste en local, envoie les données.

---

## Monitoring from Home

Une fois configuré, vous accédez à **tout depuis Grafana Cloud** :
- Dashboards live
- Alertes email/Slack
- API REST pour intégrations
- Aucun port à exposer en public

---

## Prochaines Étapes ?

1. ✓ Créer Grafana Cloud account
2. ✓ Récupérer credentials
3. ✓ Ajouter remote_write dans Prometheus.yml
4. ✓ Relancer stack local
5. ✓ Créer premiers dashboards
6. ✓ Ajouter alertes

**Durée totale** : ~1 heure

Envie de commencer ? 🚀
