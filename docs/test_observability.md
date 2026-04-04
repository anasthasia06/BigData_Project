# Test de la Stack d'Observabilité

Guide complet pour tester Prometheus + Grafana étape par étape.

## Prérequis

- Docker et Docker Compose installés
- Ports disponibles : 9090 (Prometheus), 3000 (Grafana), 8000 (API)

## Plan de Test

### ÉTAPE 1 : Démarrage de la Stack

**Objectif** : Vérifier que Prometheus et Grafana démarrent correctement

```bash
cd infra/observability
docker-compose -f docker-compose.observability.yml up -d
```

**Vérification** :
```bash
docker ps | grep -E "prometheus|grafana"
```

Vous devez voir deux conteneurs en cours d'exécution.

---

### ÉTAPE 2 : Vérifier l'Accès aux Serveurs

**Prometheus** (http://localhost:9090)
```bash
curl http://localhost:9090/-/healthy
# Attendu : 200 OK
```

**Grafana** (http://localhost:3000)
```bash
curl http://localhost:3000/api/health
# Attendu : {"database":"ok","version":"11.2.2"}
```

---

### ÉTAPE 3 : Valider la Configuration Prometheus

**Objectif** : Vérifier que Prometheus scrape correctement les targets

Accédez à : http://localhost:9090/targets

Vous devriez voir :
- **Job** : `steam_api`
- **Instance** : `host.docker.internal:8000`
- **State** : `Down` (si l'API n'est pas démarrée) ou `Up` (si elle l'est)

**Via API** :
```bash
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets'
```

Exemple de réponse :
```json
[
  {
    "labels": {
      "job": "steam_api",
      "instance": "host.docker.internal:8000"
    },
    "health": "down"
  }
]
```

---

### ÉTAPE 4 : Vérifier la Collecte des Métriques

**Important** : L'API doit être en cours d'exécution pour que les métriques soient collectées.

**Démarrer l'API (si nécessaire)** :
```bash
cd infra
docker-compose up -d api mongo elasticsearch
```

**Vérifier les métriques** :
```bash
# Depuis l'interface Prometheus
curl 'http://localhost:9090/api/v1/query?query=up{job="steam_api"}'
```

Exemple de réponse si les métriques sont collectées :
```json
{
  "status": "success",
  "data": {
    "resultType": "vector",
    "result": [
      {
        "metric": {
          "__name__": "up",
          "job": "steam_api",
          "instance": "host.docker.internal:8000"
        },
        "value": [1712250000, "1"]
      }
    ]
  }
}
```

**Via l'interface Prometheus** :
1. Allez à http://localhost:9090
2. Cliquez sur "Graph"
3. Entrez une requête PromQL, par exemple :
   - `up` - Statut d'up/down de tous les targets
   - `up{job="steam_api"}` - Statut du job steam_api
   - `rate(http_requests_total[5m])` - Taux de requêtes (si disponible)

---

### ÉTAPE 5 : Tester les Dashboards Grafana
a
**Accès** : http://localhost:3000
- **User** : admin
- **Password** : admin

**Vérifier les datasources** :
1. Allez dans Configuration > Data Sources
2. Vous devez voir `Prometheus` configurée sur `http://prometheus:9090`

**Vérifier les dashboards** :
1. Allez dans Dashboards > Browse
2. Cherchez "steam-search-observability"
3. Si disponible, ouvrez-le pour voir les graphiques

**Via API** :
```bash
curl http://localhost:3000/api/search?query=* \
  -H "Authorization: Bearer admin:admin"
```

---

## Script Automatisé

Un script complet est disponible pour tester tous les éléments :

```bash
./scripts/test_observability.sh
```

Ce script :
1. Démarre la stack d'observabilité
2. Vérifie l'accès à Prometheus et Grafana
3. Valide la configuration Prometheus
4. Teste la collecte des métriques
5. Vérifie les dashboards disponibles
6. Affiche un résumé avec les liens d'accès

---

## Troubleshooting

### Prometheus ne répond pas

```bash
# Vérifier les logs
docker logs steam_prometheus

# Vérifier la configuration
docker exec steam_prometheus cat /etc/prometheus/prometheus.yml

# Redémarrer
docker-compose -f infra/observability/docker-compose.observability.yml restart prometheus
```

### Aucune métrique collectée (target DOWN)

1. **Vérifier que l'API tourne** :
   ```bash
   curl http://localhost:8000/docs
   ```

2. **Vérifier la connectivité Docker** :
   ```bash
   docker exec steam_prometheus ping host.docker.internal
   ```

3. **Vérifier l'endpoint /metrics sur l'API** :
   ```bash
   curl http://localhost:8000/metrics
   ```

### Grafana n'affiche rien

1. **Vérifier la datasource Prometheus** :
   - Allez dans Configuration > Data Sources
   - Cliquez sur Prometheus
   - Cliquez "Test connection"

2. **Recréer les dashboards** si nécessaire :
   - Supprimez le volume Grafana : `docker volume rm observability_grafana_data`
   - Relancez : `docker-compose up -d`

---

## Configuration Prometheus.yml

Emplacement : `infra/observability/prometheus/prometheus.yml`

Configuration actuelle :
```yaml
global:
  scrape_interval: 5s        # Fréquence de scrape des métriques
  evaluation_interval: 5s    # Fréquence d'évaluation des alertes

scrape_configs:
  - job_name: steam_api
    metrics_path: /metrics    # Endpoint des métriques
    static_configs:
      - targets: ["host.docker.internal:8000"]  # L'API
```

**Pour ajouter d'autres services** :
```yaml
scrape_configs:
  - job_name: steam_api
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8000"]

  - job_name: frontend
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8501"]
```

---

## Fonctionnalités Avancées

### Créer une alerte Prometheus

Modifier `infra/observability/prometheus/prometheus.yml` pour ajouter des règles d'alerte.

### Créer un dashboard Grafana personnalisé

1. Allez dans Dashboards > Create New Dashboard
2. Ajoutez des panels avec des requêtes PromQL
3. Sauvegardez le dashboard (UUID généré automatiquement)

### Persister les données

Ajouter des volumes dans `docker-compose.observability.yml` :
```yaml
volumes:
  prometheus_data:
  grafana_data:

services:
  prometheus:
    volumes:
      - prometheus_data:/prometheus
  grafana:
    volumes:
      - grafana_data:/var/lib/grafana
```

---

## Arrêter la Stack

```bash
cd infra/observability
docker-compose -f docker-compose.observability.yml down

# Avec suppression des données
docker-compose -f docker-compose.observability.yml down -v
```

---

## Ressources Utiles

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
