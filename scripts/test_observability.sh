#!/bin/bash

# Script de test complet de la stack d'observabilité
# Usage: ./test_observability.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$PROJECT_ROOT/infra"
OBSERVABILITY_DIR="$INFRA_DIR/observability"

echo "=========================================="
echo "TEST STACK D'OBSERVABILITÉ"
echo "=========================================="

# ============ ÉTAPE 1 : Démarrage =============
echo ""
echo "ÉTAPE 1 : Démarrage de la stack d'observabilité"
echo "-------------------------------------------"
cd "$OBSERVABILITY_DIR"
docker compose -f docker-compose.observability.yml up -d
echo "✓ Stack démarrée (Prometheus + Grafana)"
sleep 5

# ============ ÉTAPE 2 : Vérification accès =============
echo ""
echo "ÉTAPE 2 : Vérification de l'accès aux serveurs"
echo "-------------------------------------------"

# Test Prometheus
echo -n "Prometheus (http://localhost:9090)... "
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
  echo "✓ ACCESSIBLE"
else
  echo "✗ INACCESSIBLE"
  exit 1
fi

# Test Grafana
echo -n "Grafana (http://localhost:3000)... "
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
  echo "✓ ACCESSIBLE"
else
  echo "✗ INACCESSIBLE"
  exit 1
fi

# ============ ÉTAPE 3 : Vérification Prometheus =============
echo ""
echo "ÉTAPE 3 : Validation configuration Prometheus"
echo "-------------------------------------------"

# Récupérer le statut des targets
TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets' 2>/dev/null || echo "[]")
TARGET_COUNT=$(echo "$TARGETS" | jq 'length' 2>/dev/null || echo "0")

echo "Nombre de targets : $TARGET_COUNT"
echo "$TARGETS" | jq '.[] | {job_name: .labels.job, instance: .labels.instance, state: .health}' 2>/dev/null || true

# ============ ÉTAPE 4 : Vérification métriques =============
echo ""
echo "ÉTAPE 4 : Vérification de la collecte de métriques"
echo "-------------------------------------------"

# Chercher les métriques du job steam_api
METRICS=$(curl -s 'http://localhost:9090/api/v1/query?query=up{job="steam_api"}' | jq '.data.result' 2>/dev/null || echo "[]")

if [ "$METRICS" != "[]" ] && [ "$METRICS" != "null" ]; then
  echo "✓ Métriques trouvées pour steam_api"
  echo "$METRICS" | jq '.[] | {job: .metric.job, instance: .metric.instance, value: .value[1]}' || true
else
  echo "⚠ Aucune métrique trouvée pour steam_api"
  echo "Conseil : Assurez-vous que l'API est en cours d'exécution sur le port 8000"
fi

# ============ ÉTAPE 5 : Vérification Grafana =============
echo ""
echo "ÉTAPE 5 : Vérification des dashboards Grafana"
echo "-------------------------------------------"

# Chercher les dashboards
DASHBOARDS=$(curl -s http://localhost:3000/api/search?query=* -H "Authorization: Bearer admin:admin" 2>/dev/null || echo "[]")
DASHBOARD_COUNT=$(echo "$DASHBOARDS" | jq 'length' 2>/dev/null || echo "0")

echo "Nombre de dashboards : $DASHBOARD_COUNT"
if [ "$DASHBOARD_COUNT" -gt 0 ]; then
  echo "$DASHBOARDS" | jq '.[] | {title: .title, type: .type}' || true
  echo "✓ Dashboards trouvés"
else
  echo "⚠ Aucun dashboard trouvé"
fi

# ============ ÉTAPE 6 : Liens d'accès =============
echo ""
echo "=========================================="
echo "RÉSUMÉ ET LIENS D'ACCÈS"
echo "=========================================="
echo ""
echo "Prometheus    : http://localhost:9090"
echo "Grafana       : http://localhost:3000 (user: admin, pass: admin)"
echo "API           : http://localhost:8000"
echo ""
echo "Commandes utiles :"
echo "  Logs:         docker-compose -f $OBSERVABILITY_DIR/docker-compose.observability.yml logs -f"
echo "  Arrêt:        docker-compose -f $OBSERVABILITY_DIR/docker-compose.observability.yml down"
echo "  Redémarrage:  docker-compose -f $OBSERVABILITY_DIR/docker-compose.observability.yml restart"
echo ""
