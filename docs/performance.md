# Performance - feat/performance

## Objectifs couverts
1. Mesure de latence des endpoints
2. Cache RAM sur /search et /recommend
3. Optimisation du ranking en RAM
4. Benchmarks et reporting

## Mesure de latence en runtime
L'API ajoute le header `X-Process-Time-Ms` pour chaque requete.
Les latences par endpoint sont aussi historisees en memoire dans `app.state.endpoint_latency`.

## Cache RAM
- `search_cache`: TTL 30s
- `recommend_cache`: TTL 30s
- Cache key basee sur les parametres de requete.

## Optimisation ranking/model
Le modele est charge au demarrage, puis un `RankingEngine` est construit en RAM avec index par genre pour accelerer `/recommend`.

## Benchmark local
Executer:

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/benchmark_api.py --base-url http://localhost:8000 --iterations 50
```

Le script genere un rapport JSON dans `docs/performance_report.json` avec:
- `avg_ms`
- `p50_ms`
- `p95_ms`
- `max_ms`

## Interprétation
- `p50_ms`: latence mediane
- `p95_ms`: latence queue de distribution (plus proche UX worst-case)
- comparez `p95_ms` avant/apres optimisation/cache
