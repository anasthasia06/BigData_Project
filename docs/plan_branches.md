# Plan détaillé par branche de développement

## feat/data-ingestion
1. Télécharger le dataset Kaggle (script ou notebook)
2. Nettoyer les données (pipeline de nettoyage)
3. Stocker les données dans MongoDB (script d’insertion)
4. Documenter le process et les scripts

## feat/features-model
1. Créer des features simples (feature engineering)
2. Entraîner un modèle baseline (collaboratif ou simple ML)
3. Sauvegarder le modèle (pickle, joblib, etc.)
4. Ajouter des tests unitaires sur le pipeline

## feat/elasticsearch
1. Créer l’index Elasticsearch (mapping, settings)
2. Indexer les jeux nettoyés
3. Tester la recherche (requêtes simples et avancées)
4. Documenter les endpoints de recherche

## feat/api
1. Créer l’API FastAPI (structure, config)
2. Implémenter endpoint /search (requête ES)
3. Implémenter endpoint /recommend (ranking/model)
4. Charger le modèle en RAM au démarrage
5. Ajouter des tests unitaires sur l’API

## feat/frontend
1. Développer l’UI Streamlit (recherche, reco)
2. Connecter l’API (requêtes HTTP)
3. Tester l’UX (retours, affichage)
4. Ajouter des tests d’intégration si possible

## feat/performance
1. Mesurer la latence des endpoints
2. Ajouter un cache RAM (ex: simple dict ou Redis)
3. Optimiser le ranking/model pour la prod
4. Benchmarks et reporting

## feat/deployment
1. Dockeriser API et frontend (si modif)
2. Déployer sur Render (vérifier build)
3. Connecter la CI/CD (vérifier auto-deploy)
4. Documenter la procédure de déploiement
