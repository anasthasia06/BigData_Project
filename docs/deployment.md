# Deployment - feat/deployment

## Objectifs couverts
1. Dockeriser API et frontend
2. Déployer sur Render
3. Connecter CI/CD pour auto-deploy
4. Documenter la procédure

## Docker local
Build et run local avec compose:

docker compose -f infra/docker-compose.yml up --build

API: http://localhost:8000
Frontend: http://localhost:8501

## Render
Le fichier render.yaml est fourni pour deploiement declaratif.

Etapes:
1. Push du repo sur GitHub
2. Connecter le repo sur Render
3. Import Blueprint depuis render.yaml
4. Configurer les env vars sensibles (ELASTIC_URI, MODEL_PATH, etc.)
5. Valider la route healthcheck API: /

## CI/CD GitHub Actions
Workflow: .github/workflows/deploy.yml

Fonctionnement:
1. Sur pull_request vers main: tests unitaires
2. Sur push sur main: tests + build docker
3. Auto-deploy Render via deploy hooks si secrets definis

Secrets GitHub requis:
- RENDER_API_DEPLOY_HOOK
- RENDER_FRONTEND_DEPLOY_HOOK

## Verification post-deploiement
1. API: GET /
2. API search: GET /search?q=action&size=5
3. API recommend: GET /recommend?n=5
4. Frontend charge sans erreur et interroge API

## Rollback
1. Re-deployer un commit precedent depuis Render
2. Ou revert commit sur main puis push (declenche auto-deploy)
