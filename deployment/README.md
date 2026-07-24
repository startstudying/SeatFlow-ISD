# SeatFlow Docker deployment

This deployment is intentionally evidence-gated:

- `data/`, `checkpoints/`, `artifacts/`, MLflow state and database state are mounted as volumes.
- Real secrets and checkpoints are never baked into images.
- With the default `.env.example` values the stack is a degraded engineering smoke stack. It can boot, but prediction requests must fail closed until a valid real checkpoint, lineage, graph snapshot and prediction artifact are mounted.
- MySQL stores business metadata only. Versioned parquet/json snapshots and manifests remain the training fact source. Neo4j is an optional rebuildable graph projection. Redis is optional cache/task infrastructure.

## Services

| Service | Default port | Purpose |
| --- | ---: | --- |
| `fastapi` | `8000` | Read-only SeatFlow snapshot/inference contract API |
| `backend-java` | `8080` | Spring Boot orchestration API |
| `frontend` | `3000` | Next.js UI |
| `mysql` | `3306` | Business metadata |
| `mlflow` | `5000` | Experiment tracking service |
| `redis` | `6379` | Task progress/cache infrastructure |
| `neo4j` | `7474/7687` | Optional graph projection profile: `graph` |

## Quick start

```powershell
Copy-Item deployment/.env.example deployment/.env
docker compose --env-file deployment/.env -f deployment/docker-compose.yml config
docker compose --env-file deployment/.env -f deployment/docker-compose.yml up --build -d
.\scripts\e2e_smoke.ps1 -EnvFile deployment/.env
```

Optional graph projection:

```powershell
docker compose --env-file deployment/.env -f deployment/docker-compose.yml --profile graph up --build -d
```

## Health and readiness

- FastAPI: `GET http://localhost:8000/health`
- Java readiness: `GET http://localhost:8080/ready`
- Java upstream check: `GET http://localhost:8080/api/v1/upstream/fastapi/health`
- Frontend: `GET http://localhost:3000/`
- MLflow: `GET http://localhost:5000/health`

## Real artifacts

Mount real evidence without changing images:

- `data/` -> `/workspace/data:ro`
- `checkpoints/` -> `/workspace/checkpoints:ro`
- `artifacts/` -> `/workspace/artifacts:ro`
- MLflow server state -> `mlflow_data` named volume

Only treat prediction as real when the mounted artifacts satisfy the inference contract:

1. model metadata exists under `checkpoints/<model_id>/metadata.json` or `checkpoints/<model_id>.json`;
2. referenced checkpoint file exists;
3. `model_validity_status=valid`;
4. `data_scope=real`;
5. lineage JSON exists under `artifacts/lineage/<lineage_hash>.json`;
6. graph snapshot and prediction artifact exist and are also real/valid.

Otherwise the API should return 404/409 fail-closed payloads with `risk_flags`.

## Shutdown

```powershell
docker compose --env-file deployment/.env -f deployment/docker-compose.yml down
```

Add `-v` only when intentionally deleting local Docker volumes.
