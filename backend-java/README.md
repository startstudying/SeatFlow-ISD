# SeatFlow Java Backend

Spring Boot 3 / Java 21 business orchestration layer for local/internal SeatFlow research workflows.

## Boundary

- Java does not train models, run inference, or synthesize predictions.
- Prediction, run, model, and backtest reads are proxied to the FastAPI contract.
- Missing FastAPI, missing checkpoint, non-real data scope, or invalid model lineage is returned as fail-closed API state.
- Authentication is currently declared as `local-internal`; this is not a multi-tenant production auth implementation.

## Key endpoints

- `GET /health`, `GET /api/v1/health`
- `GET /ready`, `GET /api/v1/readiness`
- `GET /api/v1/audit/status`
- `GET /api/v1/upstream/fastapi/health`
- `GET /api/v1/models/{modelId}`
- `GET /api/v1/runs/{runId}`
- `GET /api/v1/experiments/runs/{runId}`
- `GET /api/v1/backtests/{runId}`
- `POST /api/v1/providers/{provider}/models/{modelId}/predict`
- `POST /api/v1/predictions/query`

### 数据中心与系统设置

- `GET /api/v1/data-center/overview`
- `GET /api/v1/data-assets`, `GET /api/v1/data-quality/reports`
- `GET /api/v1/graph/snapshots`, `GET /api/v1/datasets/versions`
- `GET|POST /api/v1/sync/tasks`, `POST /api/v1/sync/tasks/{id}/cancel`
- `GET|PUT /api/v1/settings/{key}`, `GET|PUT /api/v1/data-sources/{id}`
- `GET|PUT /api/v1/models/{id}`, `GET|PUT /api/v1/experiments/{id}`
- `GET /api/v1/monitoring/services`, `GET|PUT /api/v1/users/{id}`
- `GET|PUT /api/v1/roles/{code}`, `GET|PUT /api/v1/permissions/{code}`
- `GET /api/v1/audit-logs`

Interactive OpenAPI documentation is available at `/swagger-ui/index.html`.

All write routes require an `Idempotency-Key`. Settings writes additionally carry
`expectedVersion`; data source API keys are AES-GCM encrypted and only returned
as a masked status.

## Configuration

- `SEATFLOW_JAVA_PORT` defaults to `8080`
- `SEATFLOW_FASTAPI_BASE_URL` defaults to `http://localhost:8000`
- `SEATFLOW_FASTAPI_CONNECT_TIMEOUT` defaults to `2s`
- `SEATFLOW_FASTAPI_READ_TIMEOUT` defaults to `5s`
- `SEATFLOW_REDIS_HOST` / `SEATFLOW_REDIS_PORT` configure task progress cache;
  MySQL remains the source of truth if Redis is unavailable.
- Set `SEATFLOW_NEO4J_ENABLED=true` only when the Neo4j graph profile is running.
  Published snapshots are projected as immutable snapshot/node/relation records;
  a projection failure fails the graph refresh task instead of reporting success.

## Verification

Run from `backend-java/`:

```powershell
mvn test
mvn verify
```
