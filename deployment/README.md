# SeatFlow V3 Docker deployment

`deployment/docker-compose.yml` is a thin operational entrypoint to the
canonical stack in `infra/compose/docker-compose.yml`. Both commands resolve
to the same Compose project (`seatflow`), network, service names, build
contexts, and health checks.

The stack is fail-closed. A healthy Research Engine container proves that the
HTTP process is responding; it does not prove that a research pipeline,
dataset, checkpoint, prediction, or backtest is valid. Until a verified
pipeline adapter and artifact lineage are available, research capability and
affected Control Center operations remain `DEGRADED` or `BLOCKED`.

## Services

| Service | Default host port | Purpose |
| --- | ---: | --- |
| `research-engine` | `8000` | Canonical `/research/v1` research API |
| `control-center` | `8080` | Spring Boot governance and control API |
| `research-console` | `3000` | Next.js research console |
| `postgres` | `5432` | Control Center metadata database |
| `redis` | `6379` | Task queue and cache infrastructure |
| `minio` | `9000` / `9001` | Versioned object and artifact storage |
| `mlflow` | `5000` | Experiment tracking |
| `prometheus` | `9090` | Metrics collection |
| `grafana` | `3001` | Metrics dashboards |

## Configure

Copy the example file before starting the stack:

```powershell
Copy-Item deployment/.env.example deployment/.env
```

The values in `.env.example` are local-development placeholders only. Replace
at least these values with environment-specific secrets before any shared or
non-local deployment:

- `POSTGRES_PASSWORD`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `GRAFANA_ADMIN_PASSWORD`
- `SEATFLOW_SECURITY_JWT_SECRET`

The Compose model uses required interpolation for every value above and
refuses to render when any one is missing.

## Start

```powershell
docker compose --env-file deployment/.env -f deployment/docker-compose.yml config
docker compose --env-file deployment/.env -f deployment/docker-compose.yml up --build -d
```

## Process health smoke

```powershell
Invoke-RestMethod http://localhost:8000/research/v1/health
Invoke-RestMethod http://localhost:8080/actuator/health
Invoke-WebRequest http://localhost:3000/
```

- `GET http://localhost:8000/research/v1/health` checks Research Engine process
  availability. A `DEGRADED` response with research capability `BLOCKED` is a
  truthful running state, not a successful model-readiness claim.
- `GET http://localhost:8080/actuator/health` checks the Control Center process
  and its configured infrastructure health contributors.
- `GET http://localhost:3000/` checks the Research Console HTTP process.

Do not promote a deployment based only on these process probes. Research
readiness additionally requires verified pipeline execution, artifact lineage,
sealed evaluation evidence, and the release gates defined by the project.

## Shutdown

```powershell
docker compose --env-file deployment/.env -f deployment/docker-compose.yml down
```

Add `-v` only when intentionally deleting the named local data volumes.
