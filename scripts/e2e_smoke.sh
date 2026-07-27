#!/usr/bin/env bash
# E2E smoke for the SeatFlow stack. Complements scripts/e2e_smoke.ps1.
#
# Usage: scripts/e2e_smoke.sh
# Requires: docker compose, curl, bash 4+.

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-deployment/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-deployment/.env}"
EXAMPLE_ENV="deployment/.env.example"
FASTAPI_URL="${FASTAPI_URL:-http://127.0.0.1:8000}"
JAVA_URL="${JAVA_URL:-http://127.0.0.1:8080}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:3000}"
MLFLOW_URL="${MLFLOW_URL:-http://127.0.0.1:5000}"
PROVIDER="${PROVIDER:-sample}"
TRADE_DATE="${TRADE_DATE:-2026-01-15}"
STOCK_ID="${STOCK_ID:-600000.SH}"
MODEL_ID="${MODEL_ID:-missing-checkpoint}"
PROJECT="${PROJECT:-seatflow-smoke}"

if [[ ! -f "$ENV_FILE" && -f "$EXAMPLE_ENV" ]]; then
  echo "[smoke] $ENV_FILE missing; falling back to $EXAMPLE_ENV"
  ENV_FILE="$EXAMPLE_ENV"
fi

echo "[smoke] validating compose config"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "$PROJECT" config --quiet

cleanup() {
  echo "[smoke] stopping stack"
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "$PROJECT" down -v --remove-orphans || true
}
trap cleanup EXIT

echo "[smoke] bringing stack up"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -p "$PROJECT" up -d --wait

probe() {
  local url="$1" expected="${2:-200}"
  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)
  if [[ "$code" != "$expected" ]]; then
    echo "[smoke] FAIL $url expected $expected got $code"
    return 1
  fi
  echo "[smoke] OK   $url -> $code"
}

probe_json() {
  local url="$1" jq_path="$2" expected="$3"
  local body
  body=$(curl -sS "$url")
  local actual
  actual=$(printf '%s' "$body" | python -c "import json,sys; d=json.load(sys.stdin); print($jq_path)" 2>/dev/null || echo "")
  if [[ "$actual" != "$expected" ]]; then
    echo "[smoke] FAIL $url expected $expected at $jq_path got $actual"
    return 1
  fi
  echo "[smoke] OK   $url $jq_path=$actual"
}

probe_json "$FASTAPI_URL/health"  'd["status"]'       ok
probe_json "$JAVA_URL/ready"      'd["data"]["status"]' UP
probe_json "$JAVA_URL/api/v1/upstream/fastapi/health" 'd["data"]["status"]' ok
probe "$MLFLOW_URL/health"

# Fail-closed predictions: model_id is intentionally missing so 4xx is expected.
PREDICT="$FASTAPI_URL/api/v1/providers/$PROVIDER/models/$MODEL_ID/predict"
code=$(curl -sS -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
  -d "{\"trade_date\":\"$TRADE_DATE\",\"target_stock_id\":\"$STOCK_ID\"}" \
  "$PREDICT" || true)
case "$code" in
  404|409|400) echo "[smoke] OK   $PREDICT fail-closed $code" ;;
  *) echo "[smoke] FAIL $PREDICT expected 4xx got $code"; exit 1 ;;
esac

JPREDICT="$JAVA_URL/api/v1/providers/$PROVIDER/models/$MODEL_ID/predict"
code=$(curl -sS -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' \
  -d "{\"trade_date\":\"$TRADE_DATE\",\"target_stock_id\":\"$STOCK_ID\"}" \
  "$JPREDICT" || true)
case "$code" in
  404|409|400) echo "[smoke] OK   $JPREDICT fail-closed $code" ;;
  *) echo "[smoke] FAIL $JPREDICT expected 4xx got $code"; exit 1 ;;
esac

probe "$FRONTEND_URL/"

echo "[smoke] all probes passed"
