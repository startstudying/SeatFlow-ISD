# SeatFlow Research Engine

The V3 research engine: dataset construction, model training, evaluation,
backtest replay, and daily inference jobs for the SeatFlow seat-stock
dynamic heterogeneous graph model.

See `research-engine/src/seatflow/` for the Python package.

## SeatFlow-ISD V1.0 CLI

Build or audit a versioned SeatFlow-ISD dataset:

```bash
# Build (engineering output always written; formal seal only when all gates pass)
seatflow-isd build \
  --source-root data/historical_backfill/tushare_2023_2026_v2 \
  --output-root datasets/seatflow_isd_v1 \
  --dataset-version v1.0-20260806 \
  --materialized-at 2026-08-06T16:00:00+08:00

# Audit (recompute every file hash referenced by the manifest)
seatflow-isd audit --dataset-root datasets/seatflow_isd_v1/v1.0-20260806

# Require formal validity (exit 3 if blocked)
seatflow-isd build --source-root ... --output-root ... \
  --dataset-version v1.0-formal --require-formal
```

Exit codes: `0` success, `2` invalid arguments / corrupt inputs, `3` formal
validity blocked when `--require-formal` is requested.
