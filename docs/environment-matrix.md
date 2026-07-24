# SeatFlow environment matrix

Updated: 2026-07-19

This matrix is the M0 engineering baseline. It documents reproducible local quality gates and
keeps optional ML/tracking dependencies separate from the dependency-free CPU smoke path. It does
not claim real Tushare data readiness, model validity, backtest validity, or competition readiness.

## Python package extras

| Extra | Purpose | Notes |
| --- | --- | --- |
| `dev` | Local quality gates: pytest, Ruff, mypy | Required for M0 verification. |
| `api` | FastAPI read-only snapshot service | Optional unless serving the API. |
| `data` | Free data adapter support | AKShare can support smoke/cross-checks only; it is not complete seat-detail evidence. |
| `ml` | NumPy, pandas, PyTorch, PyG | Optional. Missing PyTorch/PyG must not break the current CPU baseline. |
| `tracking` | MLflow experiment tracking | Optional. Later milestones must retain local lineage fallback when MLflow is unavailable. |

Recommended install for M0:

```powershell
python -m pip install -e ".[dev,api,data]"
```

Optional ML/tracking install (`seatflow[ml]` and `seatflow[tracking]`):

```powershell
python -m pip install -e ".[ml,tracking]"
```

## Version and capability matrix

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\doctor.ps1
```

The doctor command reports:

| Component | Required for M0 baseline | Required for later milestones | Expected behavior when absent |
| --- | --- | --- | --- |
| Python 3.11+ | Yes | Yes | Quality commands fail closed. |
| Ruff | Yes | Yes | `scripts\quality.ps1` fails. |
| mypy | Yes | Yes | `scripts\quality.ps1` fails. |
| FastAPI/Uvicorn | API tests and local serve | M7+ | API tests are skipped if FastAPI is absent. |
| PyTorch | No | M2+ tensor/model work | Doctor records missing optional dependency. |
| PyG | No | M2+ tensor/model work | Doctor skips PyG CPU smoke unless installed. |
| CUDA/GPU | No | Formal training performance only | CPU smoke remains valid; no GPU performance claim. |
| MLflow | No | M5+ tracking service | Later milestones must use local lineage fallback or fail closed. |
| JDK | No | M8 Java service | Doctor records missing tool. |
| Maven | No | M8 Java service | Doctor records missing tool. |
| Node/npm/Next.js | No | M9 UI | Doctor records missing tool. |
| Docker Compose | No | M10 deployment | Doctor records missing tool. |

## Reproducible M0 quality gate

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\quality.ps1
```

The quality script runs, in order:

1. `scripts\doctor.ps1`
2. `python -m unittest discover -s tests -v`
3. `python -m ruff check .`
4. `python -m mypy src tests`
5. `python -m seatflow preflight --config configs/default.toml`
6. `python -m compileall src tests`

Use `-IncludeBuild` to additionally run `python -m build` when the `build` package is installed.

## Data, secret, and artifact boundary

- Do not commit real Tushare tokens or other secrets.
- Do not commit raw/processed/features/graph datasets except `.gitkeep` placeholders.
- Do not commit checkpoints, MLflow runs, reports, logs, or generated release artifacts unless a
  later milestone explicitly creates a small reviewed manifest.
- Sample/synthetic preflight output is engineering smoke only.
