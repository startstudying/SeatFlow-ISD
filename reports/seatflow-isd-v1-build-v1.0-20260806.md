# SeatFlow-ISD V1.0 Build Report — v1.0-20260806

## Dataset location

- **Absolute path:** `D:\SeatFlow\datasets\seatflow_isd_v1\v1.0-20260806`
- **Build version:** `v1.0-20260806`
- **Source root:** `D:\SeatFlow\data\historical_backfill\tushare_2023_2026_v2`
- **Materialized at:** `2026-08-06T16:00:00+08:00` (`2026-08-06T08:00:00+00:00`)

## Source coverage

| Source | Path | Dates | Row count |
|---|---|---|---|
| `dragon_tiger_summary` | `processed/tushare/dragon_tiger_summary` | 870 | ~57,548 |
| `dragon_tiger_seat_event` | `processed/tushare/dragon_tiger_seat_event` | 870 | ~676,247 |
| `daily_bar` | `processed/tushare/daily_bar` | 870 | 4,075,296 |
| `daily_basic` | `r1_daily_basic/tushare/daily_basic` | 870 | 4,075,296 |
| `sw2021_l1_industry_membership` | `r1_industry_membership/...` | n/a (code-grouped) | 0 (layout mismatch) |

All four core sources cover the full 870 trading days from 2023-01-03 through
2026-08-06. The industry-membership snapshot uses a code-grouped layout that
`source_snapshots.discover_snapshot_records` does not parse; it is recorded as
absent in the quality report and will need a dedicated adapter in a follow-up
release.

## Trading-day window

- **Start date:** 2023-01-02
- **End date:** 2026-08-05
- **Trading-day count:** 870
- **Calendar coverage:** Complete (no gaps relative to the source snapshots)

## LHB stock universe

- **Universe size:** 5,311 unique stocks that appeared in any
  `dragon_tiger_summary` snapshot between 2023-01-03 and 2026-08-06
- **First-seen dates:** Recorded in `gold/lhb_stock_universe.json`
- **Membership rule:** Any stock that appears in the LHB summary even
  once is a permanent universe member for the entire build window

## Counts

| Counter | Value |
|---|---|
| Events (LHB summary × (stock, date, reason)) | 63,038 |
| Participations (named seat + proxy) | 696,549 |
| Named seats | 640,056 |
| Institutional-flow proxies | 56,493 |
| Unknown scope | 0 |
| Unbound seats (no matching event) | 0 |
| Cycles | 27,039 |
| Direct samples | 38,609 |
| SPILLOVER samples | **0** |
| Mature labels (full 5-bar future window) | 33,566 |
| UNMATURE labels (tail samples, null returns) | 5,043 |
| Market-bar rows in silver | 4,075,296 |
| Daily-basic rows in silver | 4,075,296 |
| Industry-membership rows | 0 |

## Identity-scope accounting

- `named_seat`: 640,056 — fully resolved broker branches and institutions
- `institutional_flow_proxy`: 56,493 — kept as proxy evidence, never
  relabelled as a named seat
- `unknown_scope`: 0
- `unbound_seat_count`: 0 — every seat row matched its event

## Formal validity

- **`dataset_valid`: false**
- **Formal seal emitted:** false (no `dataset_seal.json` was written)

### Formal blockers

1. `FORMAL_NO_INSTITUTIONAL_FLOW_PROXY` — 56,493 proxy seats present
2. `FORMAL_BENCHMARK_PRESENT` — verified benchmark source not provided
3. `FORMAL_CONCEPTS_PRESENT` — concept source not provided
4. `FORMAL_COMPANY_ACTIONS_PRESENT` — company-action source not provided
5. `FORMAL_TRADEABILITY_PRESENT` — tradeability source not provided

### Formal gates passed

- `FORMAL_NO_SPILLOVER` — no SPILLOVER samples were emitted
- `FORMAL_SOURCE_COVERAGE_COMPLETE` — all four core sources present and verified
- `FORMAL_PIT_VIOLATION_FREE` — no event violates `available_time >= trade_date`
- `FORMAL_NO_IMMATURE_LABEL_LEAKAGE` — 5,043 UNMATURE labels kept with null
  returns, not zero-filled

### Engineering gates passed

- `ENG_CYCLE_SAMPLE_CONSISTENCY` — cycles=27,039, direct_samples=38,609
- `ENG_UNIVERSE_MEMBERSHIP` — every sample's stock is in `lhb_stock_universe`

## Audit results

- `seatflow-isd audit --dataset-root D:\SeatFlow\datasets\seatflow_isd_v1\v1.0-20260806`
- `manifest_exists`: true
- `manifest_hash_verified`: **true**
- `hash_mismatches`: **0** (all 12 file hashes recomputed and matched)
- `file_count`: 12
- `dataset_valid`: false (matches build output)

## Engineering dataset vs. formal seal

| Aspect | Engineering output | Formal seal |
|---|---|---|
| Directory tree | ✅ Written | n/a (not emitted) |
| Bronze / Silver / Gold / Samples / Labels / Splits / Schemas / Manifests / Quality | ✅ Complete | n/a |
| `dataset_manifest.json` with full file hashes | ✅ Written | n/a |
| `quality_report.json` with all gates | ✅ Written | n/a |
| `dataset_seal.json` | ❌ Not emitted | ❌ Not emitted |
| Trainable without proxy seats | ❌ No (56,493 proxies present) | n/a |
| Trainable with equal-weight benchmark | ❌ No (no benchmark source) | n/a |

The build succeeded as an **engineering** dataset. No formal `dataset_seal.json`
was emitted because the five formal blockers listed above are unmet. Downstream
training and backtest work must not treat this output as formally sealed.

## Verification evidence

- `pytest research-engine/tests/`: **759 passed**, 3 warnings, ~42s
- `pytest research-engine/tests/dataset/ research-engine/tests/labels/`: 241 passed
- `pytest research-engine/tests/dataset/test_isd_adapter.py`: 14 passed
- `pytest research-engine/tests/dataset/test_isd_builder.py`: 6 passed
- `pytest research-engine/tests/cli/test_isd_cli.py`: 6 passed
- `pytest research-engine/tests/dataset/test_cycle.py`: 25 passed
- `pytest research-engine/tests/dataset/test_samples.py`: 14 passed
- `pytest research-engine/tests/labels/test_returns.py`: 16 passed
- `ruff check research-engine/src/seatflow/dataset/ research-engine/tests/dataset/ research-engine/tests/cli/`: clean
- `mypy research-engine/src/seatflow/dataset/isd_builder.py isd_adapter.py isd_cli.py source_snapshots.py cycle.py samples.py`: clean (0 errors)
- `mypy research-engine/src/seatflow/labels/returns.py`: 8 pre-existing errors
  (unchanged from `467de91`; not introduced by this handoff)
- `python -m compileall research-engine/src/seatflow/`: clean
- `git diff --check`: clean

## Output contract

| Layer | File | Size |
|---|---|---|
| bronze | `source_inventory.json` | 435 B |
| silver | `events.jsonl` | 26.7 MB |
| silver | `participations.jsonl` | 300.6 MB |
| silver | `market_bars.jsonl` | 1.07 GB |
| silver | `daily_basic.jsonl` | 1.35 GB |
| gold | `lhb_stock_universe.json` | 266 KB |
| gold | `cycles.jsonl` | 6.6 MB |
| samples | `direct_samples.jsonl` | 14.7 MB |
| labels | `label_rows.jsonl` | 27.2 MB |
| splits | `walk_forward_plan.json` | 59 B |
| schemas | `dataset_schema.json` | 673 B |
| manifests | `dataset_manifest.json` | 2.6 KB |
| quality_reports | `quality_report.json` | 2.6 KB |

Total ≈ 2.8 GB across 12 files. No `dataset_seal.json` emitted.

## Known limitations / residual risk

1. **Industry membership not loaded.** The `r1_industry_membership` snapshot
   uses a code-grouped layout (`l1_code=801010.SI/is_new=N/...`) that the
   date-based `discover_snapshot_records` does not parse. `industry_membership`
   row count is 0 and the silver `industry_membership.jsonl` was not written.
2. **Benchmark / concepts / company actions / tradeability** are not yet
   sourced. They are recorded as formal blockers; their absence does not break
   the engineering build.
3. **56,493 institutional-flow-proxy seats** are present. These are real
   proxy rows from the Tushare source and are correctly classified, but
   their presence blocks the formal "complete real seat graph" claim.
4. **The build took ~50 minutes** on this hardware. Hot paths inside the
   builder were optimized (pre-indexed `events_by_stock` and `bars_by_stock`)
   after the first attempt showed O(n²) scans.
5. **Atomic directory promotion** has a Windows-specific fallback to
   `shutil.move` when `os.rename` fails with `PermissionError` (large
   directory or AV/indexer hold). The fallback is exercised by the real build.

## Reproducibility

To reproduce this build from the same source root:

```bash
cd "C:\Users\wonderful\.codex\worktrees\52dc\SeatFlow"
PYTHONPATH=research-engine/src python -m seatflow.dataset.isd_cli build \
  --source-root "D:\SeatFlow\data\historical_backfill\tushare_2023_2026_v2" \
  --output-root "D:\SeatFlow\datasets\seatflow_isd_v1" \
  --dataset-version "v1.0-20260806" \
  --materialized-at "2026-08-06T16:00:00+08:00"

PYTHONPATH=research-engine/src python -m seatflow.dataset.isd_cli audit \
  --dataset-root "D:\SeatFlow\datasets\seatflow_isd_v1\v1.0-20260806"
```

## Git commits

Each task was developed against the `codex/seatflow-isd-handoff` branch.
This handoff produced the following uncommitted changes (all scoped to the
SeatFlow-ISD V1.0 build):

- `research-engine/src/seatflow/dataset/isd_adapter.py` (new)
- `research-engine/src/seatflow/dataset/isd_builder.py` (new)
- `research-engine/src/seatflow/dataset/isd_cli.py` (new)
- `research-engine/src/seatflow/dataset/__init__.py` (modified)
- `research-engine/src/seatflow/dataset/cycle.py` (modified: calendar-boundary
  clamp so the last few trading days do not raise)
- `research-engine/src/seatflow/dataset/source_snapshots.py` (modified:
  accept `trade_date` or `date` in manifests, and add `discover_and_load_records`
  to avoid doubled I/O)
- `research-engine/pyproject.toml` (modified: register `seatflow-isd` console
  script)
- `research-engine/README.md` (modified: CLI usage docs)
- `research-engine/tests/dataset/test_isd_adapter.py` (new)
- `research-engine/tests/dataset/test_isd_builder.py` (new)
- `research-engine/tests/cli/test_isd_cli.py` (new)
- `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/progress.md` (updated)
- `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-2-review.md` (new)
- `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-3-review.md` (new)
- `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-4-review.md` (new)
- `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-5-review.md` (new)
- `reports/seatflow-isd-v1-build-v1.0-20260806.md` (this file)

The large generated dataset under `D:\SeatFlow\datasets\seatflow_isd_v1\` is
gitignored and was not committed.
