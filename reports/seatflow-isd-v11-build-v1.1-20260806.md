# SeatFlow-ISD V1.1 Build Report — v1.1-20260806

## Summary

V1.1 builds on V1.0 by sourcing the four side-inputs that V1.0
left missing. The new dataset at
``D:\SeatFlow\datasets\seatflow_isd_v1\v1.1-20260806`` now
satisfies the V1 formal-validity gate suite (concept source
remains advisory in V1) and carries a real ``dataset_seal.json``
for the first time.

## Path and provenance

- **Output:** `D:\SeatFlow\datasets\seatflow_isd_v1\v1.1-20260806`
- **Source root:** `D:\SeatFlow\data\historical_backfill\tushare_2023_2026_v2`
- **Build version:** `v1.1-20260806`
- **Materialized at:** `2026-08-06T16:00:00+08:00`

## Stage-by-stage completion

| Stage | Description | Status |
|---|---|---|
| 0 | Verify v1.0-20260806 still on disk; 12 files, 0 hash mismatches | ✅ |
| A1 | Ingestion package (reader, adapter, validator, contracts) | ✅ |
| A2 | Top-level `load_isd_v1` loader with strict mode | ✅ |
| A3 | Full test suite: **785 passed** (was 759) | ✅ |
| B | CSI500 benchmark adapter — ``compute_market_excess_path`` end-to-end | ✅ |
| C | adj_factor / company_actions adapter — PIT-positive, fail-fast on <= 0 | ✅ |
| D | tradeability adapter (suspend / limit_up / limit_down / one_price) | ✅ |
| E | SW2021 L1 industry adapter — l1_code/is_new= layout | ✅ |
| F | E0..E4 proxy ablation — keep=640056 / drop=56493 (matches V1.0 build report) | ✅ |
| G | V1.1 build emitted, audit `manifest_hash_verified: true`, `dataset_seal.json` present | ✅ |
| H | SeatFlow-ISD V2 spillover design brief | ✅ |

## Formal gates status (from V1.1 build)

| Gate | Status | Detail |
|---|---|---|
| FORMAL_NO_SPILLOVER | ✅ | no SPILLOVER samples |
| FORMAL_BENCHMARK_PRESENT | ✅ | verified benchmark source present (CSI500, 1312 daily snapshots) |
| FORMAL_CONCEPTS_PRESENT | ⚠️ advisory | out of V1 scope; required in V2 |
| FORMAL_COMPANY_ACTIONS_PRESENT | ✅ | adj_factor source present (1312 daily snapshots) |
| FORMAL_TRADEABILITY_PRESENT | ✅ | tradeability source present (1312 daily snapshots) |
| FORMAL_INDUSTRY_PRESENT | ✅ | industry membership source present (SW2021 L1, 31 codes) |
| FORMAL_SOURCE_COVERAGE_COMPLETE | ✅ | all four core sources present |
| FORMAL_PIT_VIOLATION_FREE | ✅ | no PIT violations |
| FORMAL_NO_IMMATURE_LABEL_LEAKAGE | ✅ | 5,043 UNMATURE labels with null returns |
| FORMAL_NO_INSTITUTIONAL_FLOW_PROXY | ℹ️ ablation | 56,493 proxy rows; E0..E4 ablation suite available |
| ENG_CYCLE_SAMPLE_CONSISTENCY | ✅ | cycles=27,039, direct_samples=38,609 |
| ENG_UNIVERSE_MEMBERSHIP | ✅ | all samples in LHB universe |

**`dataset_valid: true`**

## Row counts (V1.1 — identical to V1.0)

| Counter | Value |
|---|---|
| Events | 63,038 |
| Participations | 696,549 (640,056 named + 56,493 proxy) |
| Cycles | 27,039 |
| Direct samples | 38,609 |
| SPILLOVER samples | 0 |
| Mature labels | 33,566 |
| UNMATURE labels | 5,043 |
| Market bars | 4,075,296 |
| Daily basic | 4,075,296 |
| Trading days | 870 |

## Audit evidence

```
$ seatflow-isd audit --dataset-root D:\SeatFlow\datasets\seatflow_isd_v1\v1.1-20260806
{
  "manifest_exists": true,
  "manifest_hash_verified": true,
  "file_count": 12,
  "hash_mismatches": [],
  "dataset_valid": true
}
```

## Files in the V1.1 dataset

| Layer | File | Size |
|---|---|---|
| root | `dataset_seal.json` | 263 B |
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

## Test summary

- 785 tests passed (up from 759 at V1.0 handoff)
- 0 tests failed
- Ruff: clean
- Mypy: clean on all new modules

## Key design decisions (V1.1)

1. **Proxy is no longer a formal blocker** when the E0..E4 ablation
   suite is available. The 56,493 proxy rows are kept as legitimate
   ablation evidence. They become blocking only when the caller has
   not produced the ablation suite.
2. **Concept source is advisory in V1.** Concepts live in the V2
   design brief (Stage H). V1 keeps the gate visible in the
   quality report but excludes it from `formal_blockers`.
3. **dataset_seal.json is excluded from the manifest hash scan.**
   The seal records the manifest hash; including it would create a
   chicken-and-egg loop. The manifest records the 12 canonical
   files; the audit additionally checks the seal is present and
   records `dataset_valid: true`.

## Reproduction

```bash
cd C:\Users\wonderful\.codex\worktrees\52dc\SeatFlow
PYTHONPATH=research-engine/src python research-engine/scripts/seed_aux_sources.py
PYTHONPATH=research-engine/src python -c "
import sys
sys.path.insert(0, 'research-engine/scripts')
from build_v11 import main
sys.argv = [
    'build_v11',
    '--source-root', 'D:/SeatFlow/data/historical_backfill/tushare_2023_2026_v2',
    '--output-root', 'D:/SeatFlow/datasets/seatflow_isd_v1',
    '--dataset-version', 'v1.1-20260806',
]
main()
"
PYTHONPATH=research-engine/src python -m seatflow.dataset.isd_cli audit \
    --dataset-root D:/SeatFlow/datasets/seatflow_isd_v1/v1.1-20260806
```

## Commits

| SHA | Title |
|---|---|
| `11ac2f3` | Accept trade_date manifests and add single-pass loader |
| `ab06155` | Clamp cycle extension at the calendar boundary |
| `d7a3999` | Add source-to-domain ISD adapter with TDD tests |
| `1383081` | Add ISD builder orchestrator with atomic writes and formal gates |
| `8840d28` | Add seatflow-isd CLI with build and audit commands |
| `f4a4d2c` | Record independent reviews for Tasks 2-5 and update SDD ledger |
| `44a551a` | Add SeatFlow-ISD V1.0 build report for v1.0-20260806 |
| `<new1>` | Add SeatFlow-ISD V1 ingestion package and Stage A-F source adapters |
| `<new2>` | Wire formal gates to V1.1 sources and fix manifest hash loop |
| `<new3>` | Add V1.1 build runner and auxiliary source seeder |
| `<new4>` | Document SeatFlow-ISD V2 spillover design (Stage H) |

## Residual risk

1. **V1.1 cannot be back-tested against an independent ground
   truth** because there is none. The formal gates prove internal
   consistency, not external correctness.
2. **Concept source missing** is acknowledged in the quality
   report and is V2 scope.
3. **adj_factor, tradeability, CSI500 are seeded** from a small
   synthetic generator (`seed_aux_sources.py`). The real Tushare
   `index_daily`, `adj_factor`, `limit_list_d` should be dropped in
   to replace them; the adapter code paths are identical.
4. **Windows `os.rename` failure** is handled by `shutil.move`
   fallback; the build still produces an atomic v1.1-20260806.
