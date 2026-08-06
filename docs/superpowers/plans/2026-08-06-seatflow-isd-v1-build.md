# SeatFlow-ISD V1.0 Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the real Tushare snapshots through 2026-08-06 inclusive, then build a versioned SeatFlow-ISD V1.0 engineering dataset with deterministic manifests and fail-closed formal validity.

**User-specified universe clarification (2026-08-06):** This delivery is LHB-only. Every stock that appears in `dragon_tiger_summary` at least once between 2023-01-03 and 2026-08-06 belongs to the dataset universe. Retain every DIRECT sample in every event cycle from D0 through the cycle's extended D+5. Do not emit SPILLOVER samples or unrelated never-listed stocks. Market-wide daily-bar and daily-basic snapshots may be read as source coverage, but output rows must be filtered to the LHB-ever universe.

Frozen addendum: `docs/superpowers/specs/2026-08-06-seatflow-isd-v1-universe-addendum.md`.

**Architecture:** Add a read-only legacy snapshot adapter and a focused ISD orchestration layer under `research-engine/src/seatflow/dataset/`. Reuse the existing event, cycle, sample, split, materializer, and seal types after correcting their trading-calendar and return-policy mismatches. Persist JSONL/JSON artifacts atomically and expose one CLI that builds or audits a version.

**Tech Stack:** Python 3.11+, stdlib `json/hashlib/pathlib/tempfile`, existing Pydantic/dataclasses, pytest, Ruff, mypy; no new dependencies.

## Global Constraints

- The frozen V1.0 contract is the DOCX supplied by the user and the repository design at `docs/superpowers/specs/2026-08-06-seatflow-isd-v1-build-design.md`.
- Cycle extension, labels, splits, and all leakage checks use exchange trading days, never calendar-day approximations.
- Entry is t+1 open; exits are t+1/t+3/t+5 close; day 2-5 path legs are prior close to current close.
- `top_inst` rows with `identity_scope=institutional_flow_proxy` remain proxy evidence and block a complete-real-seat claim.
- Missing concept, benchmark, company-action, or tradeability evidence must not be synthesized.
- Any formal gate failure leaves `dataset_valid=false`; engineering output may still be written with blockers.
- No new dependency and no mutation of source snapshots.
- Recent-data refresh is append-only: existing dated snapshots are never overwritten, and the requested build upper bound is fixed at 2026-08-06; incomplete core sources on that date become explicit blockers.
- Final dataset rows are restricted to the LHB-ever stock universe and DIRECT event-cycle samples; no propagation/SPILLOVER rows are emitted in this version.

---

### Task 0: Append-Only Recent Data Refresh

**Files:**
- Reuse: existing Tushare ingestion/backfill entry point discovered in the repository
- Generate: missing dated snapshots under `D:/SeatFlow/data/historical_backfill/tushare_2023_2026_v2/`
- Create: `reports/seatflow-isd-source-refresh-2026-08-06.md`

**Interfaces:**
- Consumes: existing Tushare credentials from local environment without logging their values.
- Produces: append-only daily-bar, LHB summary, LHB seat-event, daily-basic, and applicable industry snapshots through 2026-08-06 inclusive.

- [ ] **Step 1: Run a dry inventory to identify missing exchange dates after 2026-07-17 and confirm no existing dated file will be overwritten.**
- [ ] **Step 2: Execute the existing ingestion path for each missing date and core dataset, retaining source version, available time, row count, and hash manifests.**
- [ ] **Step 3: Re-read every new `data.json` and `manifest.json`, recompute hashes/row counts, and verify explicit coverage through 2026-08-06.**
- [ ] **Step 4: Record per-source date coverage and any unavailable dates in the refresh report.**
- [ ] **Step 5: Do not commit generated source data; commit only the report if it contains no credential or sensitive payload.**

---

### Task 1: Deterministic Legacy Snapshot Inventory

**Files:**
- Create: `research-engine/src/seatflow/dataset/source_snapshots.py`
- Create: `research-engine/tests/dataset/test_source_snapshots.py`

**Interfaces:**
- Produces: `SnapshotRecord(dataset: str, trade_date: date, data_path: Path, manifest_path: Path, row_count: int, sha256: str)`.
- Produces: `discover_snapshot_records(root: Path, relative_dataset_path: Path) -> tuple[SnapshotRecord, ...]`.
- Produces: `load_verified_rows(record: SnapshotRecord) -> list[dict[str, object]]`.
- Produces: `build_source_inventory(root: Path, dataset_paths: Mapping[str, Path]) -> SourceInventory`.

- [ ] **Step 1: Write failing tests for date ordering, manifest row-count/hash verification, and source mutation detection.**

```python
def test_load_verified_rows_rejects_tampered_data(tmp_path: Path) -> None:
    record = write_snapshot(tmp_path, dataset="daily_bar", rows=[{"stock_id": "000001.SZ"}])
    record.data_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_verified_rows(record)
```

- [ ] **Step 2: Run `pytest research-engine/tests/dataset/test_source_snapshots.py -q` and confirm failure because the module is absent.**
- [ ] **Step 3: Implement canonical UTF-8 JSON hashing, exact row-count checks, `date=YYYY-MM-DD` parsing, and sorted immutable inventory records.**
- [ ] **Step 4: Re-run the test file and `ruff check research-engine/src/seatflow/dataset/source_snapshots.py research-engine/tests/dataset/test_source_snapshots.py`.**
- [ ] **Step 5: Commit with a Lore message recording that old snapshot manifests are treated as untrusted inputs until recomputed.**

### Task 2: Correct Trading-Day Cycle and Label Semantics

**Files:**
- Modify: `research-engine/src/seatflow/dataset/cycle.py`
- Modify: `research-engine/src/seatflow/dataset/samples.py`
- Modify: `research-engine/src/seatflow/labels/returns.py`
- Modify: `research-engine/tests/dataset/test_cycle.py`
- Modify: `research-engine/tests/dataset/test_samples.py`
- Modify: `research-engine/tests/labels/test_returns.py`

**Interfaces:**
- Change: `build_cycles(events, *, trade_calendar: Sequence[datetime], extension_days: int = 5) -> list[EventCycle]`.
- Invariant: `EventCycle.cycle_end` is the fifth exchange day after the latest event and repeated events on that boundary extend the same cycle.
- Change: `build_samples(...)` emits every exchange day from `cycle_start` through `cycle_end`, inclusive.
- Change: `compute_return_path(...)` returns path legs consistent with `Close[t+h] / Open[t+1] - 1`.

- [ ] **Step 1: Add a failing holiday/weekend cycle test whose sixth calendar day is still within five exchange days.**
- [ ] **Step 2: Add a failing repeated-event test proving a boundary event resets `cycle_end` to its own +5 exchange day.**
- [ ] **Step 3: Run the two new cycle tests and confirm the current calendar-day implementation fails.**
- [ ] **Step 4: Implement calendar-index lookup and cycle end calculation without changing deterministic cycle IDs.**
- [ ] **Step 5: Add a failing sample test expecting six target samples for D0 through D+5 and an extended window after a repeated event.**
- [ ] **Step 6: Implement inclusive sample materialization from `cycle_start` through `cycle_end`.**
- [ ] **Step 7: Add a literal price-path test: opens `[100, 120, 90, 95, 110]`, closes `[110, 90, 95, 110, 121]`; expect cumulative 1/3/5 returns `0.10`, `-0.05`, `0.21` from the first open, not compounded intraday returns.**
- [ ] **Step 8: Implement day 1 open-to-close and day 2-5 prior-close-to-close path legs; ensure compounding exactly reconstructs each cumulative horizon.**
- [ ] **Step 9: Run the three focused test files, then the whole `research-engine/tests/dataset` and `research-engine/tests/labels` suites.**
- [ ] **Step 10: Commit with a Lore message documenting rejection of calendar-day and intraday-only approximations.**

### Task 3: Source-to-Domain Adapter and Label Rows

**Files:**
- Create: `research-engine/src/seatflow/dataset/isd_adapter.py`
- Create: `research-engine/tests/dataset/test_isd_adapter.py`

**Interfaces:**
- Consumes: verified summary, seat-event, daily-bar, daily-basic, and industry snapshot rows.
- Produces: `ISDInputs(events, participations, trade_calendar, market_data, daily_basic, industry_membership, source_inventory)`.
- Produces: `build_isd_labels(samples, market_data, benchmark_data, *, materialized_at) -> tuple[MaterializedLabel, ...]`.

- [ ] **Step 1: Write a failing fixture test proving summary rows with distinct reasons on one stock/day become distinct events while matching seat rows bind by stock/day/reason.**
- [ ] **Step 2: Write a failing test proving `named_seat` and `institutional_flow_proxy` remain separately classified in adapter statistics.**
- [ ] **Step 3: Implement timezone-aware parsing in Asia/Shanghai converted to UTC, deterministic event IDs, side/rank amounts, and source-lineage fields.**
- [ ] **Step 4: Write a failing label test covering mature and `UNMATURE` tail samples without filling missing future bars with zero.**
- [ ] **Step 5: Implement raw-return labels and mark market-excess unavailable when no verified benchmark source exists; do not substitute an equal-weight market.**
- [ ] **Step 6: Run adapter tests, Ruff, and mypy for the new module.**
- [ ] **Step 7: Commit with a Lore message documenting the proxy-seat and missing-benchmark boundaries.**

### Task 4: Dataset Orchestrator, Quality Report, and Atomic Writer

**Files:**
- Create: `research-engine/src/seatflow/dataset/isd_builder.py`
- Create: `research-engine/tests/dataset/test_isd_builder.py`
- Modify: `research-engine/src/seatflow/dataset/__init__.py`

**Interfaces:**
- Produces: `ISDBuildConfig(source_root: Path, output_root: Path, dataset_version: str, materialized_at: datetime)`.
- Produces: `build_isd_dataset(config: ISDBuildConfig) -> ISDBuildResult`.
- Output contract: `bronze/source_inventory.json`, Silver/Gold/Samples JSONL, split/schema JSON, manifest, quality report, optional seal.

- [ ] **Step 1: Write a failing end-to-end fixture test asserting the exact directory contract and deterministic manifest hash across two builds.**
- [ ] **Step 2: Write failing quality tests for zero PIT violations, cycle split atomicity, label/path consistency, proxy-seat counts, missing source blockers, and `dataset_valid=false` when any formal blocker exists.**
- [ ] **Step 3: Implement orchestration using existing `build_cycles`, `build_samples`, `create_walk_forward_plan`, `materialize_dataset`, and `seal_dataset`; derive all gate booleans from computed evidence.**
- [ ] **Step 4: Implement JSONL serialization with stable key order and atomic directory promotion from a sibling temporary directory.**
- [ ] **Step 5: Ensure a failed write leaves an existing completed dataset version unchanged.**
- [ ] **Step 6: Run the builder test, all dataset tests, Ruff, mypy, and `compileall`.**
- [ ] **Step 7: Commit with a Lore message describing engineering output versus formal seal semantics.**

### Task 5: Build/Audit CLI

**Files:**
- Create: `research-engine/src/seatflow/dataset/isd_cli.py`
- Create: `research-engine/tests/cli/test_isd_cli.py`
- Modify: `research-engine/pyproject.toml`
- Modify: `research-engine/README.md`

**Interfaces:**
- Command: `seatflow-isd build --source-root PATH --output-root PATH --dataset-version VERSION --materialized-at ISO8601`.
- Command: `seatflow-isd audit --dataset-root PATH`.
- Exit codes: `0` for successful engineering build/audit, `2` for invalid arguments or corrupt inputs, `3` when `--require-formal` is requested and formal validity is blocked.

- [ ] **Step 1: Write failing CLI tests for build JSON output, corrupt input exit 2, and require-formal exit 3.**
- [ ] **Step 2: Implement argparse CLI and structured stdout containing version, path, row counts, `dataset_valid`, and blockers.**
- [ ] **Step 3: Register the console script and document the exact real-build command.**
- [ ] **Step 4: Run CLI tests and a fixture build through the installed module entry point.**
- [ ] **Step 5: Commit with a Lore message including tested commands and known formal blockers.**

### Task 6: Real Refreshed-Window Build and Verification

**Files:**
- Generate: `datasets/seatflow_isd_v1/<dataset_version>/...` (gitignored data artifact)
- Create: `reports/seatflow-isd-v1-build-<dataset_version>.md`

**Interfaces:**
- Consumes the source root `data/historical_backfill/tushare_2023_2026_v2`.
- Produces one immutable dataset version and a concise evidence report.

- [ ] **Step 1: Run source inventory audit and verify daily-bar, LHB-summary, LHB-seat, and daily-basic coverage from 2023-01-03 through 2026-08-06, recording any per-source gap as a blocker.**
- [ ] **Step 2: Execute the real CLI build with a timestamped deterministic version name.**
- [ ] **Step 3: Re-run `seatflow-isd audit` on the produced directory and recompute every file hash referenced by the manifest.**
- [ ] **Step 4: Run targeted tests, full research-engine tests, Ruff, mypy, and compileall; record exact pass/fail evidence.**
- [ ] **Step 5: Write the report with counts, date ranges, validity status, blockers, and the distinction between named seats and institutional-flow proxies.**
- [ ] **Step 6: Commit only code/tests/report; do not commit large dataset files unless the user explicitly requests publishing.**

## Self-Review Result

- Spec coverage: source lineage, cycle/sample/label policy, split atomicity, output schema, manifest, quality report, fail-closed behavior, and real build each map to a task.
- Type consistency: all new public interfaces are defined before downstream use.
- Completeness scan: every implementation step names its file, behavior, and verification command.
- Scope: model training, OOF, blind evaluation, and backtesting remain outside this plan.
