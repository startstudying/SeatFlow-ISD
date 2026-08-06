# Task 3: Verified Source-to-Domain Adapter and Mature Label Rows

## Global constraints

- No new dependencies and no source-snapshot mutation.
- Consume only rows already verified by `source_snapshots.load_verified_rows`.
- Preserve point-in-time availability using timezone-aware UTC datetimes parsed from source `available_time` values.
- Never relabel `institutional_flow_proxy` as a named seat.
- Never synthesize a benchmark, concepts, company actions, or tradeability evidence.
- This delivery is LHB-only: derive the stock universe from every stock appearing at least once in verified LHB summary rows from 2023-01-03 through 2026-08-06. Filter market and daily-basic output facts to that universe; never-listed stocks are not dataset members.
- Keep the public domain types already used by cycle/sample code where practical.
- Use strict TDD and commit with Lore trailers.

## Owned files

- Create `research-engine/src/seatflow/dataset/isd_adapter.py`.
- Create `research-engine/tests/dataset/test_isd_adapter.py`.
- Write the required `task-3-report.md` under this SDD directory.

## Required input/output behavior

- Provide immutable adapter result types for events, participations, market bars, daily-basic rows, source inventory, and adapter statistics.
- Summary identity is `(stock_id, trade_date, normalized reason)`: two different reasons on one stock/day must become two deterministic events.
- A seat-event row binds only to the event with the same stock/day/normalized event reason. Missing or ambiguous associations must be counted explicitly and never guessed.
- Deterministic IDs must be string IDs derived from source lineage and natural keys; numeric identity hashes must never enter feature values.
- Preserve `source_name`, `source_version`, source trade date, source available time, and identity scope in adapted rows.
- Classify and count `named_seat`, `institutional_flow_proxy`, and unknown scopes separately. Proxy rows remain usable engineering evidence but are a formal-validity blocker downstream.
- Build a sorted unique exchange calendar from verified daily-bar dates.
- Expose the deterministic `lhb_stock_universe`; include all appearances for its members and do not construct propagation/SPILLOVER candidates.
- Build market data in the existing `{"date", "open", "close", ...}` convention with aware UTC dates and strictly positive prices; malformed core rows fail closed.
- Preserve daily-basic and industry membership as separate PIT facts; absence is represented as absence/blocker, not a fabricated value.

## Label behavior

- Define an immutable ISD label row carrying sample id, maturity status, entry/horizon timestamps, raw 1/3/5 returns when mature, optional benchmark-excess values, and a reason when unavailable.
- Entry is the next exchange day after `sample.prediction_date`; exits are that entry day's close and the third/fifth bar closes from the entry window.
- A label is mature only when five verified stock bars exist and the fifth close is observable by `materialized_at`.
- Tail samples remain present with status `UNMATURE` and null returns; missing bars must never become zero returns.
- When no verified benchmark is supplied, raw returns may be mature but all market-excess fields stay null with an explicit unavailable reason.
- When a verified benchmark is supplied, compute excess using the same corrected path convention as the stock.
- Provide conversion/evidence sufficient for the builder to create `MaterializedLabel` maturity rows without inventing mature labels.

## Required tests

1. Distinct reasons on one stock/day create distinct events and corresponding reason-matched participation bindings.
2. `named_seat` and `institutional_flow_proxy` remain distinct in rows and adapter statistics.
3. Source `+08:00` availability parses to the correct aware UTC instant and never precedes the event trade date.
4. Market bars and calendar are deterministically sorted independent of input order.
5. A never-listed stock present in market-wide daily bars is excluded, while a stock listed once remains in the LHB universe and keeps all of its verified facts needed for every event cycle.
6. A complete literal five-bar future window produces mature raw 1/3/5 returns under the frozen entry/exit rule.
7. A tail sample without five future bars is `UNMATURE` with null returns, not zero-filled.
8. Missing benchmark leaves excess unavailable rather than substituting an equal-weight market.

## Verification

- Record at least one expected RED failure before implementation.
- Run `pytest research-engine/tests/dataset/test_isd_adapter.py -q`.
- Run `pytest research-engine/tests/dataset research-engine/tests/labels -q`.
- Run Ruff on the two owned Python files, mypy on the new source module if the repository configuration supports it, and `git diff --check`.

## Report contract

Write `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-3-report.md` with status, RED evidence, implementation summary, final commands/output, commit SHA, and concerns.
