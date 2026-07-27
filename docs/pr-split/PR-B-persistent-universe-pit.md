# PR-B: Persistent Universe + Point-in-Time Schema

> Branch: `feat/persistent-universe-pit`
> Scope: PIT schema, persistent candidate pool, decayed seat state, position-continuity scoring, structural turnover diagnostics. **No model code, no checkpoints, no Parquet artifacts.**

## What this PR contains

This branch collects the R4 / R4.1 fail-closed infrastructure that makes the daily scoring universe survive the moment a stock leaves the Dragon-Tiger event pool. It is the precondition for any honest backtest: without it, the model only ever scores stocks on the day they appear on the list, which inflates RankIC and hides turnover.

### Point-in-time schema (`src/seatflow/pit/schema.py`)

- `available_at`, `prediction_time`, `entry_time`, `exit_time` fields on every record
- A row is scoreable on day `t` only if `available_at <= t < exit_time`
- Future events cannot backfill into a past snapshot

### Persistent candidate pool (`src/seatflow/r4/persistent_universe.py`)

- Daily scoring universe = `event_pool(t) ∪ current_holdings(t) ∪ watchlist(t)`
- A holding that leaves the event pool stays scoreable until it is actually sold
- Watchlist admits names the strategy is still tracking (e.g. recently exited)

### Decayed seat state (`src/seatflow/r4/`, `tests/r4/test_decayed_seat_state.py`)

- Seat activity is carried forward across trading days with a **trading-day half-life**, not re-listed daily
- `tests/r4/test_half_life_decay.py` and `tests/r4/test_event_age_uses_trading_days.py` enforce that age is measured in trading days, not calendar days
- `tests/r4/test_persistent_score_no_future_fill.py` enforces no future fill

### Position-continuity scoring (`src/seatflow/strategy/persistent_topk.py`)

- `tests/strategy/test_persistent_topk_rule_b.py` — Rule B: cost-covered rebalance, not candidate-exit selling
- `tests/strategy/test_persistent_topk_structure.py` — structure-aware top-k

### Structural turnover diagnostics (`src/seatflow/evaluation/`)

- `turnover_floor.py` — exposes Rule A's structural turnover limit before any strategy change
- `structure_preflight.py` — fail-closed preflight; if structure evidence is incomplete the gate stays closed
- `r4_1_structure_gate.py` — the actual gate decision record
- `temporal_holdings.py` — holdings-aware temporal scoring

## What this PR does NOT contain

- No `dt_seat_hgt_v3.py` (that is PR-C)
- No paper-trading business tables (that is PR-D)
- No checkpoints, no `*.pt`, no `*.pkl`, no Parquet artifacts
- No `M7`–`M12` large-scale enhancements (deferred)
- No `R6` modules (deferred — no completed control experiment yet)

## Tests

```
tests/r4/test_persistent_universe_union.py
tests/r4/test_persistent_score_decay.py
tests/r4/test_persistent_score_max_age.py
tests/r4/test_persistent_score_no_future_fill.py
tests/r4/test_decayed_seat_state.py
tests/r4/test_half_life_decay.py
tests/r4/test_event_age_uses_trading_days.py
tests/r4/test_existing_holdings_always_scoreable.py
tests/strategy/test_persistent_topk_rule_b.py
tests/strategy/test_persistent_topk_structure.py
tests/evaluation/test_r4_structure_preflight.py
tests/evaluation/test_r4_1_does_not_read_labels.py
tests/evaluation/test_r4_1_evidence_audit.py
tests/evaluation/test_r4_1_structure_gate.py
```

## Merge note

The 10 commits on this branch ahead of `origin/main` are the R4/R4.1 work that
was previously bundled into the 367-file PR #4. They are isolated here so the
trunk can review the fail-closed universe infrastructure without also reviewing
model code or paper trading.
