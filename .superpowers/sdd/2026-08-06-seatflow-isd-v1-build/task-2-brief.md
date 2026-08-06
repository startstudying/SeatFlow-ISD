# Task 2: Correct Trading-Day Cycle and Label Semantics

## Global constraints

- No new dependencies.
- Follow the frozen SeatFlow-ISD V1.0 policy exactly.
- Exchange trading days govern cycle extension, active samples, and horizons.
- Entry price is t+1 open. Horizon exits are t+1/t+3/t+5 close.
- Day 1 path leg is entry open to day-1 close; day 2-5 legs are previous close to current close.
- Existing unrelated behavior must remain covered and compatible where it does not contradict the frozen contract.
- Use strict TDD and commit with Lore trailers.

## Owned files

- Modify `research-engine/src/seatflow/dataset/cycle.py`.
- Modify `research-engine/src/seatflow/dataset/samples.py`.
- Modify `research-engine/src/seatflow/labels/returns.py`.
- Modify `research-engine/tests/dataset/test_cycle.py`.
- Modify `research-engine/tests/dataset/test_samples.py`.
- Modify `research-engine/tests/labels/test_returns.py`.

## Required behavior

### Cycle

- Public API: `build_cycles(events, *, trade_calendar: Sequence[datetime], extension_days: int = 5) -> list[EventCycle]`.
- A first event sets `cycle_end` to its fifth later exchange trading day (D+5, so D0 through D+5 is six active sample dates when all exist).
- A later event with `trade_date <= current cycle_end` stays in the same cycle and resets `cycle_end` to that event's D+5.
- A later event after `cycle_end` starts a new cycle.
- An event exactly on `cycle_end` remains in the same cycle.
- Holidays/weekends never consume extension days.
- `cycle_id` remains deterministic from stock and first event date.

### Samples

- `build_samples` emits DIRECT samples for every exchange date from `cycle_start` through `cycle_end`, inclusive.
- Repeated events therefore extend the sample window.
- `days_since_last_event` uses exchange-day distance.
- `cutoff_time`/`available_time` must stay timezone-aware and respect event availability; do not introduce future event data into earlier samples.

### Returns

- For bars with first open `O1` and closes `C1..C5`, cumulative horizons are `C1/O1-1`, `C3/O1-1`, `C5/O1-1`.
- Return path legs are `C1/O1-1`, then `C2/C1-1`, `C3/C2-1`, `C4/C3-1`, `C5/C4-1`.
- Compounding the first 1/3/5 path legs must exactly reconstruct cumulative horizons within floating tolerance.
- Missing bars must not silently become flat zero-return labels. Preserve the current public dataclass if possible, but raise a clear error for an incomplete 5-day path.
- Market excess uses the same path convention for stock and verified benchmark.

## Required tests

1. Weekend/holiday gap remains one cycle when within five exchange days.
2. Repeated event on the D+5 boundary resets `cycle_end` to its own D+5.
3. A single event emits six DIRECT sample dates D0..D+5.
4. A repeated event extends samples through the reset end date.
5. Literal prices: opens `[100,120,90,95,110]`, closes `[110,90,95,110,121]`; cumulative 1/3/5 from first open are `0.10`, `-0.05`, `0.21`.
6. Incomplete five-bar path raises instead of filling zeros.

## Verification

- Run each new test before production changes and record its expected failure.
- Final focused commands:
  - `$env:PYTHONPATH='research-engine/src'; python -m pytest research-engine/tests/dataset/test_cycle.py research-engine/tests/dataset/test_samples.py research-engine/tests/labels/test_returns.py -q`
  - `$env:PYTHONPATH='research-engine/src'; python -m pytest research-engine/tests/dataset research-engine/tests/labels -q`
  - `python -m ruff check` on all six owned files.
  - `git diff --check`.

## Report contract

Write `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-2-report.md` with status, red-test evidence, implementation summary, final commands/output, commit SHA, and concerns.
