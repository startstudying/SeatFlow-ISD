# Task 2 Independent Code Review

Reviewer: independent (handoff agent)
Range: `3e97d8a..467de91135c8303270bf1ee1df6707dfd79bcf04`
Date: 2026-08-06

## Scope

Files reviewed:

- `research-engine/src/seatflow/dataset/cycle.py`
- `research-engine/src/seatflow/dataset/samples.py`
- `research-engine/src/seatflow/labels/returns.py`
- `research-engine/tests/dataset/test_cycle.py`
- `research-engine/tests/dataset/test_samples.py`
- `research-engine/tests/labels/test_returns.py`

## Spec Compliance Verdict: PASS

All seven spec requirements from `task-2-brief.md` are correctly implemented.

### 1. Trading-day cycle calculation — PASS

`cycle.py:121-140` — `_extension_end` uses `_trading_day_index` to look up the event date's position in the exchange calendar and returns `calendar[index + extension_days]`. Holidays and weekends are never counted because they are absent from `trade_calendar`. Verified by `test_weekend_holiday_gap_within_five_exchange_days_stays_one_cycle` (test_cycle.py:117-141).

### 2. D0 through D+5 inclusive — PASS

`cycle.py:229-232` — The first event creates a cycle with `cycle_end = _extension_end(first.trade_date, calendar, 5)`, which is the 5th exchange day after D0. Since the sample window is `[cycle_start, cycle_end]` inclusive (samples.py:196-198), this yields exactly 6 sample dates (D0, D1, D2, D3, D4, D5). Verified by `test_single_event_yields_single_active_cycle` (test_cycle.py:144-151) asserting `cycle_end == 2026-01-10` (D+5 from Jan 5) and `test_single_event_emits_direct_samples_from_d0_through_d5` (test_samples.py:129-156) asserting all 6 calendar dates produce samples.

### 3. Boundary repeated-event renewal — PASS

`cycle.py:246-252` — When `ev.trade_date <= active.cycle_end`, the event stays in the same cycle and `extend_cycle` resets `cycle_end` to `_extension_end(new_event.trade_date, calendar, extension_days)`. An event exactly on `cycle_end` satisfies `<=` and is included. Verified by `test_event_on_d5_boundary_resets_cycle_end_to_its_own_d5` (test_cycle.py:153-178) where an event on D+5 (Jan 12) resets `cycle_end` to Jan 19 (its own D+5).

### 4. t+1 open entry — PASS

`returns.py:147-148` — `entry_open, _ = _bar_open_close(window[0])` uses the open of the entry-day bar as the entry price. The entry day is the first bar matching `entry_date` in `_align_window`. Verified by `test_entry_price_is_day1_open` (test_returns.py:166-180).

### 5. t+1/t+3/t+5 close exits — PASS

`returns.py:163-165` — Exit prices are set as `exits[0]` (C1), `exits[2]` (C3), and `exits[4]` (C5), which are the closes of day 1, day 3, and day 5 respectively. Verified by `test_exit_prices_recorded` (test_returns.py:142-164).

### 6. Day 1 open-to-close, days 2-5 close-to-close — PASS

`returns.py:152-161` — The loop initializes `previous_price = entry_open` (O1). For the first bar: `close / entry_open - 1.0` = C1/O1 - 1 (open-to-close). For subsequent bars: `close / previous_close - 1.0` where `previous_price` is updated to the current close after each iteration (close-to-close). Verified by `test_path_legs_compound_to_cumulative_horizons_from_first_open` (test_returns.py:69-98) with literal prices `[100,120,90,95,110]` opens and `[110,90,95,110,121]` closes, asserting d1=0.10, cum_3d=-0.05, cum_5d=0.21.

### 7. Fail-closed on missing five-day bars — PASS

`returns.py:139-144` — When `len(window) < _HORIZON_DAYS`, a `ValueError` is raised with message `"complete 5-day return path required"`. No zero-filling occurs. Verified by `test_incomplete_five_bar_path_raises_clear_error` (test_returns.py:100-115).

### Additional spec checks

- **Market excess path** uses the same corrected convention for both stock and benchmark (`returns.py:186-211`), computing per-day excess as the difference of daily returns and cumulative excess as the difference of compounded horizons. Verified by `test_excess_is_stock_minus_market_compound` and `test_excess_zero_when_stock_equals_market`.
- **Cycle ID determinism** is preserved: `_cycle_id` hashes `stock_id|cycle_start_iso` (cycle.py:97-105), and `extend_cycle` preserves the original `cycle_id` (cycle.py:166).
- **Sample window inclusivity**: `samples.py:196-198` computes `active_days` as all calendar days where `window_start <= day <= window_end`, correctly including both endpoints.

## Code Quality Verdict: PASS

### Strengths

- Clean separation of concerns: cycle construction, sample generation, and return computation are independent modules.
- Immutable frozen dataclasses throughout (`EventCycle`, `PredictionSample`, `ReturnPath`, `MarketExcessPath`).
- Timezone-awareness is enforced via `assert_aware` at all entry points.
- Deterministic ordering: cycles are sorted by `(stock_id, cycle_start)` (cycle.py:267-270); samples are emitted in calendar order.
- Error messages are specific and actionable (e.g., `"trade_date X is not in the exchange trade_calendar"`).

### Minor Findings

1. **cycle.py:98** — `_trading_day_index` uses `calendar.index(day)` which is O(n) per call. For a large calendar (870+ trading days) with many events, this could be slow. A `dict[datetime, int]` lookup map would be O(1). Not a blocker for correctness but worth noting for the real build performance.

2. **returns.py:26-28** — The docstring contains mojibake characters (e.g., `O鈫扖` instead of `O→C`) in the LaTeX-like notation. This is a cosmetic issue from encoding conversion and does not affect runtime behavior, but should be cleaned up.

3. **returns.py:127-131** — The `compute_return_path` docstring mentions "If a return cannot be computed because a bar is missing its `open` or `close` field, the corresponding daily return is set to `0.0`" but the implementation actually raises `ValueError` for incomplete paths. The docstring is stale and contradicts the fail-closed behavior. The implementation is correct; the docstring should be updated.

4. **samples.py:150-154** — `_classify_for_sample` computes `is_backtest_eligible` by checking `len(tail) < 5` where `tail = [b for b in prices if b["date"] >= prediction_date]`. This is O(n) per sample. For the real build with thousands of samples, pre-indexing bars by date would be more efficient. Not a correctness issue.

5. **cycle.py:218-220** — `build_cycles` calls `_extension_end` which raises if the calendar doesn't have enough days after the event. For the last event near the end of the calendar, this will raise. The real build must ensure the calendar extends at least 5 trading days past the last event date. This is correct fail-closed behavior, just worth noting.

## Critical Findings: None

## Important Findings: None

## Test Coverage Assessment

- **cycle.py**: 18 tests covering single event, consecutive events, boundary events, expiry, multi-stock, determinism, atomicity, and edge cases (empty, unsorted, naive datetime).
- **samples.py**: 14 tests covering D0-D+5 window, repeated-event extension, days_since_last_event reset, tradeability classification, sample ID uniqueness, cutoff time, universe filtering, holiday gaps, and empty window.
- **returns.py**: 15 tests covering literal price paths, incomplete paths, exit prices, entry price, compound horizons, path consistency, market excess, and zero-excess identity.

All spec-required tests from `task-2-brief.md` are present and verify the correct behavior.

## Conclusion

Task 2 is **APPROVED**. The implementation correctly satisfies all seven spec requirements: exchange trading-day cycle calculation, D0-D+5 inclusive sampling, boundary repeated-event renewal, t+1 open entry, t+1/t+3/t+5 close exits, day-1 open-to-close followed by close-to-close legs, and fail-closed behavior for missing bars. The five minor findings are cosmetic or performance-related and do not block proceeding to Task 3.
