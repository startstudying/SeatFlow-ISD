# Task 3 Review

Reviewer: ZCode (independent)
Date: 2026-08-06
Range: `isd_adapter.py` + `test_isd_adapter.py` (new files)

## Spec compliance verdict: PASS

| Requirement | Status | Evidence |
|---|---|---|
| Timezone-aware parsing Asia/Shanghai → UTC | PASS | `_parse_aware_utc`, `_to_aware_utc_from_date` convert all timestamps to aware UTC |
| Deterministic event IDs from (stock, trade_date, reason) | PASS | `_event_id` hashes the natural key; distinct reasons yield distinct events |
| named_seat / institutional_flow_proxy / unknown separated | PASS | `IdentityScope` enum + `_classify_identity_scope` + `AdapterStats` counters |
| Seats bind by stock/day/reason | PASS | `event_lookup` dict keyed by `(stock_id, trade_date, reason)`; unbound seats counted, not guessed |
| Deterministic lhb_stock_universe | PASS | `build_lhb_stock_universe` derives frozenset from events; first-seen dates recorded |
| Never-listed stocks filtered | PASS | Market bars and daily-basic filtered by `universe_ids` before output |
| No SPILLOVER | PASS | No spillover or propagation code exists in the adapter |
| Mature labels use real five-bar window | PASS | `build_isd_labels` requires entry + 4 future bars; returns computed from real prices |
| Tail samples UNMATURE with null returns | PASS | Missing future bars → `LabelMaturity.UNMATURE`, all return fields `None` |
| Excess null when no benchmark | PASS | `excess_unavailable_reason` set; no equal-weight market substitution |

## Code quality verdict: PASS

- Clean dataclass structure with `__post_init__` validation.
- `Any` type for source rows (untyped JSON dicts) is appropriate; domain types are strongly typed.
- Deterministic sorting throughout (events, participations, bars, calendar).
- `AdapterStats` provides explicit accounting for quality reports.
- No hardcoded pass booleans.

## Findings

### Minor
1. `build_isd_labels` signature takes `benchmark_bars` but always returns null excess — this is correct behavior (no verified benchmark exists), but the parameter exists for future binding.
2. `ISDInputs` does not expose `as_dict()` — serialization is handled by the builder via `_*_payload` helpers. Acceptable for the current architecture.

### No Critical or Important findings.

## Test evidence
- `research-engine/tests/dataset/test_isd_adapter.py`: 14 passed
- `research-engine/tests/dataset/` + `research-engine/tests/labels/`: 235 passed
- Ruff: clean
- mypy: clean (0 errors)
