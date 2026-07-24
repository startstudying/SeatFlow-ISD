# SeatFlow backtest report

Updated: 2026-07-19

This is the release-facing backtest report template and current fail-closed
status. It must not be interpreted as a completed real backtest report until the
phase-state artifact marks `backtest_valid=true` and includes matching evidence
hashes.

## Current status

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| Backtest engine implemented | Implemented | Unit-tested engineering path. |
| Real OOS backtest completed | Not claimed | Requires verified blind manifest and real data lineage. |
| Results usable for competition narrative | Not claimed | Requires `backtest_valid=true` and `competition_ready=true`. |

## Required backtest contract

The real OOS backtest must include:

- frozen blind manifest hash,
- prediction artifact hash,
- graph/data snapshot hash,
- benchmark return source hash,
- trading calendar hash,
- fees, slippage, limit-up/limit-down handling,
- availability timestamps proving no look-ahead,
- raw daily NAV rows retained outside the release manifest.

## Minimum report fields for release

| Field | Requirement |
| --- | --- |
| `run_id` | Stable identifier tied to model lineage. |
| `mode` | Must be `real`, not `sample` or `synthetic`. |
| `blind_manifest_verified` | Must be true. |
| `data_scope` | Must be `real`. |
| `seat_detail_scope` | Must be `complete_seat`. |
| `claims.real_backtest_valid` | Must be true. |

## Current blockers

- No verified complete-seat historical source is present.
- No frozen official blind/competition contract is present.
- No release-grade OOS report is present.

