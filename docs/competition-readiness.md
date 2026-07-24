# SeatFlow competition readiness

Updated: 2026-07-19

SeatFlow uses a fail-closed competition release gate. The codebase can contain
engineering implementation, sample fixtures, and smoke results while still
reporting `competition_ready=false`.

## Release gate

`scripts/package_release.ps1` is the only supported M11 packaging entry point.
It refuses to generate a release manifest unless all conditions hold:

1. `artifacts/status/phase-state.json` exists and is readable JSON.
2. `engineering_complete`, `data_ready`, `model_valid`, `backtest_valid`, and
   `competition_ready` are all `true`.
3. `data_scope` is `real`.
4. `seat_detail_scope` (or legacy `scope`) is `complete_seat`.
5. A blind manifest file exists and verifies `blind=true` and
   `label_feedback_forbidden=true`.
6. Competition contract and submission files exist.
7. If phase-state evidence hashes declare hashes for those files, the hashes
   must match exactly.

## Non-evidence

The following are not enough for release:

- screenshots of a data provider account,
- copied or exposed tokens,
- `top_inst` institutional aggregates without full brokerage seat detail,
- synthetic/sample fixtures,
- DT-SeatHGT shape smoke,
- local UI/API screenshots without real-data lineage,
- Docker Compose config without a release-ready phase state.

## Output boundary

The packaging script writes only `release-manifest.json`. It records file paths,
SHA-256 hashes, and status flags. It does not copy raw data, checkpoints,
secrets, or submission payload contents.

