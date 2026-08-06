# SeatFlow-ISD source refresh evidence — 2026-08-06

## Outcome

The append-only refresh from the 2026-07-17 baseline through 2026-08-06 completed for all 14 exchange trading days.

The existing SeatFlow ingestion path wrote new dated snapshots only. No pre-existing dated snapshot was intentionally re-ingested or overwritten.

## Provider result

- Backfill report: `D:/SeatFlow/artifacts/seatflow_backfill_recent.json`
- Requested dates: 14
- Succeeded: 14
- Failed: 0
- Skipped: 0

## Independently verified coverage

| Source | Total dates | First date | Last date | New rows | 2026-08-06 rows |
|---|---:|---|---|---:|---:|
| processed daily_bar | 870 | 2023-01-03 | 2026-08-06 | 77,377 | 5,533 |
| raw dragon_tiger_summary | 870 | 2023-01-03 | 2026-08-06 | 1,220 | 107 |
| raw dragon_tiger_seat_event | 870 | 2023-01-03 | 2026-08-06 | 13,301 | 1,074 |
| R1 daily_basic | 870 | 2023-01-03 | 2026-08-06 | 77,377 | 5,533 |

For every new dated snapshot, verification re-read `data.json` as UTF-8 JSON, recomputed the file SHA-256, compared it with `manifest.json`, and compared the decoded row count with the manifest row count. All checks passed with zero mismatches.

## Evidence boundary

This proves source snapshot availability and integrity through 2026-08-06. It does not by itself prove SeatFlow-ISD `dataset_valid=true`: concept PIT coverage, benchmark lineage, tradeability/company-action coverage, proxy-seat classification, label maturity, cycle atomicity, and the Dataset Seal remain downstream gates.
