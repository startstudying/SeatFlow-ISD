# R0 executable label V2 plan

## Scope

Add a standalone, fail-closed label V2 contract alongside the legacy binary
close-to-close label. It deliberately does not make the legacy tensor/schema
consume V2 labels until the downstream three-class contract is migrated.

## Invariants

- Entry is the next trading day's executable open; the horizon endpoint is
  T+2/T+4/T+6 for 1/3/5-day labels.
- Both prices require an actual adjustment factor, source hash, and availability
  evidence. No inferred adjusted close is permitted.
- The target is stock excess return versus a declared benchmark.
- Suspension, ST, price-limit, new-stock, and missing-data outcomes are
  explicit dispositions, never silently converted into a numeric label.
- A neutral band produces negative/neutral/positive direction classes.

## Verification

Test dates, adjustment arithmetic, benchmark excess returns, blocked
dispositions, source hashes, availability ordering, and fail-closed omissions.
