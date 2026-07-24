# D2 cleanup plan: fold-local feature transforms

## Behavior to preserve

- Tensor graph topology, target mapping, frozen label policy, and D1 schema rejection remain
  unchanged.
- Existing rolling evaluation stays fail-closed when a partition is incomplete or malformed.

## Behavior to add

1. Fit only on each fold's training samples, with a persisted transform state containing schema,
   feature order, statistics, clipping bounds, fitted sample count/date range, and manifest hash.
2. Apply signed-log plus train-derived clipping/standardization to amount-like features; apply
   train-derived winsorization/standardization to returns, momentum, and volatility; preserve
   bounded correlation/decay and binary values.
3. Reject NaN, Inf, feature-order mismatch, and non-train-fitted states; map a
   zero-variance training feature to finite zero without refitting it on test data.
4. Emit per-fold feature distribution diagnostics and attach their hashes to checkpoints.

## Regression tests before edits

- Fit/apply determinism, no test-data influence, zero-variance handling, clipping, and finite
  outputs.
- Entity IDs cannot be transformed as continuous inputs.
- Restore rejects a checkpoint whose transform state or feature schema does not match.

## Verification

- Targeted transform, tensor, training, and checkpoint tests; then full lint/typecheck/test.
- E2 uses the same one-seed/one-fold bounded data slice as E1 and reports only numerical
  stability, never candidate validity or promotion.
