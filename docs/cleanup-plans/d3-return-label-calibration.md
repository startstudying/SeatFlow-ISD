# D3 cleanup plan: fold-local return-label calibration

## Behavior to preserve

- Direction labels, graph tensors, D2 feature-transform state, and fail-closed
  checkpoint restore behavior remain unchanged.
- Public scores expose returns in the original label unit, never standardized
  training units.

## Behavior to add

1. Fit a mean/std return scaler for every horizon from training samples only.
2. Train the return head against standardized labels using Huber loss; restore
   predicted returns to original units for snapshots, scores, and selection.
3. Persist schema, horizon statistics, entry/exit definitions, state fingerprint,
   and diagnostics in checkpoints; reject a mismatched restore state.

## Regression tests before edits

- Deterministic fold-local fit/apply/inverse-transform and test-data isolation.
- Huber loss receives standardized targets.
- Prediction snapshots return original-unit values and restore rejects mismatch.

## Verification

- Targeted scaler, training, rolling, lint, type, and complete test suite.
- E3 uses the same bounded one-seed/one-fold range as E2 and compares output
  scale only; it remains `model_valid=false`.
