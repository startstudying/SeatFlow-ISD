# Research Findings

## 2026-07-21: Bounded DT-SeatHGT Real Evaluation

### What Was Tested

`dt_seathgt` was evaluated against `dt_event_seat_profile_logistic` using
verified local Tushare snapshots from 2023-02-06 through 2023-02-28. The run
used 672 samples, five seeds, three expanding folds, three training epochs,
gradient clipping, return-head selection scoring, and 45 ablations.

Artifact: `artifacts/research/dtseathgt_real_bounded_20260721/result.json`
SHA-256: `c68b6e2ed3565c5d5cd7c12e03c727c5ae45043dc39b5fdda6f2fce27997ba2b`

### Outcome

The candidate did not improve on the strong baseline. Across 15 seed/fold
pairs, mean candidate AUC was 0.503539, mean Rank IC was -0.012702, and mean
Top-K excess return after cost was -0.013002. Mean paired AUC delta was
-0.024971, mean paired Rank IC delta was -0.097469, and mean drawdown delta
was +0.012804. Five of 15 pairs had positive Top-K excess after cost.

The artifact is audited but remains `evaluated_fail_closed`, `model_valid=false`,
and `negative_result=true`.

### Interpretation

This is a negative diagnostic result, not evidence that the architecture is
universally ineffective. The window is bounded and development-only, the data
is `institutional_proxy` rather than complete-seat data, and direction
probabilities/return-head values showed calibration instability in some folds.
It does not support a performance-improvement or promotion claim.

### Constraints For Follow-up

- Do not promote or relax validity gates from this result.
- Do not use a threshold chosen from this development window for promotion.
- Do not compare a candidate with abstention against a baseline without matching
  exposure, cost, and selection policy.
- Do not treat `institutional_proxy` coverage as complete-seat evidence.

### Next Experiments

1. Audit calibration and return-head scale before increasing the search space.
2. Run isolated ablations for training duration, selection score, stock features,
   and abstention policy.
3. Repeat only the justified configuration on the full development window, then
   evaluate a sealed final blind window when labels are released.

## 2026-07-22: D1 Entity-ID Boundary and Tensor Schema

The pre-fix result and its 22-file payload are frozen under
`artifacts/baselines/pre_numerical_fix/` and verify with
`python scripts/reproduce_pre_numerical_fix.py --verify`. The legacy result remains
`evaluated_fail_closed` and `model_valid=false`.

`stock_id_hash` and `side_hash` were removed from continuous edge attributes.
New tensors use `seatflow_tensor_v3_no_id_hash`, include a deterministic schema
fingerprint, and the rolling loader rejects unversioned or mismatched spill chunks.
The legacy chunks and checkpoints are therefore quarantined by incompatibility;
they remain preserved only as baseline evidence.

The new E1 diagnostic entry point enforces one seed, one fold, and 3-10 epochs.
It completed against the verified historical-backfill snapshots with 672 samples and
remains `evaluated_fail_closed` / `model_valid=false`; it is a schema-boundary diagnostic,
not a validity or promotion result. Its return-head scale mismatch is deferred to D3.

## 2026-07-22: D4 Independent Feature-Representability Probe

The failed E4 micro-overfit gate was followed by an independent probe that did
not use DT-SeatHGT or its training loop. On the identical 32-sample slice, all
32 current target `stock.x` rows were unique with no exact feature/label
conflicts. A two-hidden-layer MLP fitted the real horizon-1 labels to AUC 1.0
and accuracy 1.0 for seeds 2026, 2027, and 2028; the feature-derived synthetic
positive control also passed all seeds.

This rules out target-feature non-representability as the explanation for the
E4 training-slice memorization failure. The remaining D4 branch is the
DT-SeatHGT model/training path. The next permitted experiment is a
direction-only run on the same samples, followed by a dropout-only isolation
only if direction-only still fails. This remains fail-closed evidence and does
not justify rolling evaluation, promotion, or trading claims.
