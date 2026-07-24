# D2 / E2: Fold-local transforms and numerical-stability diagnostic

Date: 2026-07-22

## Scope and gate

E2 changed exactly one primary variable from E1: fold-local numeric transforms.
It used the same verified historical-backfill snapshot range (2023-02-06 through
2023-02-28), one fold, seed `2026`, and three epochs.  This is a numerical
diagnostic only and is not a model-validity, performance, or promotion claim.

## Reproducible evidence

- Command: `python scripts/run_e2_fold_local_transforms.py`
- Result: `artifacts/diagnostics/E2_FOLD_LOCAL_TRANSFORMS_V2/result.json`
  - SHA-256: `37c870a1d1d2019d427201a30e6e3aa6edefdd339b325e7cc11cef34cfc4d7e8`
- Fold state: `artifacts/transforms/E2_FOLD_LOCAL_TRANSFORMS_V2/fold_0/feature_transform.json`
  - SHA-256: `0b5dd245b4054fd5f05e621443bf2015dc688e22db4b1b1f10a278583960c504`
- Diagnostics: `artifacts/diagnostics/E2_FOLD_LOCAL_TRANSFORMS_V2/fold_0/feature_distribution.json`
  - SHA-256: `6b168b526697a4ca8b869c0aeab268c0e1200368b623f077b66a186ce65edfbc`

The state is version `seatflow_fold_graph_transform_v1`, contains a 64-character
training manifest hash and per-node/per-edge tensor-block transform states.  The
checkpoint contains the identical transform fingerprint, serialized state, and
diagnostics.  Restore rejects a mismatched transform state.

## E2 observations

- 672 samples; one seed; one fold; audited verified snapshots.
- Candidate AUC: 0.472152; RankIC: -0.019055; Top-K mean return: 0.003003.
- Return predictions range from -0.270436 to 0.194247 while labels range from
  -0.123937 to 0.100546.  This is materially closer than the pre-D2 diagnostic
  mismatch but is not return-head calibration evidence; D3 remains required.
- Result status is `evaluated_fail_closed`, with `model_valid=false` and
  `negative_result=true`.

## Gate decision

D2's data-isolation and artifact requirements are evidenced: transforms are
fitted only from each training fold, applied to cloned train/test graphs, and
persisted with feature distribution diagnostics.  The experiment does not pass
or bypass any downstream performance or promotion gate.  Proceed to D3 only as
the next isolated variable: fold-local return-label scaling and inverse transform.
