# D3 / E3: Fold-local return-label calibration

Date: 2026-07-22

## Scope and evidence

E3 changed only the return-head target treatment after D2: return labels are
standardized from the training fold, optimized with Huber loss, then inverse
transformed before every exported prediction, score, and selection decision.
The data range, one fold, seed `2026`, and three epochs match E2.

- Result: `artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/result.json`
  - SHA-256: `01caedcc6e8995699cd537c5dda18cd7f5791d953e643694d5ea00cff84ac9d1`
- Return state: `artifacts/transforms/E3_CALIBRATED_RETURN_HEAD/fold_0/return_transform.json`
  - SHA-256: `0e6185e4f5c092e3af2ea2f5f1c635dd960e05b475e45eefd727834988156f63`
- Distribution diagnostics: `artifacts/diagnostics/E3_CALIBRATED_RETURN_HEAD/fold_0/return_distribution.json`
  - SHA-256: `1b84d3d334f1891f45586cc36366aa9d0bff7ade338690c64ae5c6ed96b41bfa`

The serialized state records `label_schema_version`, horizons, per-horizon
mean/std/count, training manifest hash, and the current close-to-close entry
and exit definitions.  Checkpoint restore verifies the identical return-state
fingerprint and serialized payload.

## E3 observation

The 672-sample diagnostic is audited but remains `evaluated_fail_closed` and
`model_valid=false`. Candidate AUC is 0.489985, RankIC 0.011977, and Top-K
mean return is 0.004503. Exported predictions range from -0.031518 to 0.040352,
versus label returns from -0.123937 to 0.100546. This is the same order of
magnitude and is sufficient only for D3's scale gate; it is not evidence of
predictive validity or readiness for automated trading.
