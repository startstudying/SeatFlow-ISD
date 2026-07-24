# Pre-numerical-fix baseline

## Scope

This document freezes the legacy `dt_seathgt` development result before the
numerical-stability work and tensor-schema invalidation begin. It is evidence
of a failure mode, not a promotable model result. The frozen payload is
`artifacts/baselines/pre_numerical_fix/`; its hash manifest is the authority
for later comparisons.

## Frozen run

- Source run: `artifacts/research/dtseathgt_real_bounded_20260721`
- Candidate / baseline: `dt_seathgt` / `dt_event_seat_profile_logistic`
- Data: real Tushare snapshots, `institutional_proxy` seat detail, 17 event
  as-of days from 2023-02-06 through 2023-02-28; observed labels through
  2023-03-10.
- Design: 5 seeds (2026–2030), 3 development rolling folds, 3 epochs,
  gradient clip norm 1.0, return-head expected-return selection policy.
- Audit: 672 samples, 82 snapshot manifests, 4 tensorized chunks, and 15
  candidate seed/fold rows. The source result reports verified input evidence.

## Result: fail closed

The recorded result status is `evaluated_fail_closed`, with
`model_valid=false` and `negative_result=true`. Across the 15 candidate
seed/fold rows, mean AUC was 0.503539, mean Rank IC was -0.012702, mean Top-K
excess return after cost was -0.013002, and only 5 rows had a positive Top-K
excess return after cost.

This baseline must not be promoted, used to relax validity gates, or used to
choose a production threshold. Its data scope is not complete-seat evidence.

## Numerical signals retained for comparison

The frozen `result.json` contains per-fold/per-seed AUC, Rank IC, Top-K return,
drawdown, candidate scores, direction probabilities, return-head outputs, and
label returns. It also contains the old schema version (`1`) and the audit
fields needed to relate the result to the legacy chunks and checkpoints.

The old result has calibration instability: direction probabilities may be
near zero while expected returns are several whole units, despite label returns
being order-of-magnitude fractions. It is retained so numerical fixes can be
compared against evidence rather than memory.

## Evidentiary limits

The legacy result does **not** persist per-step losses, raw logits, gradient
statistics, or a standalone node/edge statistics table. Those values cannot be
reconstructed from checkpoints and pickle chunks without executing the old
model code, and must not be invented for the comparison report. The frozen
result does persist per-fold/per-seed probabilities, return-head values, label
returns, metrics, the schema version, and an audit summary (including chunk
count and snapshot-manifest hash). Future E1 reporting must explicitly record
the previously missing diagnostics before drawing a numerical-stability claim.

## Payload and verification

The baseline payload contains the result, experiment configuration, matured
label store, all 15 legacy checkpoints, and all 4 tensorized chunks. The
manifest records each payload file's relative source path, byte count, and
SHA-256.

Verify the frozen evidence without reading or executing model code:

```powershell
python scripts/reproduce_pre_numerical_fix.py --verify
```

This command verifies every file hash and re-derives the key failure-state
summary from the frozen `result.json`. It is intentionally not a retraining
command: the legacy tensor schema will be invalid after the numerical fix, and
rerunning it with changed code would not be a valid reproduction.
