# D4 root-cause: independent feature-representability probe

Date: 2026-07-22

## Question

E4 failed to exceed the mandatory training AUC gate on a fixed 32-sample
slice.  This independent experiment distinguishes two explanations for that
micro-overfit failure:

1. the labels cannot be represented by the current target-stock features; or
2. the DT-SeatHGT model/training path is preventing the fit.

## Independent probe

The probe does not import or call DT-SeatHGT or its training loop.  It reads
only the same 32 samples and the current eight-dimensional `stock.x` row at
each sample's target node.  A two-hidden-layer MLP is trained on horizon-1
direction labels with seeds 2026, 2027, and 2028.  The identical probe is also
trained on deterministic labels derived from those features as a positive
control.

Artifact: `artifacts/diagnostics/D4_ROOT_CAUSE_FEATURE_PROBE/result.json`

Artifact payload SHA-256: `4d940d8de260641e799d2f0774db5b6ea23dd7c508e7c4701c0ffd7dab7498c5`

## Result

- All 32 target feature rows are unique; there are no duplicate rows with
  conflicting horizon-1 labels.
- Real-label runs reached AUC 1.0 and accuracy 1.0 for all three seeds.
- Synthetic feature-derived positive-control runs reached AUC 1.0 and
  accuracy 1.0 for all three seeds.
- The real-label probe reached the gate in 25, 50, and 25 epochs.

## Root-cause decision

For the fixed E4 training slice, the horizon-1 labels are representable and
memorized from the current target-stock features alone.  The E4 failure is
therefore not attributable to raw target-feature non-representability on this
slice.  The remaining root-cause branch is the DT-SeatHGT model/training path,
including task interference, regularization, message-passing behavior, and
optimization configuration.

This conclusion is deliberately narrow.  It does not establish out-of-sample
signal, feature sufficiency for generalization, model validity, promotion
readiness, or trading readiness.  All downstream evaluation remains blocked.

## Next single-variable localization

Run a direction-only DT-SeatHGT micro-fit on the identical samples and
horizon, changing only removal of the auxiliary return objective.  If it
passes, multi-task interference is the leading cause.  If it fails, retain the
direction-only setup and remove dropout as the next single-variable test.

