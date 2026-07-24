# R2 edge-consistency status

Edge-perturbation consistency is deliberately deferred from the first
R2-SeatRankLite implementation.

## Frozen first-version decision

- Edge-consistency loss weight is `0`.
- No random edge dropping is applied during the engineering smoke test.
- No consistency result may be used to select the E0–E4 development structure.

## Activation gate

The experiment may be declared only after the E4 baseline is stable. A future
protocol must predeclare the non-core edge definition, the fixed 5% drop rate,
the ranking-consistency metric, and the acceptance rule before results are
viewed.

This deferred experiment does not enter the first R2 model, loss, or validation
claim.
