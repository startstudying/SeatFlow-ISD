# SeatFlow DT-SeatHGT model card

Updated: 2026-07-19

This document is the M11 release-facing model-card template. It is intentionally
fail-closed: it records the current engineering status without claiming a valid
competition model until real complete-seat data, frozen blind evidence, lineage,
and out-of-sample backtest evidence are present.

## Current status

| Claim | Status | Evidence boundary |
| --- | --- | --- |
| DT-SeatHGT/PyG implementation exists | Implemented | Engineering and CPU shape smoke only. |
| Real complete-seat training completed | Not claimed | Requires verified full historical seat-detail source and audit report. |
| Model is valid for competition ranking | Not claimed | Requires `model_valid=true` in `artifacts/status/phase-state.json`. |
| Blind protocol is frozen and verified | Not claimed | Requires a verified blind manifest hash in phase-state evidence. |
| Release can be packaged | Not claimed | Requires `competition_ready=true` plus M11 evidence files. |

## Intended model

- Architecture: dynamic temporal heterogeneous graph model over stock, seat, and
  flow relations.
- Core implementation: DT-SeatHGT/PyG path with explicit multi-horizon heads.
- Required target horizons: 1, 3, and 5 trading days.
- Required data scope for release claims: `data_scope=real` and
  `seat_detail_scope=complete_seat`.

## Required evidence before any "valid model" claim

1. Data audit report proving the graph snapshot is built from complete historical
   brokerage seat details, not only `top_inst` institutional aggregates.
2. Frozen blind split manifest with feature and label hashes and no label
   feedback.
3. Training lineage artifact binding:
   - data snapshot hash,
   - config hash,
   - code hash,
   - seed list,
   - split hash,
   - blind manifest hash,
   - checkpoint hash.
4. Audited multi-seed, rolling-fold raw metrics for the DT model and a strong
   baseline.
5. Fixed ablation matrix showing stable positive contribution from seat-profile,
   time-edge, and direction-edge components.
6. Promotion record setting `model_valid=true` and `backtest_valid=true` only
   after the unified model/backtest gate passes. `competition_ready=true`
   additionally requires an explicit verified release contract. Contaminated OOS
   evidence and any unsealed or non-independent blind manifest remain negative
   results and cannot promote.

## Known current limitations

- No real complete-seat dataset is committed in this repository.
- No user token or secret is stored or referenced by this model card.
- Engineering smoke results are not model-quality evidence.
- A Tushare screenshot/token alone is not sufficient evidence and should be
  rotated if it was exposed.
