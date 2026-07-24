# SeatFlow Production V1 — Execution Prompt

You are the sole delivery owner for `SeatFlow Production V1`. Execute the full plan in
`.omx/plans/seatflow-production-v1-rebuild.md` to a verifiable conclusion. Do not skip,
reorder, or relax its hard gates; do not manually set `model_valid=true`; do not introduce
automatic live trading.

## Operating contract

1. Inspect the current repository, artifacts, data availability, and prior findings before
   each phase. Preserve unrelated work.
2. Work in the plan's order: D0 -> D1/E1 -> D2/E2 -> D3/E3 -> D4/E4 -> R0/R1/R2 ->
   R3 -> R4 -> R5. A failed gate must remain fail-closed and redirect work to its stated
   remediation path; never claim performance from a numerical diagnostic.
3. Run one major variable per experiment and record code revision, config, input manifest,
   feature/tensor schema, seeds, logs, metrics, checkpoint, and output hashes.
4. Enforce the semantic split: anonymous institutional rows are `InstitutionFlowEvent`, not
   stable seats. Entity IDs and hashes never enter continuous feature tensors.
5. Keep raw data and manifests; quarantine incompatible chunks, checkpoints, and registries
   instead of silently reusing them. All loaders must validate the current schema and hashes.
6. Use fold-local transformations and label scaling only; retain fit statistics and sample
   lineage in artifacts. Ensure finite inputs and outputs, bounded diagnostics, and explicit
   failure messages.
7. Use the same eligibility, exposure, cost, execution, and risk rules for baseline,
   backtest, and paper trading. The product remains recommendation/paper-trading only.

## Required evidence per phase

- **D0:** immutable legacy baseline manifest and reproduction report.
- **D1/E1:** no pseudo-continuous entity hashes, new tensor schema, legacy artifact rejection,
  and a one-seed/one-fold/3–10 epoch diagnostic using verified snapshots.
- **D2–D4:** leakage-safe fold transforms, calibrated Huber return head, resumable
  validation-driven trainer, micro-overfit evidence, and all numerical gates.
- **R0–R2:** data audit, tradable label V2, matched Logistic/LightGBM/CatBoost/MLP baselines,
  daily ranking metrics, and multi-window/fold/seed promotion evidence.
- **R3–R5:** semantically correct bidirectional FlowHGT graph, ablation evidence, model
  registry/degradation, paper-trading parity, fail-closed service/UI, and complete-seat
  augmentation only when lawful audited data exists.

## Completion rule

For every numbered task and gate in the plan, provide a concrete artifact, test or command
output proving it. If a necessary external data source is unavailable, record the failed-closed
result, preserve the evidence, continue with every independent task, and state precisely what
external state prevents the next gated phase. Never mark the plan complete on indirect evidence.
