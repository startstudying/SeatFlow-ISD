# D4 / E4: Micro-overfit gate — failed closed

Date: 2026-07-22

E4 trained the fixed D4 configuration on exactly 32 current-schema samples:
100 epochs, AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient clip
`1.0`, hidden dimension `64`, two layers, and dropout `0.3`.

Artifact: `artifacts/diagnostics/E4_MICRO_OVERFIT/result.json`

The loss decreased (6.365469 to 3.144942), outputs remained finite and did not
explode, but train AUC reached only 0.706349, below the mandatory `>0.95`
micro-overfit gate. The artifact remains `evaluated_fail_closed`,
`model_valid=false`, and `negative_result=true`.

## Gate decision

The plan explicitly forbids rolling evaluation after a failed E4. Therefore no
R0/R1/R2 rolling, long-window research, paper trading, or production claim may
follow this artifact. The next permitted action is a separately documented D4
root-cause investigation; it must retain the failed result and alter only one
diagnosed variable per follow-up experiment.

## Control mechanics check

`artifacts/diagnostics/E4_MICRO_OVERFIT_CONTROL_LABELS/result.json` is a
separate deterministic control experiment. It derives binary labels from the
already-present `stock_return_1d` continuous feature, without using an entity
identifier or hash, and reaches the mechanical AUC gate. This demonstrates
that the current-schema optimizer, heads, and checkpoint restore can fit a
learnable fixed control target. It is not an out-of-sample result, not a real
label result, and does not establish predictive validity or permit promotion.
