# Future-blind protocol

R2 is not future-blind merely because an interval is called out-of-sample.
Any period whose realized labels were viewed during research is contaminated
and remains historical OOS evidence.

## Freeze prerequisites

Before the first blind prediction, freeze and hash:

- model structure and CPU state;
- experiment configuration;
- code commit;
- training-data cutoff;
- seeds 2026–2030 and equal weights;
- prediction pipeline.

The blind period begins on the first trading day whose label has not been read
after all six items are frozen.

## Append-only prediction record

Every prediction record contains:

- `prediction_date`
- `stock_id`
- `predicted_rank`
- `predicted_residual_return`
- `model_hash`
- `config_hash`
- `data_cutoff`
- `generated_at`
- `file_sha256`

Records are appended to JSONL. An existing `(prediction_date, stock_id)` record
cannot be overwritten. Feature and label files are frozen in a manifest with
independent hashes.

## Blind-period prohibitions

During collection, do not read realized labels, modify the model or
configuration, change seeds or weights, compare candidates, or pause the model
because of observed market performance. Nested label-like keys are rejected by
the prediction guard.

After the period ends, unseal once, evaluate the frozen candidate, and record
`model_valid = true` or `model_valid = false`. Development-gate success alone
never sets `model_valid = true`.
