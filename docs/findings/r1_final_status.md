# R1 final status

R1 is frozen as historical research evidence. It is not a valid model and its
out-of-sample results are not an independent validation set because those
results participated in multiple rounds of research selection.

## Frozen decision

- `model_valid = false`
- `research_only = true`
- `independent_validation = false`
- `oos_selection_contaminated = true`
- Positive-return runs: 6 of 15
- Required development threshold: 10 of 15

## Permitted use

R1 may be used only as a historical comparison and as diagnostic evidence for
the problems R2 is intended to address.

## Prohibited reuse

R2 must not read R1 results to choose loss weights, seeds, folds, Top-5
thresholds, features, model structure, or ensemble weights. The R1 artifacts
remain read-only, and later R2 results must not change this status.
