# D4 observable training cleanup plan

## Scope

Complete the existing numerical-stability training path without changing the
graph architecture or the research label definition.

## Guardrails

- Preserve `model_valid=false` for every training artifact.
- Keep the failed real-label E4 evidence; a control-label check may validate
  mechanics only and must never be reported as predictive validity.
- Do not mix a validation split into feature or return-transform fitting.
- Resume only when the stored run identity, configuration, sample hashes, and
  transform hashes match exactly; otherwise fail closed.

## Changes

1. Add a chronological, fold-local validation holdout for scheduler and early
   stopping monitoring.
2. Persist an atomic in-progress checkpoint after each epoch, including model,
   optimizer, scheduler, monitor state, and epoch metrics; recovery requires
   an exact run identity, configuration, sample, and transform-hash match.
3. Add regression tests for observability, validation-only scheduling, and
   strict resume identity checks.
4. Document both the failed real-label and passing control-label E4 outcomes.

## Verification

Run targeted training and rolling tests, then repository lint, type checking,
and the full test suite. Re-run E4 only after its provenance report is updated.
