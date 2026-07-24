# R1 Unblock Attempt — 2026-07-23

## Outcome

R1 remains `evaluated_fail_closed`. No threshold was relaxed and
`model_valid` remains `false`.

The investigation corrected one feature-semantics defect, added point-in-time
features required by the production plan, and completed a five-seed/three-fold
ensemble check. These changes substantially improved daily ranking quality but
did not establish reproducibly positive long-only Top-5 excess return.

## Frozen inputs

- V2 tensor dataset: 44,033 samples, 856 source dates.
- Dataset canonical evidence SHA-256:
  `c6fe493b6169e63cb0a8b8ef29d073b4142c97a7d97f1d120cc5282a66544fc9`.
- Original R1 canonical evidence SHA-256:
  `ab9a5f5a7dd2fee54a88f33c426f55126a2f3bbe042259431655cb7af138fe8b`.
- Seeds: 2026, 2027, 2028, 2029, 2030.
- Folds: the exact frozen R1 fold memberships.

## Corrected amount semantics

The original table baseline aggregated the already tensorized
`log1p(amount)` edge values. A sum of logarithms is not a total amount. The
baseline extractor now inverts the tensor transform before aggregating raw
buy/sell amounts. A regression test fixes this contract.

The corrected 20-feature matrix manifest is
`R1_TABULAR_MATRIX_RAW_AMOUNT_856D/manifest.json`, with canonical evidence
SHA-256
`61629a351800cf82256da7da391c16e3cd0b9b4a60c38b56444c54fbcadc6fab`.

The isolated five-seed probe remained negative:

- Logistic Top-5 mean: -0.5811%.
- MLP Top-5 mean: -0.4276%.
- LightGBM Top-5 mean: -0.3661%.

Canonical evidence SHA-256:
`662cda1c5867eec309d2dc9ac417e03d2be55a4619e4d7eb99c4ad70af2bf8ef`.

## Point-in-time feature blocks

Three audited feature blocks were evaluated without changing the frozen labels,
folds, exposure, or Top-5 policy:

1. Six-trading-date-purged stock event frequency and realized event history.
2. Daily amount and institutional-flow normalization.
3. Tushare `daily_basic` turnover, volume-ratio, share-count, and market-value
   fields with explicit missing indicators.

The `daily_basic` backfill completed all 856 dates. Its canonical evidence
SHA-256 is
`9359ca7743725f25febcb5a2b9b484c06888f360c5efbc424f898d981ef589d6`.
The final 57-feature matrix has 100% daily-basic sample coverage; its manifest
canonical evidence SHA-256 is
`4bccf8b584653c2385b866fd13fe492e136cdcf28381a449816c75c687d96372`.

The best isolated feature result was the 60-day-minimum market-normalized
LightGBM:

- Daily RankIC mean: 0.04383.
- Daily Top-5 excess mean: -0.02860%.
- Positive seed-fold fraction: 2/3.

It failed because the average Top-5 excess return remained non-positive.
Canonical evidence SHA-256:
`20e7748c7bfd694a5f7a94ae3040f3f196bf1ca2162de112067b2ada06a75562`.

A one-year minimum training-history check did not repair the failure:

- Daily RankIC mean: 0.06209.
- Daily Top-5 excess mean: -0.02728%.
- Positive seed-fold fraction: 1/3.

Canonical evidence SHA-256:
`25bd026d06931c085572f36a713664b0d14f81f3cb289015dcbc588992fc1a1d`.

## Fixed equal-weight ensemble

An untuned equal-weight Logistic/MLP/LightGBM ensemble was evaluated across all
15 seed-fold pairs on the complete daily-basic matrix. The one-seed diagnostic
was promising, but the required five-seed run rejected the candidate:

- Daily RankIC mean: 0.05664.
- Daily Top-5 excess mean: -0.04570%.
- Positive seed-fold fraction: 1/3.
- R1 performance gate: failed.

Canonical evidence SHA-256:
`e0123cef51f6b92df6e5a3301c77cc64d5cb8be23891b540fec14ec5fc161a64`.

## Remaining blockers

1. No tested candidate has positive mean Top-5 excess return with at least
   two-thirds positive seed-fold runs.
2. The production plan requires a CatBoost rolling baseline, but CatBoost is
   not installed. It was not silently substituted.
3. Historical point-in-time industry and concept coverage is still absent.
   The current `stock_basic` endpoint is available, but applying a current
   industry snapshot to historical rows would leak future classification state.
4. There is no sealed future-blind result; all current OOS research remains
   development evidence.

The next lawful R1 action is to complete the CatBoost baseline after explicit
dependency authorization, or acquire audited historical industry/concept
snapshots. Further tuning against these same folds would be contaminated
selection and is not an acceptable unblock path.

## Authorized CatBoost and point-in-time industry continuation

The CatBoost dependency was explicitly authorized and installed as version
1.2.10. The R1 baseline runner had an implementation defect: it detected
CatBoost but never included it in the model loop, so
`catboost_disposition="completed"` was unreachable. The runner now executes the
required model and only reports completion after every requested seed-fold run
exists.

The complete CatBoost check on the 57-feature daily-basic matrix produced:

- Daily RankIC mean: 0.04884.
- Daily Top-5 excess mean: -0.16851%.
- Positive seed-fold fraction: 7/15.
- Canonical evidence SHA-256:
  `4c37f670d3d741b604f23312789dff81483d1e01f26b2eab2c30aba8c9d23bc9`.

CatBoost is therefore no longer a dependency blocker, but its baseline did not
establish the required positive signal.

The authorized Tushare account also exposed historical SW2021 industry
membership through `index_member_all`, including `in_date` and `out_date`.
Sixty-two audited current/historical L1 snapshots were materialized with
canonical evidence SHA-256
`75c750c99d84cfc0b2db16d9fc1be085c7a2ef56b2b1bcc7ee82734c21c52b86`.
Point-in-time assignment covered 99.973% of R1 samples and expanded the matrix
to 90 features without using labels.

On the original 60-day-minimum folds, industry-enhanced CatBoost improved but
still failed:

- Daily RankIC mean: 0.04685.
- Daily Top-5 excess mean: -0.13856%.
- Positive seed-fold fraction: 8/15.
- Canonical evidence SHA-256:
  `9f280bc871c73eb3872d93796eba0ed0c2b04c101e295bb6c417d5e4db402eba`.

The original fold 0 trains on only about 60 trading dates and tests for more
than one year. A separately recorded minimum-252-day training-history
experiment, already allowed as a single changed variable, removed that
short-history defect. Its industry-enhanced CatBoost result was the closest
valid candidate:

- Daily RankIC mean: 0.06449.
- Daily Top-5 excess mean: +0.00338%.
- Positive seed-fold fraction: 6/15.
- Canonical evidence SHA-256:
  `a2a688a0167037e9ca2aa970632473eb11fb86ecd1f273af1f6d78aa567e65bf`.

The mean-return condition passed, but the two-thirds stability condition did
not. Deterministic CatBoost, native categorical industry encoding, and
point-in-time broad-market state were each tested as isolated one-seed
diagnostics and rejected before five-seed expansion.

R1 remains fail-closed. The remaining performance blocker is now narrow:
current features provide positive average cross-sectional information, but no
tested candidate is positive in at least two-thirds of seed-fold runs. A sealed
future-blind period or genuinely new point-in-time concept information is
required before another selection decision; post-hoc aggregation tuning on
these contaminated OOS folds is not admissible.

## Concept-data availability audit

The final independent data path was checked against the configured Tushare
account:

- `ths_member` is accessible, but the documented placeholder membership dates
  are absent in the live response. A sampled 319-member board had zero rows
  with `in_date` or `out_date`, and every row was current (`is_new=Y`).
- `tdx_member` is accessible with dated snapshots, but sampled requests
  returned no rows for 2023-01-03, 2024-01-03, or 2025-01-03. Rows were first
  observed in the May 2025 period.
- `dc_concept_cons` is documented as beginning on 2026-02-03.

None of these sources can lawfully populate the frozen 2023–2024 folds.
Backfilling current memberships would leak future classification state. The
audit is recorded at
`artifacts/diagnostics/R1_CONCEPT_DATA_AVAILABILITY_20260723/result.json`.

No further admissible model-selection branch remains on the currently exposed
data. R1 can resume when either audited effective-dated concept membership
covering 2023 onward becomes available or the sealed future-blind window can be
evaluated without label feedback contamination.
