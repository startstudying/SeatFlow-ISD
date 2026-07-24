# R0: executable label V2 evidence — fail closed

Date: 2026-07-22

## Implemented contract

`executable_excess_return_v2` uses a T+1 executable open and ends at T+2,
T+4, and T+6 for the 1/3/5-day horizons.  Stock returns use an actual,
source-hashed Tushare `adj_factor`; benchmark returns use the corresponding
Tushare `index_daily` opening price. Each label records its benchmark, price
policy, adjustment policy, availability, disposition, and source hash.

The tensor record now carries a `label_schema_version`. A V2 sample requires
the dedicated three-class direction head, while legacy samples require the
binary BCE head. The legacy rolling evaluator explicitly rejects V2 rather
than applying binary AUC/rank metrics to `-1/0/1` targets; the dedicated V2
rolling evaluator remains a required R0 deliverable.

## External data evidence

`scripts/probe_tushare_label_v2.py` verified read-only Tushare access and
stored a two-row `adj_factor` query with row and query hashes in
`artifacts/diagnostics/R0_TUSHARE_LABEL_V2_READINESS/result.json`.

`scripts/build_r0_v2_candidate_dataset.py` queried a frozen 2024-01 window
for stock `000001.SZ`, benchmark `000300.SH`, trading calendar, actual factors,
and a suspension response. It wrote three horizon candidates to
`artifacts/diagnostics/R0_V2_CANDIDATE_DATASET/result.json`.

## Gate decision

The candidate builder now binds the complete suspension, listing, and
name-change responses into every label's market-status hash, including a clear
"not suspended" result. It applies the existing board-aware upper and lower
price-limit rule to every date from T+1 through T+6. The frozen probe produced
three eligible V2 horizon labels for one stock/date; its exact result is
recorded in the candidate artifact.

This is only a single-stock contract probe, not a research dataset. No V2
tensor chunk/checkpoint is produced until the candidate builder is extended to
the complete eligible universe and the dedicated three-class rolling evaluator
is implemented. No model or promotion claim is made.

## Downstream schema migration evidence

`src/seatflow/datasets/v2_labels.py` verifies the whole artifact payload hash
before deserializing any label, then preserves the artifact's explicitly
fail-closed state. `build_window_samples(..., v2_labels_by_key=...)` accepts
only one label schema per sample, requires all requested horizons to be
eligible, and does not fall back to legacy labels. `tensorize_window_samples`
then emits `executable_excess_return_v2`, with three-way direction labels and
excess returns. This establishes an auditable V2 training-input path while
preventing the single-stock probe from being represented as a completed
historical training set.

## Production V1 protocol freeze

`configs/experiments/production_v1.toml` is a distinct, frozen Production V1
research contract. It declares `InstitutionFlowEvent` as the only supported
anonymous-entity semantics, requires `executable_excess_return_v2`, lists the
required Logistic Regression/LightGBM/CatBoost/MLP baselines, seals the future
blind window, and records an unvalidated DT-Flow family. It deliberately does
not assert a stable-seat source or model validity. The verified lock is
`artifacts/protocol/seatflow-production-v1-v2/protocol.lock.json`, SHA-256
`ea64743016f942c674b981691ea6723b5a69b1bc1528d4fb380f016318b7a382`.

## Daily candidate and label-availability audit

`scripts/audit_r0_v2_label_coverage.py` verifies every discovered V2 label
artifact's fail-closed status, schema, and file/artifact hashes before it
aggregates any count. Its initial report,
`artifacts/diagnostics/R0_V2_LABEL_COVERAGE_AUDIT/result.json`, covers 185 of
856 verified event dates. It exposes 28,134 candidate label records: 9,378 at
each of the 1/3/5-day horizons, with 6,977 eligible at each horizon (74.40%).
The audit explicitly records 2,631 `missing_market_status`, 3,057
`new_stock_restricted`, 1,014 `price_limit_blocked`, and 501 `suspended`
labels rather than silently treating them as eligible. Its direction counts
are 12,998 negative and 7,933 positive eligible labels; the absence of a
neutral count is itself visible. The report is `audited_fail_closed` and keeps
`model_valid=false`.

## Identity stability and feature-coverage audit

`scripts/audit_seat_identity_stability.py` independently scanned all 856
verified event snapshots (798,517 events), stored their aggregate source hash
`75f2f679a2e73ee5b548b4e15587f0606ce6f1dc76c189b0d8c85f66dd130687`, and
remains `audited_fail_closed`. It found 59,817 `institutional_flow_proxy` IDs:
every one is active for exactly one trade date, yielding zero 5/20/60-day
repeat rate. They are therefore `InstitutionFlowEvent` data only, never
StableSeat data. The snapshot also contains 13,550 name-form IDs with a
79.76% multi-date repeat ratio, but source repetition does not independently
prove complete, stable, beneficial-owner, or position-continuity semantics;
the report deliberately leaves `stable_seat_authorized=false`.

The same audit makes feature gaps explicit: `available_time` and event reason
coverage are 100%, while industry and concept coverage are both 0%. These
missing static relations prohibit a claim of complete graph coverage and are
a direct M0/R0 blocker, not values to impute silently.

`scripts/build_m0_data_contract_audit.py` now binds the identity report, V2
coverage report, and frozen Production V1 protocol into
`artifacts/diagnostics/M0_DATA_CONTRACT_AUDIT/result.json`. It declares the
only currently permitted research entity as `InstitutionFlowEvent`, retains
`model_valid=false`, and records four blocking conditions: no StableSeat
authorization, missing industry coverage, missing concept coverage, and
incomplete V2 label materialization. This is the authoritative M0 contract
state for the present data, rather than an inference from a subset experiment.

## First complete event-date cross-section

The read-only materializer was run for every target in the verified Tushare
Dragon-Tiger seat-event snapshot dated 2024-01-02. Its artifact,
`artifacts/diagnostics/R0_V2_UNIVERSE_2024_01_02/result.json`, contains 56
target stocks and 168 V2 label records. 147 records are executable; the other
records retain their explicit price-limit, suspension, ST, new-stock, or
missing-data disposition rather than being dropped from the evidence.

`scripts/build_r0_v2_tensorized_snapshot.py` joined the hash-verified label
artifact to manifest-verified historical snapshots and wrote
`artifacts/diagnostics/R0_V2_TENSORS_2024_01_02/`. It contains 49 tensorized
stock samples with all three eligible horizons, each declared as
`executable_excess_return_v2`. Its manifest binds the tensor-chunk hash, the
label artifact hash, and the verified snapshot hash. It remains
`materialized_fail_closed` with `model_valid=false`: one cross-section is
training-input evidence, not a rolling evaluation or a promotion decision.

The same read-only procedure was repeated for 2024-01-03 and 2024-01-04.
The three independently hashed event-date artifacts cover 169 target-stock
occurrences, 507 V2 label records, and 420 executable labels. Their tensor
chunks contain 49, 51, and 40 complete-horizon V2 samples respectively. This
is multi-date materialization evidence only; it is still far short of the
multi-year rolling dataset required to pass R0/R1.

## V2 three-class fold smoke decision

The 15-date, 622-sample V2 tensor dataset was evaluated in three purged and
embargoed folds using a one-epoch, one-seed three-class diagnostic. The fully
traceable result is
`artifacts/diagnostics/R0_V2_THREE_CLASS_FOLD_SMOKE_2024_01_02_22/result.json`.
It is a negative result: fold 0 RankIC is -0.1391, folds 0 and 1 have negative
Top-K mean excess return, and all three folds have zero direction accuracy.
The later folds' positive RankIC values are not sufficient counter-evidence
because the sample is short, the training windows are tiny, and the sign is not
consistent. The result remains `evaluated_fail_closed`; it does not pass R0 or
authorize R1/R2 graph-model validity claims.

## Expanded coverage and repeated negative smoke decision

The bounded, resumable materializer has now produced 52 separately
hash-verified V2 label and tensor chunks: 37 dates from 2023-01-03 through
2023-03-01, plus 15 dates from 2024-01-02 through 2024-01-22. The assembler
verified every source chunk's schema, status, and hash before producing
`artifacts/diagnostics/R0_V2_TENSOR_DATASET_2023_01_03_2024_01_22_52D/`:
1,763 samples, all with `executable_excess_return_v2`, and dataset SHA-256
`241aa0e4a510e3f8a51547f4a58a079a809dc199dad02368a3741b8995e9d43a`.

Keeping the three-class model, one epoch, seed, and fold construction fixed,
the same smoke experiment was repeated with **only historical V2 sample
coverage changed**. Its result is
`artifacts/diagnostics/R0_V2_THREE_CLASS_FOLD_SMOKE_52D/result.json`; each
checkpoint hash and the assembled-dataset hash are recorded there. The three
folds have RankIC -0.0258, 0.0096, and 0.0344, and their Top-K mean excess
returns are -3.52%, -1.47%, and -1.49%. Direction accuracy is 0.0 in every
fold. This is again a negative, fail-closed result. It neither establishes
signal nor permits R1/R2 advancement.

The same experiment was then repeated on a separately frozen 82-date dataset
after 20 further date chunks had been materialized. The assembler wrote
`artifacts/diagnostics/R0_V2_TENSOR_DATASET_2023_01_03_2024_01_22_82D/` with
2,835 samples and dataset SHA-256
`b3a8819a82c8d9c2d15ea2d8e0499ed390720899f7f7fd69f39d591cb9e4c0db`.
`artifacts/diagnostics/R0_V2_THREE_CLASS_FOLD_SMOKE_82D/result.json` records
the sole changed variable as historical V2 sample coverage. Its RankIC values
are -0.0532, -0.0034, and 0.0017. Two later folds have positive Top-K mean
excess return (7.00% and 3.29%), but the first is -3.69%, direction accuracy
is still zero in every fold, and the first training window has only 35 samples.
This is evidence of an inconclusive short-window diagnostic, not a promotion:
the artifacts remain `assembled_fail_closed`/`evaluated_fail_closed` and
`model_valid=false`.

After 122 date chunks were available, the assembler created
`artifacts/diagnostics/R0_V2_TENSOR_DATASET_2023_01_03_2024_01_22_122D/` with
4,597 samples and dataset SHA-256
`f9db89dc75d982ad689e14f55c675c9e1925e98a6e8153f18fa40e1c5a809f1b`.
The ordinary six-date-minimum smoke control is recorded in
`R0_V2_THREE_CLASS_FOLD_SMOKE_122D_MIN6/`; it again has a 35-sample first
training fold. On the **same frozen dataset**, the only changed experiment
variable was then `minimum_training_history_60_dates`, recorded in
`R0_V2_THREE_CLASS_FOLD_SMOKE_122D_MIN60/result.json`. Its training-fold
sizes are 1,723, 2,598, and 3,552 samples; RankIC is 0.0286, -0.0055, and
0.0132; and Top-K mean excess return is +3.23%, -3.55%, and -1.64%.
The larger history resolves the tiny-first-fold diagnostic defect but does not
produce consistently positive excess return. It is therefore an audited,
negative R0 result, not authorization to start R1.

Two independently logged date shards then added 40 more verified V2 chunks.
The refreshed coverage audit reports 222 of 856 event dates, 33,483 candidate
label records, 8,391 eligible labels at each horizon (75.18%), and explicit
dispositions of 2,901 missing-market-status, 3,444 new-stock-restricted, 1,434
price-limit-blocked, and 531 suspended labels. The corresponding 222-date
tensor dataset contains 8,391 samples with SHA-256
`33a3ff0cd32bf0ec4bfe9470441d8f85832e1de9455a62a95b7426184f028214`.
Keeping the 60-date minimum training history fixed, the sole changed variable
in `R0_V2_THREE_CLASS_FOLD_SMOKE_222D_MIN60/result.json` is historical V2
coverage. Its three Top-K mean excess returns are -0.65%, -2.43%, and -2.18%;
RankIC is 0.0010, -0.0138, and 0.0271. This is a decisive fail-closed result:
larger historical coverage did not make the diagnostic strategy consistently
profitable, so graph-model promotion remains prohibited pending R1 baselines.

The two year-end shards expanded the audited coverage to 257 dates and 39,591
candidate label records; 10,019 labels are eligible at each horizon (75.92%).
Their fully verified tensor dataset has 10,019 samples and SHA-256
`2eb3ec7ce3dfc4fbfb7f9b07fafde59d089e984370e9df129a3b97fcd1549e59`.
With the same 60-date minimum training history, the sole changed variable in
`R0_V2_THREE_CLASS_FOLD_SMOKE_257D_MIN60/result.json` is historical coverage.
Its Top-K mean excess returns are -1.75%, -1.02%, and -2.34%, with RankIC
0.0057, -0.0217, and 0.0395. The positive last-fold RankIC cannot outweigh
negative Top-K performance in every fold; this remains `evaluated_fail_closed`
and does not permit R1.

Two non-overlapping 2024 Q1 shards then raised coverage to 297 dates and
49,545 candidate label records, with 12,688 eligible labels at every horizon
(76.83%). The verified 297-date tensor dataset contains 12,688 samples and
has SHA-256 `cc4b8928f3106662b747244a9711936509448e007585861056d4ff597ffce92d`.
Keeping the 60-date minimum training history fixed, the only changed variable
in `R0_V2_THREE_CLASS_FOLD_SMOKE_297D_MIN60/result.json` is historical V2
coverage. Top-K mean excess return is -1.75%, -2.79%, and +5.20%; RankIC is
0.0070, 0.0004, and 0.0281. Two negative folds, together with near-zero early
RankIC, are insufficient for a reproducible signal claim. The result remains
`evaluated_fail_closed`, `model_valid=false`, and graph-model promotion is prohibited.

## Coverage audit and remaining materialization gap

The verified historical backfill contains 856 common daily-bar and seat-event
snapshot dates, from 2023-01-03 through 2026-07-17. Two subsequent validated
batches added 2023-03-02, 03, 06, 07, 08 (143 full-horizon tensor samples),
2023-03-09, 10, 13, 14, 15 (144 samples), 20 dates from 2023-03-16 through
2023-04-13 (802 samples), and 20 dates from 2023-04-14 through 2023-05-16
(946 samples), and 20 dates from 2023-05-17 through 2023-06-13 (816 samples).
An additional batch from 2023-06-14 through 2023-07-13 supplied 829 samples,
and later batches from 2023-07-14 through 2023-08-10 (695 samples) and
2023-08-11 through 2023-09-07 (728 samples), followed by two non-overlapping
shards from 2023-09-08 through 2023-11-10, then 2023-11-13 through
2023-12-29, and two 2024 Q1 shards through 2024-03-26. 297 dates are now
materialized and 559 verified source dates remain outside the V2 dataset. The
52-, 82-, 122-, 222-, 257-, and 297-date datasets are frozen inputs for their
respective smoke experiments; no later chunks were retroactively added to
them. The current slice also contains a large 2023-to-2024 coverage gap, so it
is not a contiguous multi-regime backtest. Continue materializing the remaining
verified dates in bounded, resumable batches before making any cross-regime
claim or advancing to R1.

## 320-date concurrent-materialization snapshot

While two later, non-overlapping materialization shards were still running, a
manifest-only scan froze the 320 chunks that had already completed into
`artifacts/diagnostics/R0_V2_TENSOR_DATASET_320D_SNAPSHOT/`. The snapshot has
13,866 samples, dataset SHA-256
`fc629a02949734970da8ecd423c1bac198d542cd049283e3b3da11b16746bb5b`, and an
`assembled_fail_closed` manifest with `model_valid=false`. Chunks that finished
after its manifest scan are intentionally excluded, so this is a reproducible
snapshot rather than a moving target.

Keeping the three-class model, seed, one epoch, and 60-date minimum training
history unchanged, `R0_V2_THREE_CLASS_FOLD_SMOKE_320D_MIN60/result.json`
changes only `historical_v2_sample_coverage_320_dates`. Its Top-K mean excess
returns are -1.75%, -2.93%, and -0.44%; RankIC is 0.0091, -0.0024, and
0.0203. All folds have negative Top-K excess return, so the additional data
does not establish a signal. The result is `evaluated_fail_closed` with
`model_valid=false`; it does not authorize R1. A fresh coverage audit is
required only after the two in-flight shards finish, because the live artifact
directory has since advanced beyond this frozen 320-date input.

## 333-date audited snapshot

After both shards completed, `R0_V2_LABEL_COVERAGE_AUDIT/result.json` recorded
333 materialized dates out of 856 verified event dates, 57,345 candidate label
records, and a 76.48% eligible-label rate at every horizon. The corresponding
M0 contract audit remains `audited_fail_closed`: StableSeat authorization and
industry/concept coverage are unavailable, and 523 verified event dates remain
outside the V2 dataset. This is evidence of a bounded, traceable
`InstitutionFlowEvent` dataset, not proof of a stable-seat graph.

`artifacts/diagnostics/R0_V2_TENSOR_DATASET_333D/` freezes the completed
chunks into 14,620 samples with dataset SHA-256
`297b5b09368f02a896d091963b68a0e38026553ceb7d9585131613da69fdc47d` and
manifest SHA-256 `d6bc885ebe6806771baced2b8626a151a716c60dd480ebb71f7782a0e42c43d1`.
With the same fixed three-class smoke configuration and 60-date training
minimum, `R0_V2_THREE_CLASS_FOLD_SMOKE_333D_MIN60/result.json` changes only
`historical_v2_sample_coverage_333_dates`. The three RankIC values are 0.0096,
0.0037, and 0.0192; Top-K mean excess returns are -1.75%, -3.26%, and -2.49%.
Every fold is negative on the executable Top-K metric, so the experiment is
`evaluated_fail_closed`, `model_valid=false`, and graph-model promotion remains prohibited.

## 373-date audited snapshot

Two further non-overlapping, resume-safe shards (2024-05-23 through
2024-06-20 and 2024-06-21 through 2024-07-18) completed with 20 date chunks
each and empty error logs. The refreshed coverage audit records 373 of 856
verified event dates, 64,872 candidate records, 75.90% horizon eligibility,
and 483 dates not yet materialized. The M0 contract still reports four
fail-closed blockers, including incomplete coverage and the absence of
StableSeat authorization.

`artifacts/diagnostics/R0_V2_TENSOR_DATASET_373D/` contains 16,413 samples,
dataset SHA-256 `c6c678bb06e62bdeb8ff86ea6f94a30f5fda66232249ce32a7e8e37ec0f55030`,
and manifest SHA-256 `59c5b50a87cb62a9f56a7151860daad9374cb90bc1adb7c05524a65e28d5516a`.
`R0_V2_THREE_CLASS_FOLD_SMOKE_373D_MIN60/result.json` keeps every experimental
control fixed and changes only `historical_v2_sample_coverage_373_dates`.
RankIC is 0.0057, 0.0221, and 0.0167; Top-K mean excess returns are -1.75%,
+5.66%, and -3.05%. The isolated positive middle fold is contradicted by two
negative folds, so this is not reproducible signal evidence. It remains
`evaluated_fail_closed`, `model_valid=false`; graph-model promotion is still prohibited.

## 453-date audited snapshot and batch-attempt trace

Four further bounded date shards completed after a local `PYTHONPATH`
propagation defect was fixed in the batch runner. The original two failed
attempts remain immutable stderr evidence; the new
`R0_V2_BATCH_ATTEMPT_AUDIT/result.json` records their hashes and explicitly
classifies nonempty stderr as `recorded_failure`. The two retry shards and two
later shards each completed 20 dates with empty retry stderr logs. This is an
operational recovery record, not indirect evidence of data or model validity.

The refreshed V2 coverage audit now contains 453 of 856 verified event dates,
82,011 candidate label records, a 75.95% eligible-label rate, and 403 dates
outside the materialized V2 set. The M0 contract audit remains
`audited_fail_closed` with four blockers. The frozen
`R0_V2_TENSOR_DATASET_453D/` has 20,762 samples, dataset SHA-256
`0ab54bc558c76a87051f5a477d9a68d6db3e857cf226b38f31b716680dec6ca4`, and
manifest SHA-256 `7b67e15d5c8ca5ca11fe29e1a1d4c1fc4fe3699e46ef9cc5c1df6335a2db476f`.

Keeping the fixed three-class smoke configuration and 60-date training minimum,
`R0_V2_THREE_CLASS_FOLD_SMOKE_453D_MIN60/result.json` changes only
`historical_v2_sample_coverage_453_dates`. RankIC is -0.0006, 0.0287, and
0.0070; executable Top-K mean excess return is -1.12%, -3.09%, and +0.78%.
The final fold alone is insufficient against two negative folds and a negative
early RankIC. The experiment remains `evaluated_fail_closed`,
`model_valid=false`, and graph-model promotion remains prohibited.

## 533-date audited snapshot

Four more non-overlapping 20-date shards completed with empty stderr logs.
The V2 coverage audit records 533 of 856 verified event dates, 99,045 candidate
records, 76.80% horizon eligibility, and 323 unmaterialized dates. The M0
contract remains `audited_fail_closed` with the same four blockers.

`R0_V2_TENSOR_DATASET_533D/` contains 25,357 samples, dataset SHA-256
`0b1a667829d3fde3ea0ba8b291d07c5a7a8a0a5385aaf5617c90f66db51df0a2`, and
manifest SHA-256 `ae0c4871b7844961b4a6e81af7681195f4f48be9c4f3ee1b9db90d29c5eb1068`.
`R0_V2_THREE_CLASS_FOLD_SMOKE_533D_MIN60/result.json` is a complete
`evaluated_fail_closed` artifact with three checkpoints; it changes only
`historical_v2_sample_coverage_533_dates`. Its RankIC values are 0.0044,
0.0226, and -0.0085; Top-K excess returns are -1.12%, -2.12%, and +3.11%.
One positive late fold cannot establish reproducible signal against two
negative Top-K folds and a negative late RankIC. It remains
`model_valid=false`; graph-model promotion is prohibited.

## 613-date audited snapshot

Four further 2025 shards completed 80 dates with empty stderr logs. Coverage
reached 613 of 856 verified event dates, with 115,392 candidate records,
77.63% horizon eligibility, and 243 dates still outside the V2 set. M0 remains
`audited_fail_closed` with four blockers.

`R0_V2_TENSOR_DATASET_613D/` freezes 29,861 samples with dataset SHA-256
`f20d9ab37fc9344398afac52e96c36284aeb3ed9867b1653bdef359a7a941ee7` and
manifest SHA-256 `76e6f119ac2eaa16aed79de22cfddd36c6c27073453eb6f1b95890d225675620`.
The 613-date fold smoke was run as a separately logged process to avoid the
interactive command time limit; its stdout, stderr, checkpoints, and complete
result artifact are retained. With only
`historical_v2_sample_coverage_613_dates` changed, RankIC is 0.0020, 0.0106,
and 0.0017, while Top-K excess return is -1.12%, -3.05%, and +3.23%. Two
negative folds and near-zero RankIC across all folds remain insufficient for a
reproducible-signal claim. The artifact stays `evaluated_fail_closed` and
`model_valid=false`; graph-model promotion remains prohibited.

## 693-date audited snapshot

The next four 2025 shards completed with empty stderr logs, raising coverage
to 693 of 856 verified event dates (131,070 candidate records and 78.80%
horizon eligibility). 163 dates remain; the M0 data contract therefore stays
`audited_fail_closed` with four blockers.

`R0_V2_TENSOR_DATASET_693D/` contains 34,426 samples with dataset SHA-256
`68801724d3476ef59a6a5120e41cdb687d962469f5175337e5bb4ae170e17afc` and
manifest SHA-256 `be97b717f51cb7d27e5316e7980551c85668187539ca54e4e80115e9a58c1183`.
The separately logged 693-date fold run changes only
`historical_v2_sample_coverage_693_dates`. RankIC is 0.0101, 0.0046, and
-0.0181; Top-K excess return is -1.12%, +0.67%, and +1.96%. Despite two later
positive Top-K values, the first fold loses money and the final RankIC is
negative. These three folds do not establish reproducible signal; the result
remains `evaluated_fail_closed` with `model_valid=false`, and graph-model promotion is prohibited.

## 773-date audited snapshot and recovered batch output

The recovered retry shard and three adjacent 2025–2026 shards produced a
773-date V2 set. The original retry's child had already written a valid tensor
manifest before its detached stdout failed; the coverage audit therefore
correctly counts 773 (not the 772 estimated from batch progress alone). The
stdout defect is preserved in the failed-attempt audit, while the dataset is
accepted only because its manifest/hash validation succeeds. Coverage is now
773 of 856 dates, with 147,282 candidate records, 79.29% eligibility, and 83
dates remaining.

The directory name `R0_V2_TENSOR_DATASET_772D/` is historical only; its
manifest records the authoritative 773-date count, 38,925 samples, dataset
SHA-256 `78344d5294556c562fc0275351220430ca487afa5e1d62999837bd3251f2b305`,
and manifest SHA-256 `1d0be539c18887b2e13c24101907f4be9d461e9a4624b299bf9ee1d25a248f7d`.
The separately logged fold artifact changes only
`historical_v2_sample_coverage_773_dates`. RankIC is 0.0165, 0.0023, and
-0.0237; Top-K excess return is -0.84%, +0.61%, and +1.51%. The negative
first-fold return and negative final-fold RankIC reject a reproducible-signal
claim. It remains `evaluated_fail_closed`, `model_valid=false`, and R1 stays
prohibited.

## 856-date complete V2 coverage, schema migration, and final R0 gate

The label-coverage audit now proves that all 856 verified event dates from
2023-01-03 through 2026-07-17 have a source-hashed
`executable_excess_return_v2` artifact; `missing_event_date_count=0`.
It records 167,286 candidate labels, 132,494 eligible labels, and explicit
non-eligible dispositions rather than substituted values: 10,689
`missing_market_status`, 8,796 `new_stock_restricted`, 11,841
`price_limit_blocked`, 2,652 `suspended`, 9 `st_restricted`, and 805
`unobserved_horizon`. The latter prevents any price or status after the frozen
2026-07-17 observation boundary from entering a training target.

`R0_V2_TENSOR_DATASET_856D/manifest.json` migrated the downstream dataset
contract to record every input chunk: 856 `source_as_of_dates`, 850
`training_as_of_dates`, six `zero_sample_as_of_dates`, 44,033 complete
1/3/5-day tensor samples, dataset SHA-256
`c6fe493b6169e63cb0a8b8ef29d073b4142c97a7d97f1d120cc5282a66544fc9`, and
manifest SHA-256 `845c092519cc952f810257fb94c38e77eaf08f306089544be1c5e94d2c93660d`.
The six excluded dates retain label and chunk manifests with zero tensors; they
are not silently dropped and cannot become partial-horizon samples.

With only `historical_v2_sample_coverage_856_dates` changed,
`R0_V2_THREE_CLASS_FOLD_SMOKE_856D_MIN60/result.json` is
`evaluated_fail_closed`, `model_valid=false`. Fold RankIC is 0.0158, 0.0073,
and -0.0423, while executable Top-K excess return is -0.82%, +1.56%, and
-1.32%. This is not reproducibly positive (two of three folds lose money and
the final RankIC is negative), so graph-model and paper-trading gates remain closed. R1
strong baselines remain a required diagnostic stage. The M0
contract audit also remains `audited_fail_closed`: only
`InstitutionFlowEvent` is authorized, while StableSeat authorization and
industry/concept coverage are absent. No R1 baseline, graph expansion, paper
trading, or live trading is permitted from this result.

## R1: five-seed V2 tabular baseline gate

`R1_V2_TABULAR_BASELINES_856D_5SEED/result.json` completes the required
five-seed, three-fold V2 table-baseline diagnostic on the same 856-source-date
dataset, 60-date minimum train window, and executable three-class label schema.
It contains 45 independently saved checkpoints (three available models × five
seeds × three folds), fold-local scaler state and fitted-sample hashes. Its
result SHA-256 is
`ab9a5f5a7dd2fee54a88f33c426f55126a2f3bbe042259431655cb7af138fe8b`.

All 15 runs of every available baseline have negative daily Top-K excess
return: Logistic Regression averages -0.5462% (worst -0.6531%), MLP averages
-0.4116% (worst -0.7212%), and LightGBM averages -0.4230% (worst -0.7006%).
Their mean daily RankIC values are respectively 0.0240, 0.0263, and 0.0294,
but no positive Top-K run exists for any model. CatBoost is recorded as
`dependency_unavailable`; it was neither silently omitted nor substituted.

The R1 artifact is therefore `evaluated_fail_closed`, `model_valid=false`.
Together with the M0 data-contract blockers (no StableSeat authorization and
no industry/concept coverage), this is affirmative no-stable-signal evidence
under the current authorized data scope. R2 graph research, R3 graph semantics,
paper trading, and production promotion remain forbidden. The appropriate
outcome is to preserve these artifacts and return to data acquisition/coverage
repair rather than alter thresholds or add graph complexity.
