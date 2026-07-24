# R3 Task 25 — SeatRankLite loss study

## Decision

Task 25 completed all 18 immutable runs. No objective passed the frozen
`research_protocol_r3.toml` inner-development gates. No objective is selected,
and Task 26 is not allowed.

The evaluator failed closed because the frozen R2 evidence does not contain the
RankIC median or seed-level net-spread standard deviation needed for the two
relative gates. Independently of those missing references, every objective also
failed the absolute requirements that at least four of six runs have positive
net spread and that both fold means are non-negative.

## Capacity and execution evidence

- CUDA preflight passed for the legalized P50, P90, and maximum production
  graphs without reading outcomes.
- Maximum preflight reserved memory was 83,886,080 bytes (0.078 GiB), below the
  7.2 GiB ceiling.
- The planned manifest contains exactly 3 objectives × 2 folds × 3 seeds.
- All 18 runs completed and their artifact hashes were verified before outcome
  evaluation.
- Every run saved daily predictions, Top-5 holdings, training history, resource
  usage, model state, metadata, and a run-completion manifest.
- Execution-only GPU scheduling changes and the mathematically equivalent
  LambdaRank synchronization optimization are recorded in
  `artifacts/r3_task25/execution_parallel_addendum.json`. They did not change
  the planned manifest, model parameters, objectives, folds, seeds, epochs, or
  pair sets.

## Aggregate results

| Objective | Median RankIC | Mean RankIC | Mean net Top-5 spread | Worst fold net spread | Positive net runs | Seed net-spread std | Top-5 Jaccard | Top-10 Jaccard | Rank 5/6 margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L0 | 0.007593 | 0.007341 | -0.000076 | -0.003286 | 3/6 | 0.001521 | 0.410380 | 0.467506 | 0.003076 |
| L1 | 0.022947 | 0.024281 | -0.001691 | -0.002972 | 1/6 | 0.000177 | 0.523672 | 0.607350 | 0.006397 |
| L2 | 0.033662 | 0.035500 | -0.002084 | -0.002147 | 0/6 | 0.000543 | 0.563502 | 0.717949 | 0.016719 |

L2 produced the strongest ranking and overlap measures, but its net Top-5
spread was negative in every run. Transaction costs and turnover therefore
prevent promotion despite better ranking quality.

## Fixed-gate result

| Gate | L0 | L1 | L2 |
| --- | --- | --- | --- |
| Exactly six runs | Pass | Pass | Pass |
| Peak GPU ≤ 7.2 GiB | Pass | Pass | Pass |
| At least four positive-net runs | Fail (3) | Fail (1) | Fail (0) |
| Both fold means non-negative | Fail | Fail | Fail |
| Median RankIC not below frozen R2 | Not evaluable | Not evaluable | Not evaluable |
| Seed net-spread std reduced ≥10% vs R2 | Not evaluable | Not evaluable | Not evaluable |

Selection status: `blocked_missing_r2_reference`.

Selected objective: none.

Task 26 allowed: false.

## A–F failure evidence

### A — Optimization randomness

Seed-level net-spread standard deviation was 0.001521 for L0, 0.000177 for L1,
and 0.000543 for L2. L1 was the most stable by this measure, but stability did
not translate into positive net spread.

### B — Top-5 boundary noise

Top-5 Jaccard improved from 0.410380 (L0) to 0.523672 (L1) and 0.563502 (L2).
Mean Rank 5/6 margin improved from 0.003076 to 0.006397 and 0.016719. Boundary
stability improved materially under L2, but it did not repair net performance.

### C — Market-regime drift

Performance varied substantially by regime. For example, L0 net spread was
0.003482 in high-activity observations and -0.003066 in low-activity
observations. L1 and L2 also remained negative in their low-activity groups.
The complete activity, candidate-dispersion, market-amount, volatility, and
market-direction breakdown is preserved in `evaluation.json`.

### D — Turnover cost

Aggregate net Top-5 spread was negative for every objective: -0.000076 (L0),
-0.001691 (L1), and -0.002084 (L2). Turnover was approximately 0.926, 0.953,
and 0.961 respectively. The stronger L1/L2 rankings did not cover the frozen
buy/sell costs.

### E — Seat-graph increment

Status: insufficient evidence. The immutable 18-run manifest did not predeclare
an inference ablation capable of isolating the seat graph's incremental value.
No post-hoc ablation was added.

### F — Label/objective mismatch

Status: not established. Mean RankIC rose from 0.007341 (L0) to 0.024281 (L1)
and 0.035500 (L2), while mean net spread became more negative. This is
consistent with a possible ranking-to-trading mismatch, but the present study
does not isolate label mismatch causally.

## Stop condition

Work stops before Task 26. No GRU, FiLM, seat profile, hypergraph, or other
prohibited architecture is implemented automatically.
