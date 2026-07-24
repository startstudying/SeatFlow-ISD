# R2-SeatRankLite experiment report

Status date: 2026-07-23

## Current status

- Engineering implementation: complete through Task 21
- CUDA engineering smoke: passed
- Development 3 folds × 5 seeds: not run
- `development_candidate = false`
- `model_valid = false`
- Future-blind entered: no

No historical 15-run return claim is made because the full dataset experiment
has not been executed.

## GPU smoke evidence

Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB.  
PyTorch: 2.11.0+cu128.  
Mode: CUDA FP16 autocast, FP32 loss, GradScaler, one graph at a time.

| Seed | Epochs | Parameters | Peak allocated | Peak reserved | Losses |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026 | 2 | 171,316 | 19,952,128 B | 25,165,824 B | 0.486807, 0.362135 |
| 2027 | 2 | 171,316 | 19,952,128 B | 25,165,824 B | 0.553240, 0.518669 |

Both seed checkpoints contain CPU-only tensors. The second seed started and
completed without OOM. Peak reserved memory was about 24 MiB, far below the
7.4GB hard limit. This fixture is intentionally small and does not establish
the peak for a production-size snapshot.

## Graph smoke shape

- Stock nodes: 6
- Seat nodes: 6
- Industry nodes: 2
- Trade and reverse edges: 12
- Industry and reverse edges: 12
- Total edges: 24
- Seat cooccurrence: disabled
- Concept graph: disabled

## Development metrics

The following remain unavailable until the fixed 3-fold × 5-seed run:

- per-fold/per-seed RankIC;
- RankIC mean and standard deviation;
- Top-5 gross and net spread;
- turnover and market-regime tables;
- positive-return run count;
- fold 0/1/2 positive-seed counts;
- seed prediction correlations.

Therefore the 10/15 gate, two-positive-fold gate, regime-concentration checks,
and trading-cost improvement gate are not evaluated.

## Stop condition

Do not add Mamba, MoE, deeper HGT, learnable hypergraphs, or larger hidden
dimensions. If the fixed R2-Lite experiments fail, re-audit label alignment,
candidate-pool construction, trade timing, baseline portfolio, and transaction
costs before increasing model complexity.
