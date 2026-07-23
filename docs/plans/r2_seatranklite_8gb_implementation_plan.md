# SeatFlow R2-SeatRankLite: 8GB GPU implementation plan

This repository plan is frozen from the R2 request executed on 2026-07-23.
Tasks are ordered and must not use R1 outcomes for R2 selection.

1. Freeze R1 as contaminated research-only evidence.
2. Centralize device placement and CUDA accounting.
3. Add FP16 autocast and GradScaler.
4. Run seeds serially and save CPU-only state.
5. Remove numeric identity hashes.
6. Apply amount transforms exactly once with train-only scaling.
7. Enforce typed edge schemas.
8. Disable Seat–Seat cooccurrence for R2.
9. Add reverse trade and industry relations; disable concepts.
10. Aggregate duplicate snapshot edges.
11. Apply Stock–Stock top-k sparsity and 60k/100k edge limits.
12. Implement two-layer SeatRankLite with four prediction heads.
13. Restrict edge gates to `[edge_count, num_heads]`.
14. Enforce three-million recommended and five-million hard parameter limits.
15. Build candidate-pool residual return labels.
16. Add the fixed date-local composite ranking loss.
17. Defer optional edge consistency until after stable E4 evidence.
18. Freeze the 8GB experiment configuration.
19. Train one graph at a time on CUDA and verify serial seed release.
20. Compute transaction costs from portfolio weight changes.
21. Report predeclared market-regime metrics.

## Required gates

- Normal CUDA peak: at most 6.8GB.
- Hard CUDA peak: at most 7.4GB.
- Five fixed seeds: 2026–2030, equal-weight only.
- Development threshold: at least 10 positive runs of 15 and at least two
  positive fold means.
- `model_valid` remains false until a separately frozen future-blind period is
  unsealed.

The engineering smoke is not a return evaluation. Full 3-fold × 5-seed
development runs begin only after the complete test suite passes and suitable
historical data are available.
