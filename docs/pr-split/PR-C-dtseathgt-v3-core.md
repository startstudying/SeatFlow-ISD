# PR-C: DT-SeatHGT v3 Core

> Branch: `feat/dtseathgt-v3-core`
> Scope: `dt_seat_hgt_v3.py`, typed edge-gated relation, three-stage trainer, loss functions, E0/E1/E3 experiment matrix, core explainer, architecture + trainer tests. **No M7–M12, no R6, no checkpoints, no results directories, no placeholder findings.**

## What this PR contains

### Core model (`src/seatflow/models/dt_seat_hgt_v3.py`, 926 lines)

- `DTSeatHGTv3Config` frozen dataclass with stock / seat / relation / head hyperparameters
- `StockTemporalEncoder` (GRU over 20-day sequences)
- `StockStaticEncoder` (MLP over 8 static features)
- `BaseFusion` → `base_embedding`
- `BaseRelationBranch` (Industry→Stock, Stock→Stock)
- `SeatResidualBranch` (Seat→Stock buys/sells only — the residual the model learns on top of the volume-price baseline)
- `RelationRouter` (typed edge-gated relation with `seat_gate`)
- Multi-task heads: Rank, Return, Direction
- Uncertainty head

### Seat style / persona + persistent memory (`src/seatflow/models/`, `src/seatflow/memory/`)

- `seat_style_encoder.py` — `SeatStyleEncoder` for the seat soft-persona probability profile (trend / reversal / event-driven / institution-following / short-term-high-turnover)
- `dt_seat_hgt_memory.py` (729 lines) — temporal memory wiring; `use_seat_style_encoder` flag toggles E3 (off) vs E4 (on)
- `memory/industry_memory.py`, `memory/seat_memory.py`, `memory/stock_memory.py` — persistent seat state with trading-day half-life decay (the "持久席位状态" priority)

### Three-stage trainer + losses (`src/seatflow/training/`)

- `dt_seat_hgt_v3_losses.py` (329 lines) — `DTSeatHGTv3LossWeights` frozen dataclass: return SmoothL1, ranking pairwise, direction, uncertainty NLL, gate sparsity, style regularization (conditional entropy, marginal balance, temporal consistency)
- `dtseathgt_v3.py` (846 lines) — three-stage progressive unfreezing trainer (stage 1 base, stage 2 relations, stage 3 heads)
- `v4_loss.py` (338 lines) — `V4Loss` consumed by the trainer

### Typed edge-gated relation (`src/seatflow/models/layers/`)

- `edge_gated_relation.py` — `BaseRelationBranch`, `SeatResidualBranch`, `TypedEdgeGatedRelationLayer` (the typed edge-gated relation with `seat_gate`)

### Data adapter (`src/seatflow/datasets/`)

- `stock_sequences.py` — `SEQUENCE_FEATURES` + `extract_stock_sequences` (consumed by architecture tests)

### Core explainer (`src/seatflow/explain/`)

- `relation_contribution.py` — relation-level explanation: which Seat→Stock edges shifted the final score
- `critical_subgraph.py` — critical subgraph extraction

### E0/E1/E3 experiment matrix (`scripts/`)

- `run_e0_2x3.py`, `run_e0_full.py`, `run_e0_mini.py` — E0: stock volume-price GRU/MLP baseline
- `run_e1_2x3.py`, `run_e1_mini.py` — E1: stock + industry/stock relations
- `run_e3_2x3.py`, `run_e3_mini.py` — E3: stock + Seat→Stock residual graph

### Tests (72/73 passing)

- `tests/models/test_dt_seat_hgt_v3.py`
- `tests/models/test_dt_seat_hgt_v3_architecture.py`
- `tests/models/test_edge_gated_relation.py`
- `tests/test_dtseathgt_v3_training.py`

The single failure, `test_edge_gate_modulates_attention`, is a pre-existing regression from v4 commit `5c74277` (tracked in memory as `[[pre-existing-failing-test-edge-gate]]`, LOW priority, not on the gate path) and is not introduced by PR-C.

## What this PR does NOT contain

- No M7–M12 large-scale enhancements (deferred — not part of the model core)
- No R6 modules (deferred — no completed control experiment yet; M12 doc still has `待实验后填写` and `[True/False]` placeholders, so it is framework only, not a finished research result)
- No checkpoints (`*.pt`), no tensor chunks (`*.pkl`), no results directories, no Parquet artifacts
- No paper-trading business tables (that is PR-D)
- No CI / Docker / dependency changes (that is PR-A)

## Known follow-ups (tracked, not in this PR)

- E1 currently equals E0 because industry nodes are not populated. This is the [[phase2-industry-data-available-not-wired]] issue — sw2021 industry membership is backfilled but the seat-event industry field is empty. E1 may not be claimed valid until industry nodes are populated. This is Week-5 model-verification work, not PR-C.
- E3 seat branch Train=NaN on raw monetary seat features (~1e10) — partial LayerNorm fix in place, data-side normalization still needed. See [[e3-seat-branch-train-nan]].

## Merge note

The model code on this branch was previously bundled into the 367-file PR #4.
It is isolated here so the trunk can review the DT-SeatHGT v3 architecture and
its tests without also reviewing CI hygiene, PIT infrastructure, or paper trading.
