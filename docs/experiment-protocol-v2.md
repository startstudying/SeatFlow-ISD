# SeatFlow Experiment Protocol v2

## Purpose

This protocol separates development evaluation from a future blind evaluation for
the institutional-seat proxy research track. It does not make a complete-seat or
trading-validity claim.

## Frozen windows

| Window | Dates | Labels | Status | Permitted use |
| --- | --- | --- | --- | --- |
| train | 2023-02-06 to 2024-06-30 | observed | development | fit model parameters |
| valid | 2024-07-01 to 2024-12-31 | observed | development | choose frozen settings |
| dev_oos | 2025-01-01 to 2025-12-31 | observed | development OOS | report once per registered run |
| contaminated_oos | 2026-01-01 to 2026-07-17 | observed | contaminated OOS | frozen-model robustness report only |
| future_blind | 2026-07-20 to 2026-12-31 | unreleased | future blind | independent evaluator only |

`contaminated_oos` is not a blind set because its labels have already been
viewed. It must not drive feature selection, hyperparameter selection, threshold
selection, or model-family selection.

## Leakage controls

1. Each 1/3/5-day label begins at `t+1`; train/valid/dev partitions use the
   protocol purge and embargo before a later partition is evaluated.
2. A seat profile at `t` may only use historical outcomes whose full horizon
   matured before `t`.
3. Candidate and feature filtering may use only data known at `t`. Realized
   `t+1` price limits, suspensions, and missing bars belong to execution
   simulation, not pre-trade filtering.
4. `future_blind` labels are sealed. Training code must call the protocol label
   access guard before loading labels for any named evaluation window.

## Required metadata

Every new report and lineage record must include `data_scope`,
`seat_detail_scope`, `label_status`, `blind_status`, `model_family`, and hashes
for data, config, code, seed, split, checkpoint, and protocol where applicable.

## Scope limitation

The current data route is `institutional_proxy`, based on Tushare `top_inst`.
It cannot support a complete brokerage-seat or Northbound-seat claim. The main
task is ranking same-day Dragon-Tiger candidates by 1/3/5-day excess return, not
market-wide stock selection or future seat-stock link prediction.
