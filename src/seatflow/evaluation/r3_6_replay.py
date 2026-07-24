"""Frozen, training-free R3.6 cost-aware SeatRank replay."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from seatflow.evaluation.cost_aware_strategy import (
    StrategyTransition,
    simulate_p0,
    simulate_p1,
    simulate_p2,
    simulate_p3,
)
from seatflow.evaluation.frozen_artifacts import (
    FrozenTask25Store,
    build_frozen_manifest,
    sha256_file,
    verify_frozen_manifest,
)
from seatflow.evaluation.r3_5_audit import _join_predictions, _truth_rows
from seatflow.evaluation.realistic_costs import RealisticCost, split_turnover_costs
from seatflow.evaluation.return_units import (
    ReturnAggregationMethod,
    return_units,
)
from seatflow.evaluation.strategy_gate import (
    evaluate_strategy_gate,
    select_unique_strategy,
)
from seatflow.evaluation.walk_forward_calibration import (
    ridge_walk_forward_expected_returns,
)

PARENT_COMMIT = "0e2ccfb"
SIGNALS = {"L0": "S0", "L2": "S1"}
STRATEGIES = ("P0", "P1", "P2", "P3")
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 5.0)


def run_r3_6_replay(
    repository_root: str | Path,
    samples: Sequence[Any],
) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    output_root = repository / "artifacts" / "r3_6"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_count_before = _checkpoint_count(repository)

    frozen = build_frozen_manifest(repository, source_commit=PARENT_COMMIT)
    verify_frozen_manifest(repository, frozen)
    frozen_payload = _freeze_r3_5_evidence(repository, frozen)
    _write_json(output_root / "r3_5_frozen_manifest.json", frozen_payload)
    store = FrozenTask25Store(repository, frozen)
    run_manifest = _mapping(store.read_json("run_manifest.json"))
    truth, _ = _truth_rows(samples)

    run_results: list[dict[str, object]] = []
    daily_results: list[dict[str, object]] = []
    for run in _mapping_sequence(run_manifest["run_order"]):
        objective = str(run["objective"])
        if objective not in SIGNALS:
            continue
        relative = Path(str(run["output_directory"])) / "daily_predictions.jsonl"
        joined = _join_predictions(_mapping_sequence(store.read_json(relative)), truth)
        calibrated = ridge_walk_forward_expected_returns(joined, alpha=1.0)
        expected = {
            (row.prediction_date, row.stock_id): row.expected_gross_return
            for row in calibrated
        }
        rows = [
            {
                **row,
                "expected_return": expected[
                    (str(row["prediction_date"]), str(row["stock_id"]))
                ],
            }
            for row in joined
        ]
        grouped = _group_by_date(rows)
        transitions = _strategy_transitions(grouped)
        for strategy in STRATEGIES:
            for multiplier in COST_MULTIPLIERS:
                summary, daily = _evaluate_path(
                    grouped,
                    transitions[strategy],
                    cost_multiplier=multiplier,
                )
                identity = f"{SIGNALS[objective]}/{strategy}"
                run_result = {
                    "strategy_id": identity,
                    "signal": SIGNALS[objective],
                    "objective": objective,
                    "strategy": strategy,
                    "run_id": str(run["run_id"]),
                    "fold": int(cast("int", run["fold"])),
                    "seed": int(cast("int", run["seed"])),
                    "cost_multiplier": multiplier,
                    **summary,
                }
                run_results.append(run_result)
                daily_results.extend(
                    {
                        "strategy_id": identity,
                        "run_id": str(run["run_id"]),
                        "fold": run["fold"],
                        "seed": run["seed"],
                        "cost_multiplier": multiplier,
                        **row,
                    }
                    for row in daily
                )

    gate_evaluations: dict[str, dict[str, object]] = {}
    for signal in SIGNALS.values():
        baseline = _gate_rows(run_results, signal, "P0")
        for strategy in ("P1", "P2", "P3"):
            identity = f"{signal}/{strategy}"
            gate_evaluations[identity] = evaluate_strategy_gate(
                _gate_rows(run_results, signal, strategy),
                baseline,
            )
    selection = select_unique_strategy(gate_evaluations)
    checkpoint_count_after = _checkpoint_count(repository)
    if checkpoint_count_after != checkpoint_count_before:
        raise RuntimeError("R3.6 strategy replay created or removed a checkpoint")
    selection.update(
        {
            "gate_cost_multiplier": 1.0,
            "checkpoint_count_before": checkpoint_count_before,
            "checkpoint_count_after": checkpoint_count_after,
            "model_valid_eligible": False,
            "independent_validation": False,
        }
    )
    payload = {
        "schema_version": "r3_6_cost_aware_replay.v1",
        "frozen_parent_commit": PARENT_COMMIT,
        "development_only": True,
        "candidate_exit_rule": "A",
        "candidate_exit_rule_b": {
            "status": "unavailable",
            "reason": "frozen predictions contain no out-of-pool scores",
        },
        "run_count": len(run_results),
        "expected_run_count": 2 * 6 * 4 * 4,
        "runs": run_results,
        "gate_evaluations": gate_evaluations,
        "selection": selection,
    }
    if len(run_results) != 192:
        raise RuntimeError(f"expected 192 replay results, got {len(run_results)}")
    _write_json(output_root / "strategy_replay.json", payload)
    _write_json(
        output_root / "strategy_daily_results.json",
        {
            "schema_version": "r3_6_strategy_daily.v1",
            "rows": daily_results,
        },
    )
    _write_json(output_root / "strategy_gate_decision.json", selection)
    verify_frozen_manifest(repository, frozen)
    return payload


def _strategy_transitions(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, tuple[StrategyTransition, ...]]:
    selection_cost = RealisticCost()
    return {
        "P0": simulate_p0(grouped),
        "P1": simulate_p1(grouped, candidate_exit_rule="A"),
        "P2": simulate_p2(grouped, buy_rank=5, hold_rank=10),
        "P3": simulate_p3(
            grouped,
            buy_rank=5,
            hold_rank=10,
            switch_margin=0.0,
            cost=selection_cost,
        ),
    }


def _evaluate_path(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    transitions: Sequence[StrategyTransition],
    *,
    cost_multiplier: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cost = RealisticCost(
        commission=3.0 * cost_multiplier,
        stamp=5.0 * cost_multiplier,
        buy_slip=5.0 * cost_multiplier,
        sell_slip=10.0 * cost_multiplier,
        min_commission=5.0 * cost_multiplier,
        position_notional=100_000.0,
    )
    previous_candidate: dict[str, float] = {}
    previous_baseline: dict[str, float] = {}
    daily: list[dict[str, object]] = []
    cumulative_net = 0.0
    peak_net = 0.0
    max_drawdown = 0.0
    for transition in transitions:
        rows = grouped[transition.prediction_date]
        row_by_stock = {str(row["stock_id"]): row for row in rows}
        selected_ids = tuple(
            stock for stock in transition.candidate if stock in row_by_stock
        )
        selected_set = set(selected_ids)
        baseline_ids = tuple(
            sorted(stock for stock in row_by_stock if stock not in selected_set)
        )
        candidate_weights = _equal_weights(selected_ids)
        baseline_weights = _equal_weights(baseline_ids)
        candidate_turnover = _turnover(candidate_weights, previous_candidate)
        baseline_turnover = _turnover(baseline_weights, previous_baseline)
        candidate_cost = split_turnover_costs(
            buy_turnover=candidate_turnover[0],
            sell_turnover=candidate_turnover[1],
            cost=cost,
        )
        baseline_cost = split_turnover_costs(
            buy_turnover=baseline_turnover[0],
            sell_turnover=baseline_turnover[1],
            cost=cost,
        )
        candidate_cost_return = candidate_cost.total / cost.position_notional
        baseline_cost_return = baseline_cost.total / cost.position_notional
        candidate_return = _mean_return(selected_ids, row_by_stock)
        baseline_return = _mean_return(baseline_ids, row_by_stock)
        gross = candidate_return - baseline_return
        cost_diff = candidate_cost_return - baseline_cost_return
        net = gross - cost_diff
        if abs(net - (gross - candidate_cost_return + baseline_cost_return)) > 1e-15:
            raise AssertionError("gross-cost-net identity failed")
        cumulative_net += net
        peak_net = max(peak_net, cumulative_net)
        max_drawdown = min(max_drawdown, cumulative_net - peak_net)
        candidate_return_units = return_units([candidate_return]).to_dict()
        baseline_return_units = return_units([baseline_return]).to_dict()
        daily_gross_units = return_units([gross]).to_dict()
        daily_cost_units = return_units([cost_diff]).to_dict()
        daily_net_units = return_units([net]).to_dict()
        daily.append(
            {
                "prediction_date": transition.prediction_date,
                "holdings": list(selected_ids),
                "candidate_return": candidate_return,
                "candidate_return_units": candidate_return_units,
                "baseline_return": baseline_return,
                "baseline_return_units": baseline_return_units,
                "gross_spread": gross,
                "gross_spread_units": daily_gross_units,
                "candidate_buy_turnover": candidate_turnover[0],
                "candidate_sell_turnover": candidate_turnover[1],
                "candidate_turnover": 0.5
                * (candidate_turnover[0] + candidate_turnover[1]),
                "baseline_buy_turnover": baseline_turnover[0],
                "baseline_sell_turnover": baseline_turnover[1],
                "candidate_cost_components": asdict(candidate_cost),
                "baseline_cost_components": asdict(baseline_cost),
                "candidate_cost_return": candidate_cost_return,
                "baseline_cost_return": baseline_cost_return,
                "cost_difference": cost_diff,
                "cost_difference_units": daily_cost_units,
                "net_spread": net,
                "net_spread_units": daily_net_units,
                "selection_reason": transition.reason,
            }
        )
        previous_candidate = candidate_weights
        previous_baseline = baseline_weights
    gross_values = [_as_float(row["gross_spread"]) for row in daily]
    net_values = [_as_float(row["net_spread"]) for row in daily]
    cost_values = [_as_float(row["cost_difference"]) for row in daily]
    turnover_values = [_as_float(row["candidate_turnover"]) for row in daily]
    gross_units = return_units(
        gross_values,
        aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
    )
    net_units = return_units(
        net_values,
        aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
    )
    cost_units = return_units(
        cost_values,
        aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
    )
    worst_units = return_units(
        [min(net_values)],
        aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
        number_of_days=len(daily),
    )
    drawdown_units = return_units(
        [max_drawdown],
        aggregation_method=ReturnAggregationMethod.SUM,
        number_of_days=len(daily),
    )
    return (
        {
            "mean_gross_spread": gross_units.decimal,
            "mean_gross_spread_units": gross_units.to_dict(),
            "mean_net_spread": net_units.decimal,
            "mean_net_spread_units": net_units.to_dict(),
            "mean_cost_diff": fmean(cost_values),
            "mean_cost_diff_units": cost_units.to_dict(),
            "mean_turnover": fmean(turnover_values),
            "positive_net_date_ratio": fmean(value > 0 for value in net_values),
            "worst_daily_net_spread": min(net_values),
            "worst_daily_net_spread_units": worst_units.to_dict(),
            "max_additive_drawdown": max_drawdown,
            "max_additive_drawdown_units": drawdown_units.to_dict(),
            "day_count": len(daily),
        },
        daily,
    )


def _gate_rows(
    run_results: Sequence[Mapping[str, object]],
    signal: str,
    strategy: str,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in run_results
        if row["signal"] == signal
        and row["strategy"] == strategy
        and float(cast("float", row["cost_multiplier"])) == 1.0
    ]


def _freeze_r3_5_evidence(
    repository: Path,
    task25_manifest: Mapping[str, object],
) -> dict[str, object]:
    paths = [
        *sorted(
            path.relative_to(repository).as_posix()
            for path in (repository / "artifacts" / "r3_5").glob("*")
            if path.is_file()
        ),
        "artifacts/r3_task25/frozen_manifest.json",
        "docs/plans/r3_5_economic_alignment_and_top5_tail_failure_audit_plan.md",
        "docs/findings/r3_5_economic_alignment_and_top5_tail_failure_audit.md",
    ]
    records = []
    for relative in paths:
        path = repository / relative
        if not path.is_file():
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "r3_6_r3_5_frozen_manifest.v1",
        "source_commit": PARENT_COMMIT,
        "task25_manifest_schema_version": task25_manifest["schema_version"],
        "task25_checkpoint_count": task25_manifest["checkpoint_count"],
        "task25_run_file_count": task25_manifest["run_file_count"],
        "r3_5_files": records,
    }


def _group_by_date(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    result: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        result.setdefault(str(row["prediction_date"]), []).append(row)
    return result


def _equal_weights(stocks: Sequence[str]) -> dict[str, float]:
    if not stocks:
        return {}
    weight = 1.0 / len(stocks)
    return {stock: weight for stock in stocks}


def _turnover(
    current: Mapping[str, float],
    previous: Mapping[str, float],
) -> tuple[float, float]:
    buy = 0.0
    sell = 0.0
    for stock in current.keys() | previous.keys():
        delta = current.get(stock, 0.0) - previous.get(stock, 0.0)
        if delta > 0:
            buy += delta
        else:
            sell -= delta
    return buy, sell


def _mean_return(
    stocks: Sequence[str],
    row_by_stock: Mapping[str, Mapping[str, object]],
) -> float:
    if not stocks:
        return 0.0
    return fmean(
        float(cast("float", row_by_stock[stock]["realized_return"]))
        for stock in stocks
    )


def _checkpoint_count(repository: Path) -> int:
    return len(list((repository / "artifacts" / "r3_task25").glob("**/*.pt"))) + len(
        list((repository / "checkpoints").glob("**/*.pt"))
    )


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def _as_float(value: object) -> float:
    return float(cast("str | float | int", value))


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise TypeError("expected list of mappings")
    return list(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
