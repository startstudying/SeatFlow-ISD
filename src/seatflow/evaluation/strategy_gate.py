"""Strict, fail-closed R3.6 cost-aware strategy gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from statistics import fmean, pstdev
from typing import cast


def evaluate_strategy_gate(
    runs: Sequence[Mapping[str, object]],
    baseline_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate one strategy against its signal-matched P0 baseline."""

    if len(runs) != 6 or len(baseline_runs) != 6:
        raise ValueError("strategy gate requires exactly six strategy and P0 runs")
    strategy = _index_runs(runs)
    baseline = _index_runs(baseline_runs)
    if strategy.keys() != baseline.keys():
        raise ValueError("strategy and P0 run identities must match")

    fold_means = {
        str(fold): fmean(
            _number(row, "mean_net_spread")
            for key, row in strategy.items()
            if key[0] == fold
        )
        for fold in sorted({key[0] for key in strategy})
    }
    seed_means = {
        str(seed): fmean(
            _number(row, "mean_net_spread")
            for key, row in strategy.items()
            if key[1] == seed
        )
        for seed in sorted({key[1] for key in strategy})
    }
    strategy_net = [_number(row, "mean_net_spread") for row in strategy.values()]
    baseline_net = [_number(row, "mean_net_spread") for row in baseline.values()]
    mean_net = fmean(strategy_net)
    mean_baseline_net = fmean(baseline_net)
    mean_gross = fmean(_number(row, "mean_gross_spread") for row in strategy.values())
    mean_turnover = fmean(_number(row, "mean_turnover") for row in strategy.values())
    baseline_turnover = fmean(
        _number(row, "mean_turnover") for row in baseline.values()
    )
    mean_cost = fmean(_number(row, "mean_cost_diff") for row in strategy.values())
    baseline_cost = fmean(
        _number(row, "mean_cost_diff") for row in baseline.values()
    )
    cost_reduction = _reduction(mean_cost, baseline_cost)
    turnover_reduction = _reduction(mean_turnover, baseline_turnover)
    worst_run = min(strategy_net)
    worst_baseline = min(baseline_net)
    positive_runs = sum(value > 0 for value in strategy_net)
    checks = {
        "at_least_four_positive_net_runs": positive_runs >= 4,
        "both_fold_means_non_negative": (
            len(fold_means) == 2 and all(value >= 0 for value in fold_means.values())
        ),
        "mean_gross_non_negative": mean_gross >= 0,
        "mean_net_above_p0": mean_net > mean_baseline_net,
        "cost_reduction_at_least_30_percent": cost_reduction >= 0.30,
        "turnover_reduction_at_least_30_percent": turnover_reduction >= 0.30,
        "worst_run_not_below_p0": worst_run >= worst_baseline,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "positive_net_runs": positive_runs,
            "fold_mean_net_spread": fold_means,
            "mean_gross_spread": mean_gross,
            "mean_net_spread": mean_net,
            "p0_mean_net_spread": mean_baseline_net,
            "mean_cost_diff": mean_cost,
            "p0_mean_cost_diff": baseline_cost,
            "cost_reduction_vs_p0": cost_reduction,
            "mean_turnover": mean_turnover,
            "p0_mean_turnover": baseline_turnover,
            "turnover_reduction_vs_p0": turnover_reduction,
            "worst_run_net_spread": worst_run,
            "p0_worst_run_net_spread": worst_baseline,
            "seed_net_spread_std": (
                pstdev(seed_means.values()) if len(seed_means) > 1 else 0.0
            ),
        },
    }


def select_unique_strategy(
    evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Select the best passing strategy using the frozen lexicographic order."""

    passing = [
        (name, result)
        for name, result in evaluations.items()
        if bool(result.get("passed"))
    ]
    if not passing:
        return {
            "selected_strategy": None,
            "task48_allowed": False,
            "status": "negative_result_under_current_data",
        }

    def key(item: tuple[str, Mapping[str, object]]) -> tuple[object, ...]:
        name, result = item
        metrics = _mapping(result["metrics"])
        return (
            _number(metrics, "worst_run_net_spread"),
            _as_int(metrics["positive_net_runs"]),
            -_number(metrics, "seed_net_spread_std"),
            -_number(metrics, "mean_turnover"),
            _number(metrics, "mean_net_spread"),
            name,
        )

    selected = max(passing, key=key)[0]
    return {
        "selected_strategy": selected,
        "task48_allowed": True,
        "status": "unique_strategy_selected",
        "passing_strategies": [name for name, _ in sorted(passing)],
    }


def _index_runs(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], Mapping[str, object]]:
    indexed: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in rows:
        key = (_as_int(row["fold"]), _as_int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate gate run identity: {key}")
        indexed[key] = row
    return indexed


def _reduction(current: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0 if current >= baseline else 1.0
    return (baseline - current) / baseline


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def _number(row: Mapping[str, object], field: str) -> float:
    value = float(cast("str | float | int", row[field]))
    if not isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _as_int(value: object) -> int:
    return int(cast("str | int", value))
