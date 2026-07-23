"""Post-completion evaluation and fail-closed selection for R3 Task 25."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from seatflow.evaluation.portfolio_costs import (
    TransactionCostModel,
    calculate_turnover_cost,
)
from seatflow.evaluation.stability import jaccard, spearman
from seatflow.training.r3_task25 import group_samples_by_date


def create_completion_manifest(
    planned_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    planned_path = Path(planned_manifest_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"completion manifest already exists: {output}")
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    roots = planned_path.parent
    completed: list[dict[str, object]] = []
    for run in planned["run_order"]:
        run_directory = roots / run["output_directory"]
        completion_path = run_directory / "run_complete.json"
        if not completion_path.exists():
            raise ValueError(f"missing run completion: {run['run_id']}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion["run_id"] != run["run_id"]:
            raise ValueError("run completion identity mismatch")
        if completion["planned_manifest_sha256"] != _sha256(planned_path):
            raise ValueError("run completion planned manifest hash mismatch")
        for artifact in completion["artifacts"]:
            path = run_directory / artifact["path"]
            if (
                not path.exists()
                or path.stat().st_size != artifact["bytes"]
                or _sha256(path) != artifact["sha256"]
            ):
                raise ValueError(f"run artifact integrity mismatch: {path}")
        completed.append(
            {
                "run_id": run["run_id"],
                "run_complete_path": completion_path.as_posix(),
                "run_complete_sha256": _sha256(completion_path),
            }
        )
    if len(completed) != 18:
        raise ValueError("Task25 completion manifest requires 18 runs")
    payload = {
        "schema_version": "r3_task25_completion_manifest.v1",
        "planned_manifest_path": planned_path.as_posix(),
        "planned_manifest_sha256": _sha256(planned_path),
        "completed_run_count": len(completed),
        "all_runs_complete": True,
        "runs": completed,
    }
    _write_json_exclusive(output, payload)
    return output


def evaluate_completed_study(
    samples: Sequence[Any],
    *,
    planned_manifest_path: str | Path,
    completion_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    planned_path = Path(planned_manifest_path)
    completion_path = Path(completion_manifest_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Task25 evaluation already exists: {output}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["completed_run_count"] != 18:
        raise ValueError("Task25 evaluation is locked until all 18 runs complete")
    planned = json.loads(planned_path.read_text(encoding="utf-8"))
    if completion["planned_manifest_sha256"] != _sha256(planned_path):
        raise ValueError("Task25 completion does not bind the planned manifest")
    truth, market = _truth_and_market_rows(samples)
    run_results: list[dict[str, object]] = []
    daily_by_run: dict[str, list[dict[str, object]]] = {}
    prediction_rows_by_key: dict[
        tuple[str, int, int, str], list[dict[str, object]]
    ] = {}
    for run in planned["run_order"]:
        root = planned_path.parent / run["output_directory"]
        predictions = _read_jsonl(root / "daily_predictions.jsonl")
        daily = _evaluate_run_daily(predictions, truth, market)
        run_id = str(run["run_id"])
        daily_by_run[run_id] = daily
        prediction_rows_by_key[
            (
                str(run["objective"]),
                _as_int(run["fold"]),
                _as_int(run["seed"]),
                run_id,
            )
        ] = predictions
        resources = json.loads((root / "resources.json").read_text(encoding="utf-8"))
        run_results.append(
            {
                "run_id": run_id,
                "objective": run["objective"],
                "fold": run["fold"],
                "seed": run["seed"],
                "rank_ic": fmean(_as_float(row["rank_ic"]) for row in daily),
                "top5_gross_spread": fmean(
                    _as_float(row["top5_gross_spread"]) for row in daily
                ),
                "top5_net_spread": fmean(
                    _as_float(row["top5_net_spread"]) for row in daily
                ),
                "turnover": fmean(_as_float(row["turnover"]) for row in daily),
                "positive_net_date_ratio": fmean(
                    _as_float(row["top5_net_spread"]) > 0 for row in daily
                ),
                "rank5_6_margin": fmean(
                    _as_float(row["rank5_6_margin"]) for row in daily
                ),
                "peak_reserved_bytes": resources["peak_reserved_bytes"],
            }
        )
    objective_results = {
        objective: _objective_summary(
            objective,
            [row for row in run_results if row["objective"] == objective],
            daily_by_run,
            prediction_rows_by_key,
        )
        for objective in ("L0", "L1", "L2")
    }
    selection = _apply_frozen_gates(objective_results)
    evidence = _failure_evidence(objective_results)
    payload = {
        "schema_version": "r3_task25_evaluation.v1",
        "planned_manifest_sha256": _sha256(planned_path),
        "completion_manifest_sha256": _sha256(completion_path),
        "run_results": run_results,
        "objectives": objective_results,
        "selection": selection,
        "failure_evidence": evidence,
        "task26_allowed": bool(selection["selected_objective"]),
    }
    _write_json_exclusive(output, payload)
    return output


def _truth_and_market_rows(
    samples: Sequence[Any],
) -> tuple[dict[tuple[str, str], float], dict[str, dict[str, float]]]:
    by_date = group_samples_by_date(samples)
    truth: dict[tuple[str, str], float] = {}
    market: dict[str, dict[str, float]] = {}
    for day, rows in by_date.items():
        values = [float(sample.returns[1]) for sample in rows]
        mean = fmean(values)
        residuals = [value - mean for value in values]
        for sample, residual in zip(rows, residuals, strict=True):
            truth[(day, str(sample.target_stock_id))] = residual
        graph = rows[0].graph
        schema = graph.metadata["feature_schema"]["stock"]
        features = graph.data["stock"].x
        return_index = schema.index("stock_return_1d")
        volatility_index = schema.index("stock_volatility_20d")
        amount_index = schema.index("stock_volume_ratio_20d")
        market[day] = {
            "market_return": float(features[:, return_index].float().mean()),
            "market_volatility": float(features[:, volatility_index].float().mean()),
            "market_amount": float(features[:, amount_index].float().mean()),
            "activity": float(len(rows)),
            "candidate_dispersion": pstdev(residuals) if len(residuals) > 1 else 0.0,
        }
    return truth, market


def _evaluate_run_daily(
    predictions: Sequence[dict[str, object]],
    truth: Mapping[tuple[str, str], float],
    market: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        day = str(row["prediction_date"])
        stock = str(row["stock_id"])
        if (day, stock) not in truth:
            raise ValueError(f"prediction has no frozen target: {day}/{stock}")
        if not isfinite(_as_float(row["score"])):
            raise ValueError("prediction score must be finite")
        by_date[day].append(row)
    previous: dict[str, float] = {}
    result: list[dict[str, object]] = []
    cost_model = TransactionCostModel()
    for day, rows in sorted(by_date.items()):
        if len({str(row["stock_id"]) for row in rows}) != len(rows):
            raise ValueError(f"duplicate prediction identity on {day}")
        ordered = sorted(
            rows,
            key=lambda row: (-_as_float(row["score"]), str(row["stock_id"])),
        )
        scores = [_as_float(row["score"]) for row in rows]
        targets = [truth[(day, str(row["stock_id"]))] for row in rows]
        selected = ordered[:5]
        rest = ordered[5:]
        weight = 1.0 / len(selected)
        current = {str(row["stock_id"]): weight for row in selected}
        costs = calculate_turnover_cost(current, previous, model=cost_model)
        previous = current
        selected_mean = fmean(
            truth[(day, str(row["stock_id"]))] for row in selected
        )
        rest_mean = (
            fmean(truth[(day, str(row["stock_id"]))] for row in rest)
            if rest
            else 0.0
        )
        gross = selected_mean - rest_mean
        result.append(
            {
                "prediction_date": day,
                "rank_ic": spearman(scores, targets),
                "top5_gross_spread": gross,
                "top5_net_spread": gross - costs.total_cost,
                "turnover": costs.turnover,
                "rank5_6_margin": (
                    _as_float(ordered[4]["score"]) - _as_float(ordered[5]["score"])
                    if len(ordered) >= 6
                    else 0.0
                ),
                **market[day],
            }
        )
    return result


def _objective_summary(
    objective: str,
    runs: Sequence[dict[str, object]],
    daily_by_run: Mapping[str, Sequence[dict[str, object]]],
    predictions: Mapping[
        tuple[str, int, int, str], Sequence[dict[str, object]]
    ],
) -> dict[str, object]:
    by_fold: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in runs:
        by_fold[_as_int(row["fold"])].append(row)
        by_seed[_as_int(row["seed"])].append(row)
    seed_means = {
        seed: fmean(_as_float(row["top5_net_spread"]) for row in values)
        for seed, values in by_seed.items()
    }
    stability = _cross_seed_stability(objective, predictions)
    all_daily = [
        daily
        for row in runs
        for daily in daily_by_run[str(row["run_id"])]
    ]
    return {
        "run_count": len(runs),
        "positive_net_runs": sum(
            _as_float(row["top5_net_spread"]) > 0 for row in runs
        ),
        "fold_net_means": {
            str(fold): fmean(_as_float(row["top5_net_spread"]) for row in values)
            for fold, values in by_fold.items()
        },
        "median_rank_ic": median(_as_float(row["rank_ic"]) for row in runs),
        "mean_rank_ic": fmean(_as_float(row["rank_ic"]) for row in runs),
        "mean_net_spread": fmean(
            _as_float(row["top5_net_spread"]) for row in runs
        ),
        "worst_fold_net_spread": min(
            fmean(_as_float(row["top5_net_spread"]) for row in values)
            for values in by_fold.values()
        ),
        "seed_net_spread_std": pstdev(seed_means.values()),
        "mean_rank5_6_margin": fmean(
            _as_float(row["rank5_6_margin"]) for row in runs
        ),
        "top5_jaccard": stability["top5_jaccard"],
        "top10_jaccard": stability["top10_jaccard"],
        "cross_seed_prediction_std": stability["prediction_std"],
        "market_regimes": _market_regime_summary(all_daily),
        "maximum_peak_reserved_bytes": max(
            _as_int(row["peak_reserved_bytes"]) for row in runs
        ),
    }


def _cross_seed_stability(
    objective: str,
    predictions: Mapping[
        tuple[str, int, int, str], Sequence[dict[str, object]]
    ],
) -> dict[str, float]:
    accumulators: dict[str, list[float]] = defaultdict(list)
    for fold in (0, 1):
        by_seed: dict[int, dict[str, dict[str, float]]] = {}
        for seed in (2026, 2027, 2028):
            matching = [
                rows
                for (candidate, candidate_fold, candidate_seed, _), rows in predictions.items()
                if candidate == objective
                and candidate_fold == fold
                and candidate_seed == seed
            ]
            if len(matching) != 1:
                raise ValueError("missing cross-seed prediction run")
            day_rows: dict[str, dict[str, float]] = defaultdict(dict)
            for row in matching[0]:
                day_rows[str(row["prediction_date"])][str(row["stock_id"])] = (
                    _as_float(row["score"])
                )
            by_seed[seed] = day_rows
        for day in sorted(by_seed[2026]):
            identities = [set(by_seed[seed][day]) for seed in (2026, 2027, 2028)]
            if identities[1:] != identities[:-1]:
                raise ValueError("cross-seed stock identities differ")
            stocks = sorted(identities[0])
            top: dict[int, list[str]] = {
                seed: sorted(
                    stocks,
                    key=lambda stock: (-by_seed[seed][day][stock], stock),
                )
                for seed in (2026, 2027, 2028)
            }
            for left, right in ((2026, 2027), (2026, 2028), (2027, 2028)):
                accumulators["top5_jaccard"].append(
                    jaccard(top[left][:5], top[right][:5])
                )
                accumulators["top10_jaccard"].append(
                    jaccard(top[left][:10], top[right][:10])
                )
            for stock in stocks:
                accumulators["prediction_std"].append(
                    pstdev(by_seed[seed][day][stock] for seed in (2026, 2027, 2028))
                )
    return {name: fmean(values) for name, values in accumulators.items()}


def _market_regime_summary(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    thresholds = {
        field: median(_as_float(row[field]) for row in rows)
        for field in (
            "market_volatility",
            "market_amount",
            "activity",
            "candidate_dispersion",
        )
    }
    groups = {
        "up_market": [row for row in rows if _as_float(row["market_return"]) >= 0],
        "down_market": [row for row in rows if _as_float(row["market_return"]) < 0],
    }
    for field in thresholds:
        groups[f"high_{field}"] = [
            row for row in rows if _as_float(row[field]) > thresholds[field]
        ]
        groups[f"low_{field}"] = [
            row for row in rows if _as_float(row[field]) <= thresholds[field]
        ]
    return {
        name: {
            "sample_days": len(values),
            "rank_ic": fmean(_as_float(row["rank_ic"]) for row in values),
            "top5_gross_spread": fmean(
                _as_float(row["top5_gross_spread"]) for row in values
            ),
            "top5_net_spread": fmean(
                _as_float(row["top5_net_spread"]) for row in values
            ),
            "turnover": fmean(_as_float(row["turnover"]) for row in values),
        }
        for name, values in groups.items()
        if values
    }


def _apply_frozen_gates(
    objectives: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    decisions: dict[str, dict[str, object]] = {}
    for name, row in objectives.items():
        checks = {
            "exact_six_runs": row["run_count"] == 6,
            "positive_net_runs_at_least_four": _as_int(row["positive_net_runs"]) >= 4,
            "both_fold_means_non_negative": all(
                _as_float(value) >= 0
                for value in _as_mapping(row["fold_net_means"]).values()
            ),
            "peak_gpu_at_most_7_2_gib": _as_int(row["maximum_peak_reserved_bytes"])
            <= int(7.2 * 1024**3),
            "rank_ic_median_not_below_r2": None,
            "seed_std_reduced_at_least_ten_percent": None,
        }
        blockers = [
            "missing_r2_rank_ic_median_floor",
            "missing_r2_seed_net_spread_std",
        ]
        decisions[name] = {
            "checks": checks,
            "blockers": blockers,
            "passed": False,
        }
    return {
        "status": "blocked_missing_r2_reference",
        "selected_objective": None,
        "decisions": decisions,
        "reason": (
            "The frozen R2 evidence contains no complete RankIC median or seed net "
            "spread standard deviation, so the relative gates cannot be evaluated."
        ),
    }


def _failure_evidence(
    objectives: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    rank_ics = [_as_float(row["mean_rank_ic"]) for row in objectives.values()]
    return {
        "A": {
            "classification": "optimization_randomness",
            "evidence": {
                name: row["seed_net_spread_std"] for name, row in objectives.items()
            },
        },
        "B": {
            "classification": "top5_boundary_noise",
            "evidence": {
                name: {
                    "top5_jaccard": row["top5_jaccard"],
                    "rank5_6_margin": row["mean_rank5_6_margin"],
                }
                for name, row in objectives.items()
            },
        },
        "C": {
            "classification": "market_regime_drift",
            "evidence": {
                name: row["market_regimes"] for name, row in objectives.items()
            },
        },
        "D": {
            "classification": "turnover_cost",
            "evidence": {
                name: row["mean_net_spread"] for name, row in objectives.items()
            },
        },
        "E": {
            "classification": "seat_graph_no_increment",
            "status": "insufficient_evidence",
            "reason": "No predeclared inference ablation exists in the 18-run manifest.",
        },
        "F": {
            "classification": "label_objective_mismatch",
            "status": (
                "supported" if all(value <= 0 for value in rank_ics) else "not_established"
            ),
            "evidence": {"objective_mean_rank_ic": dict(zip(objectives, rank_ics, strict=True))},
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _as_float(value: object) -> float:
    return float(str(value))


def _as_int(value: object) -> int:
    return int(str(value))


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
