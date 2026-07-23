"""Fail-closed cross-seed stability diagnostics for ranking experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import fmean, pstdev
from typing import Any

EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_SEEDS = (2026, 2027, 2028, 2029, 2030)
FAILURE_CLASSES = {
    "A": "optimization_randomness",
    "B": "top5_boundary_noise",
    "C": "market_regime_drift",
    "D": "turnover_cost",
    "E": "seat_graph_no_increment",
    "F": "label_objective_mismatch",
}


@dataclass(frozen=True)
class PredictionRow:
    fold: int
    seed: int
    prediction_date: str
    stock_id: str
    score: float
    target: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> PredictionRow:
        return cls(
            fold=int(str(row["fold"])),
            seed=int(str(row["seed"])),
            prediction_date=str(row["prediction_date"]),
            stock_id=str(row["stock_id"]),
            score=float(str(row["score"])),
            target=float(str(row["target"])),
        )


@dataclass(frozen=True)
class RunMetric:
    fold: int
    seed: int
    rank_ic: float
    gross_spread: float
    net_spread: float
    turnover: float
    regime: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> RunMetric:
        return cls(
            fold=int(str(row["fold"])),
            seed=int(str(row["seed"])),
            rank_ic=float(str(row["rank_ic"])),
            gross_spread=float(str(row["gross_spread"])),
            net_spread=float(str(row["net_spread"])),
            turnover=float(str(row["turnover"])),
            regime=None if row.get("regime") is None else str(row["regime"]),
        )


@dataclass(frozen=True)
class StabilityThresholds:
    rank_ic_seed_std_high: float = 0.03
    rank_ic_close_std: float = 0.01
    top5_jaccard_low: float = 0.40
    regime_rank_ic_gap_high: float = 0.04
    ablation_increment_min: float = 0.0


def analyze_stability(
    predictions: Iterable[PredictionRow],
    run_metrics: Iterable[RunMetric],
    *,
    ablations: Mapping[str, float] | None = None,
    thresholds: StabilityThresholds | None = None,
) -> dict[str, Any]:
    """Analyze a complete 3x5 result set or return explicit missing evidence."""

    thresholds = StabilityThresholds() if thresholds is None else thresholds
    prediction_rows = tuple(predictions)
    metric_rows = tuple(run_metrics)
    evidence = _evidence_gaps(prediction_rows, metric_rows, ablations)
    if evidence:
        return {
            "status": "insufficient_evidence",
            "failure_classes": [],
            "failure_classification": {},
            "evidence_gaps": evidence,
            "thresholds": asdict(thresholds),
        }

    pairwise = _pairwise_daily_metrics(prediction_rows)
    folds = _fold_metrics(metric_rows, pairwise)
    regimes = _regime_metrics(metric_rows)
    failure_classes = _classify_failures(folds, regimes, ablations or {}, thresholds)
    return {
        "status": "complete",
        "failure_classes": failure_classes,
        "failure_classification": {
            code: FAILURE_CLASSES[code] for code in failure_classes
        },
        "folds": folds,
        "regimes": regimes,
        "thresholds": asdict(thresholds),
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires aligned sequences of at least two values")
    left_mean, right_mean = fmean(left), fmean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = sqrt(sum(value * value for value in centered_left)) * sqrt(
        sum(value * value for value in centered_right)
    )
    return 0.0 if denominator == 0 else sum(
        a * b for a, b in zip(centered_left, centered_right, strict=True)
    ) / denominator


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(_average_ranks(left), _average_ranks(right))


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _evidence_gaps(
    predictions: Sequence[PredictionRow],
    metrics: Sequence[RunMetric],
    ablations: Mapping[str, float] | None,
) -> list[str]:
    gaps: list[str] = []
    metric_keys = {(row.fold, row.seed) for row in metrics}
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    missing = sorted(expected - metric_keys)
    if missing:
        gaps.append(f"missing_run_metrics:{missing}")
    prediction_keys = {(row.fold, row.seed) for row in predictions}
    missing_predictions = sorted(expected - prediction_keys)
    if missing_predictions:
        gaps.append(f"missing_prediction_runs:{missing_predictions}")
    if not predictions:
        gaps.extend(["missing_stock_identity", "missing_targets"])
    elif any(not row.stock_id for row in predictions):
        gaps.append("missing_stock_identity")
    required_ablations = {
        "full",
        "no_seat_stock",
        "no_industry_stock",
        "no_stock_stock",
        "stock_only",
    }
    if ablations is None or not required_ablations.issubset(ablations):
        gaps.append("missing_inference_ablations")
    if metrics and not all(row.regime for row in metrics):
        gaps.append("missing_market_regimes")
    return gaps


def _pairwise_daily_metrics(rows: Sequence[PredictionRow]) -> dict[int, dict[str, float]]:
    by_day_seed: dict[tuple[int, str, int], dict[str, PredictionRow]] = defaultdict(dict)
    for row in rows:
        by_day_seed[(row.fold, row.prediction_date, row.seed)][row.stock_id] = row
    accumulators: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    days = sorted({(fold, day) for fold, day, _ in by_day_seed})
    for fold, day in days:
        seed_maps = {
            seed: by_day_seed[(fold, day, seed)]
            for seed in EXPECTED_SEEDS
        }
        stock_sets = {frozenset(values) for values in seed_maps.values()}
        if len(stock_sets) != 1:
            raise ValueError("cross-seed comparison requires identical stock identities")
        stocks = sorted(next(iter(stock_sets)))
        for left_index, left_seed in enumerate(EXPECTED_SEEDS):
            for right_seed in EXPECTED_SEEDS[left_index + 1 :]:
                left_scores = [seed_maps[left_seed][stock].score for stock in stocks]
                right_scores = [seed_maps[right_seed][stock].score for stock in stocks]
                accumulators[fold]["pearson"].append(pearson(left_scores, right_scores))
                accumulators[fold]["spearman"].append(spearman(left_scores, right_scores))
                left_order = sorted(
                    stocks,
                    key=lambda stock: seed_maps[left_seed][stock].score,
                    reverse=True,
                )
                right_order = sorted(
                    stocks,
                    key=lambda stock: seed_maps[right_seed][stock].score,
                    reverse=True,
                )
                accumulators[fold]["top5_jaccard"].append(jaccard(left_order[:5], right_order[:5]))
                accumulators[fold]["top10_jaccard"].append(
                    jaccard(left_order[:10], right_order[:10])
                )
        margins: list[float] = []
        for seed in EXPECTED_SEEDS:
            scores = sorted(
                (row.score for row in seed_maps[seed].values()), reverse=True
            )
            if len(scores) >= 6:
                margins.append(scores[4] - scores[5])
        accumulators[fold]["rank5_6_margin"].extend(margins)
        for stock in stocks:
            accumulators[fold]["daily_prediction_std"].append(
                pstdev(seed_maps[seed][stock].score for seed in EXPECTED_SEEDS)
            )
    return {
        fold: {name: fmean(values) for name, values in metrics.items()}
        for fold, metrics in accumulators.items()
    }


def _fold_metrics(
    metrics: Sequence[RunMetric],
    pairwise: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    grouped: dict[int, list[RunMetric]] = defaultdict(list)
    for row in metrics:
        grouped[row.fold].append(row)
    result: dict[str, dict[str, float]] = {}
    for fold, rows in sorted(grouped.items()):
        rank_ics = [row.rank_ic for row in rows]
        result[str(fold)] = {
            "rank_ic_mean": fmean(rank_ics),
            "rank_ic_seed_std": pstdev(rank_ics),
            "gross_spread_mean": fmean(row.gross_spread for row in rows),
            "net_spread_mean": fmean(row.net_spread for row in rows),
            "turnover_mean": fmean(row.turnover for row in rows),
            **pairwise[fold],
        }
    return result


def _regime_metrics(metrics: Sequence[RunMetric]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RunMetric]] = defaultdict(list)
    for row in metrics:
        grouped[str(row.regime)].append(row)
    return {
        regime: {
            "rank_ic": fmean(row.rank_ic for row in rows),
            "gross_spread": fmean(row.gross_spread for row in rows),
            "net_spread": fmean(row.net_spread for row in rows),
            "turnover": fmean(row.turnover for row in rows),
            "sample_count": float(len(rows)),
        }
        for regime, rows in sorted(grouped.items())
    }


def _classify_failures(
    folds: Mapping[str, Mapping[str, float]],
    regimes: Mapping[str, Mapping[str, float]],
    ablations: Mapping[str, float],
    thresholds: StabilityThresholds,
) -> list[str]:
    failures: set[str] = set()
    if any(row["rank_ic_seed_std"] >= thresholds.rank_ic_seed_std_high for row in folds.values()):
        failures.add("A")
    if any(
        row["rank_ic_seed_std"] <= thresholds.rank_ic_close_std
        and row["top5_jaccard"] < thresholds.top5_jaccard_low
        for row in folds.values()
    ):
        failures.add("B")
    regime_rank_ics = [row["rank_ic"] for row in regimes.values()]
    regime_gap = max(regime_rank_ics) - min(regime_rank_ics) if regime_rank_ics else 0.0
    if regime_gap >= thresholds.regime_rank_ic_gap_high:
        failures.add("C")
    if any(row["gross_spread_mean"] > 0 >= row["net_spread_mean"] for row in folds.values()):
        failures.add("D")
    if ablations["full"] - ablations["no_seat_stock"] <= thresholds.ablation_increment_min:
        failures.add("E")
    if all(row["rank_ic_mean"] <= 0 for row in folds.values()):
        failures.add("F")
    return sorted(failures)


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + end - 1) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average
        start = end
    return ranks
