from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from seatflow.models.strong_baselines import (
    BASELINE_NAMES,
    BaselineEvaluation,
    BaselineName,
    BaselineScore,
    run_strong_baseline,
)
from seatflow.research import ResearchSample

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RollingFold:
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train: tuple[ResearchSample, ...]
    test: tuple[ResearchSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_sample_ids": [sample.sample_id for sample in self.train],
            "test_sample_ids": [sample.sample_id for sample in self.test],
        }


@dataclass(frozen=True)
class SeedFoldEvaluation:
    seed: int
    fold_id: str
    baseline: BaselineEvaluation

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "fold_id": self.fold_id,
            "baseline": self.baseline.to_dict(),
        }


@dataclass(frozen=True)
class BaselineAggregate:
    baseline_name: str
    seed_count: int
    fold_count: int
    run_count: int
    mean_auc: float | None
    worst_fold_auc: float | None
    mean_rank_ic: float | None
    worst_fold_rank_ic: float | None
    mean_top_k_mean_return: float
    worst_fold_top_k_mean_return: float
    raw_scores: tuple[BaselineScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "seed_count": self.seed_count,
            "fold_count": self.fold_count,
            "run_count": self.run_count,
            "mean_auc": _round_optional(self.mean_auc, 6),
            "worst_fold_auc": _round_optional(self.worst_fold_auc, 6),
            "mean_rank_ic": _round_optional(self.mean_rank_ic, 6),
            "worst_fold_rank_ic": _round_optional(self.worst_fold_rank_ic, 6),
            "mean_top_k_mean_return": round(self.mean_top_k_mean_return, 12),
            "worst_fold_top_k_mean_return": round(self.worst_fold_top_k_mean_return, 12),
            "raw_scores": [score.to_dict() for score in self.raw_scores],
        }


@dataclass(frozen=True)
class RollingEvaluation:
    seeds: tuple[int, ...]
    folds: tuple[RollingFold, ...]
    results: tuple[SeedFoldEvaluation, ...]
    aggregates: tuple[BaselineAggregate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "folds": [fold.to_dict() for fold in self.folds],
            "results": [result.to_dict() for result in self.results],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
        }


def build_rolling_folds(
    samples: tuple[ResearchSample, ...],
    *,
    n_folds: int = 3,
    min_train_dates: int = 3,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> tuple[RollingFold, ...]:
    """Create deterministic expanding-window folds with label-end purge and embargo."""

    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    if min_train_dates < 1:
        raise ValueError("min_train_dates must be positive")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")
    ordered = tuple(sorted(samples, key=lambda item: (item.as_of_date, item.stock_id)))
    if not ordered:
        raise ValueError("samples must not be empty")
    dates = tuple(sorted({sample.as_of_date for sample in ordered}))
    if len(dates) < min_train_dates + n_folds:
        raise ValueError("not enough unique dates for requested rolling folds")

    evaluation_dates = dates[min_train_dates:]
    chunks = _contiguous_chunks(evaluation_dates, n_folds)
    folds: list[RollingFold] = []
    for index, chunk in enumerate(chunks):
        test_start = chunk[0]
        test_end = chunk[-1]
        train_as_of_cutoff = test_start - timedelta(days=embargo_days)
        train_label_cutoff = test_start - timedelta(days=purge_days)
        train = tuple(
            sample
            for sample in ordered
            if sample.as_of_date < train_as_of_cutoff
            and sample.label_end_date < train_label_cutoff
        )
        test = tuple(sample for sample in ordered if test_start <= sample.as_of_date <= test_end)
        if not train:
            raise ValueError(f"rolling fold {index} has empty train partition")
        if not test:
            raise ValueError(f"rolling fold {index} has empty test partition")
        folds.append(
            RollingFold(
                fold_id=f"fold_{index}",
                train_start=train[0].as_of_date,
                train_end=max(sample.as_of_date for sample in train),
                test_start=test_start,
                test_end=test_end,
                train=train,
                test=test,
            )
        )
    return tuple(folds)


def run_rolling_evaluation(
    samples: tuple[ResearchSample, ...],
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    baseline_names: tuple[BaselineName, ...] = BASELINE_NAMES,
    n_folds: int = 3,
    min_train_dates: int = 3,
    purge_days: int = 0,
    embargo_days: int = 0,
    top_k: int = 5,
    epochs: int = 300,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> RollingEvaluation:
    if len(seeds) != 5:
        raise ValueError("rolling evaluation must use exactly 5 seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("rolling evaluation seeds must be unique")
    if not baseline_names:
        raise ValueError("baseline_names must not be empty")
    folds = build_rolling_folds(
        samples,
        n_folds=n_folds,
        min_train_dates=min_train_dates,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    results = tuple(
        SeedFoldEvaluation(
            seed=seed,
            fold_id=fold.fold_id,
            baseline=run_strong_baseline(
                baseline_name,
                fold.train,
                fold.test,
                seed=seed,
                top_k=top_k,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
            ),
        )
        for baseline_name in baseline_names
        for seed in seeds
        for fold in folds
    )
    aggregates = tuple(
        _aggregate_baseline(baseline_name, seeds, folds, results)
        for baseline_name in baseline_names
    )
    return RollingEvaluation(seeds=seeds, folds=folds, results=results, aggregates=aggregates)


def _aggregate_baseline(
    baseline_name: str,
    seeds: tuple[int, ...],
    folds: tuple[RollingFold, ...],
    results: tuple[SeedFoldEvaluation, ...],
) -> BaselineAggregate:
    selected = tuple(
        result for result in results if result.baseline.baseline_name == baseline_name
    )
    fold_auc = tuple(
        _mean_optional(
            tuple(
                result.baseline.metrics.auc
                for result in selected
                if result.fold_id == fold.fold_id
            )
        )
        for fold in folds
    )
    fold_rank_ic = tuple(
        _mean_optional(
            tuple(
                result.baseline.metrics.rank_ic
                for result in selected
                if result.fold_id == fold.fold_id
            )
        )
        for fold in folds
    )
    fold_top_k = tuple(
        _mean(
            tuple(
                result.baseline.top_k_mean_return
                for result in selected
                if result.fold_id == fold.fold_id
            )
        )
        for fold in folds
    )
    return BaselineAggregate(
        baseline_name=baseline_name,
        seed_count=len(seeds),
        fold_count=len(folds),
        run_count=len(selected),
        mean_auc=_mean_optional(tuple(result.baseline.metrics.auc for result in selected)),
        worst_fold_auc=_min_optional(fold_auc),
        mean_rank_ic=_mean_optional(tuple(result.baseline.metrics.rank_ic for result in selected)),
        worst_fold_rank_ic=_min_optional(fold_rank_ic),
        mean_top_k_mean_return=_mean(
            tuple(result.baseline.top_k_mean_return for result in selected)
        ),
        worst_fold_top_k_mean_return=min(fold_top_k),
        raw_scores=tuple(
            score for result in selected for score in result.baseline.scores
        ),
    )


def _contiguous_chunks(values: tuple[date, ...], count: int) -> tuple[tuple[date, ...], ...]:
    if len(values) < count:
        raise ValueError("cannot create non-empty folds from fewer dates than folds")
    chunks: list[tuple[date, ...]] = []
    start = 0
    for index in range(count):
        remaining_values = len(values) - start
        remaining_chunks = count - index
        size = (remaining_values + remaining_chunks - 1) // remaining_chunks
        chunk = values[start : start + size]
        if not chunk:
            raise ValueError("rolling fold date chunk is empty")
        chunks.append(chunk)
        start += size
    return tuple(chunks)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return sum(present) / len(present)


def _min_optional(values: tuple[float | None, ...]) -> float | None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return None
    return min(present)


def _round_optional(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)
