from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from seatflow.research import (
    MatrixDataset,
    ResearchMetrics,
    ResearchSample,
    evaluate_predictions,
    predict_proba,
    samples_to_matrix,
    train_l2_logistic,
)

BaselineName = Literal[
    "naive_event_mean",
    "price_only_logistic",
    "event_only_logistic",
    "event_seat_profile_logistic",
]

BASELINE_NAMES: tuple[BaselineName, ...] = (
    "naive_event_mean",
    "price_only_logistic",
    "event_only_logistic",
    "event_seat_profile_logistic",
)

PRICE_FEATURE_NAMES = frozenset(
    {
        "close_to_open",
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_ratio_20d",
        "amplitude_20d",
        "volatility_20d",
    }
)
EVENT_FEATURE_NAMES = frozenset(
    {"event_buy_amount", "event_sell_amount", "event_net_amount", "event_count"}
)
SEAT_PROFILE_PREFIXES: tuple[str, ...] = ("seat_", "seat_profile_")


@dataclass(frozen=True)
class BaselineScore:
    baseline_name: str
    seed: int
    sample_id: str
    as_of_date: date
    stock_id: str
    score: float
    label: int
    label_return: float

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "seed": self.seed,
            "sample_id": self.sample_id,
            "as_of_date": self.as_of_date.isoformat(),
            "stock_id": self.stock_id,
            "score": round(self.score, 12),
            "label": self.label,
            "label_return": round(self.label_return, 12),
        }


@dataclass(frozen=True)
class BaselineEvaluation:
    baseline_name: str
    seed: int
    scores: tuple[BaselineScore, ...]
    metrics: ResearchMetrics
    top_k: int
    top_k_mean_return: float

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "seed": self.seed,
            "scores": [score.to_dict() for score in self.scores],
            "metrics": self.metrics.to_dict(),
            "top_k": self.top_k,
            "top_k_mean_return": round(self.top_k_mean_return, 12),
        }


def run_strong_baseline(
    baseline_name: BaselineName,
    train_samples: tuple[ResearchSample, ...],
    test_samples: tuple[ResearchSample, ...],
    *,
    seed: int,
    top_k: int = 5,
    epochs: int = 300,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> BaselineEvaluation:
    """Fit one deterministic audit baseline and emit per-sample test scores."""

    if not train_samples:
        raise ValueError("train_samples must not be empty")
    if not test_samples:
        raise ValueError("test_samples must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    train = samples_to_matrix(train_samples)
    test = samples_to_matrix(test_samples)
    if baseline_name == "naive_event_mean":
        probability = sum(train.y) / len(train.y)
        probabilities = tuple(float(probability) for _ in test.y)
    elif baseline_name == "price_only_logistic":
        probabilities = _fit_predict_subset(
            train,
            test,
            PRICE_FEATURE_NAMES,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    elif baseline_name == "event_only_logistic":
        probabilities = _fit_predict_subset(
            train,
            test,
            EVENT_FEATURE_NAMES,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    elif baseline_name == "event_seat_profile_logistic":
        seat_profile_names = frozenset(
            name
            for name in train.feature_names
            if any(name.startswith(prefix) for prefix in SEAT_PROFILE_PREFIXES)
        )
        if not seat_profile_names:
            raise ValueError(
                "event_seat_profile_logistic requires at least one seat-profile feature "
                "with a 'seat_' prefix"
            )
        probabilities = _fit_predict_subset(
            train,
            test,
            EVENT_FEATURE_NAMES | seat_profile_names,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    else:
        raise ValueError(f"unsupported baseline_name: {baseline_name}")

    scores = tuple(
        BaselineScore(
            baseline_name=baseline_name,
            seed=seed,
            sample_id=sample.sample_id,
            as_of_date=sample.as_of_date,
            stock_id=sample.stock_id,
            score=score,
            label=sample.label,
            label_return=sample.label_return,
        )
        for sample, score in zip(test_samples, probabilities, strict=True)
    )
    metrics = evaluate_predictions(probabilities, test.y, test.label_returns)
    return BaselineEvaluation(
        baseline_name=baseline_name,
        seed=seed,
        scores=scores,
        metrics=metrics,
        top_k=top_k,
        top_k_mean_return=top_k_mean_return(scores, top_k),
    )


def top_k_mean_return(scores: tuple[BaselineScore, ...], top_k: int) -> float:
    if not scores:
        raise ValueError("scores must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    dates = tuple(sorted({score.as_of_date for score in scores}))
    daily_returns: list[float] = []
    for as_of_date in dates:
        day_scores = tuple(score for score in scores if score.as_of_date == as_of_date)
        selected = sorted(day_scores, key=lambda item: (-item.score, item.sample_id))[:top_k]
        daily_returns.append(sum(score.label_return for score in selected) / len(selected))
    return sum(daily_returns) / len(daily_returns)


def _fit_predict_subset(
    train: MatrixDataset,
    test: MatrixDataset,
    selected_names: frozenset[str],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[float, ...]:
    selected_indexes = tuple(
        index for index, name in enumerate(train.feature_names) if name in selected_names
    )
    if not selected_indexes:
        raise ValueError("selected baseline feature set is empty")
    if test.feature_names != train.feature_names:
        raise ValueError("train and test samples must share feature_names")
    train_subset = _select_columns(train, selected_indexes)
    test_subset = _select_columns(test, selected_indexes)
    model = train_l2_logistic(
        train_subset,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    return predict_proba(model, test_subset.x)


def _select_columns(dataset: MatrixDataset, indexes: tuple[int, ...]) -> MatrixDataset:
    return MatrixDataset(
        sample_ids=dataset.sample_ids,
        as_of_dates=dataset.as_of_dates,
        feature_names=tuple(dataset.feature_names[index] for index in indexes),
        x=tuple(tuple(row[index] for index in indexes) for row in dataset.x),
        y=dataset.y,
        label_returns=dataset.label_returns,
    )
