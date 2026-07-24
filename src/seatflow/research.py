from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from math import exp, isfinite, sqrt
from typing import Any

from seatflow.data.contracts import DailyBar, DragonTigerSummary, normalize_stock_id

FEATURE_NAMES: tuple[str, ...] = (
    "event_buy_amount",
    "event_sell_amount",
    "event_net_amount",
    "event_count",
    "close_to_open",
    "return_1d",
    "return_5d",
    "return_20d",
    "volume_ratio_20d",
    "amplitude_20d",
    "volatility_20d",
)


@dataclass(frozen=True)
class ResearchSample:
    sample_id: str
    as_of_date: date
    stock_id: str
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    label_return: float
    label: int
    label_start_date: date
    label_end_date: date

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "as_of_date": self.as_of_date.isoformat(),
            "stock_id": self.stock_id,
            "feature_names": list(self.feature_names),
            "features": [round(value, 12) for value in self.features],
            "label_return": round(self.label_return, 12),
            "label": self.label,
            "label_start_date": self.label_start_date.isoformat(),
            "label_end_date": self.label_end_date.isoformat(),
        }


@dataclass(frozen=True)
class MatrixDataset:
    sample_ids: tuple[str, ...]
    as_of_dates: tuple[date, ...]
    feature_names: tuple[str, ...]
    x: tuple[tuple[float, ...], ...]
    y: tuple[int, ...]
    label_returns: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_ids": list(self.sample_ids),
            "as_of_dates": [item.isoformat() for item in self.as_of_dates],
            "feature_names": list(self.feature_names),
            "x": [[round(value, 12) for value in row] for row in self.x],
            "y": list(self.y),
            "label_returns": [round(value, 12) for value in self.label_returns],
        }


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    l2: float
    seed: int
    epochs: int
    learning_rate: float
    losses: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": [round(value, 12) for value in self.mean],
            "scale": [round(value, 12) for value in self.scale],
            "weights": [round(value, 12) for value in self.weights],
            "bias": round(self.bias, 12),
            "l2": self.l2,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "losses": [round(value, 12) for value in self.losses],
        }


@dataclass(frozen=True)
class ResearchMetrics:
    sample_count: int
    accuracy: float
    positive_rate: float
    mean_probability: float
    rank_ic: float | None
    auc: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "accuracy": round(self.accuracy, 6),
            "positive_rate": round(self.positive_rate, 6),
            "mean_probability": round(self.mean_probability, 6),
            "rank_ic": round(self.rank_ic, 6) if self.rank_ic is not None else None,
            "auc": round(self.auc, 6) if self.auc is not None else None,
        }


@dataclass(frozen=True)
class TimeSplitResult:
    train: tuple[ResearchSample, ...]
    valid: tuple[ResearchSample, ...]
    test: tuple[ResearchSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "train": [sample.sample_id for sample in self.train],
            "valid": [sample.sample_id for sample in self.valid],
            "test": [sample.sample_id for sample in self.test],
        }


def build_research_samples(
    events: Iterable[DragonTigerSummary],
    bars: Iterable[DailyBar],
    lookback_days: int = 20,
    positive_return_threshold: float = 0.0,
) -> tuple[ResearchSample, ...]:
    """Build event samples from trailing OHLCV only and future open-to-open labels."""

    if lookback_days < 20:
        raise ValueError("lookback_days must be at least 20")
    bars_by_stock = _group_bars(bars)
    events_by_key = _group_events(events)
    samples: list[ResearchSample] = []

    for (stock_id, as_of_date), stock_events in sorted(
        events_by_key.items(), key=lambda item: item[0]
    ):
        ordered_bars = bars_by_stock.get(stock_id)
        if ordered_bars is None:
            continue
        as_of_index = _bar_index(ordered_bars, as_of_date)
        if as_of_index is None or as_of_index + 2 >= len(ordered_bars):
            continue
        lookback = ordered_bars[as_of_index - lookback_days + 1 : as_of_index + 1]
        if len(lookback) != lookback_days:
            continue
        label_start = ordered_bars[as_of_index + 1]
        label_end = ordered_bars[as_of_index + 2]
        if label_start.trade_date <= as_of_date or label_end.trade_date <= label_start.trade_date:
            raise ValueError("label open-to-open window must be strictly after as_of_date")
        if label_start.open_price <= 0:
            raise ValueError("label start open_price must be positive")
        label_return = _decimal_ratio(label_end.open_price, label_start.open_price) - 1.0
        samples.append(
            ResearchSample(
                sample_id=f"{as_of_date.isoformat()}:{stock_id}",
                as_of_date=as_of_date,
                stock_id=stock_id,
                feature_names=FEATURE_NAMES,
                features=_features_from_lookback(stock_events, lookback),
                label_return=label_return,
                label=1 if label_return > positive_return_threshold else 0,
                label_start_date=label_start.trade_date,
                label_end_date=label_end.trade_date,
            )
        )
    return tuple(samples)


def samples_to_matrix(samples: Sequence[ResearchSample]) -> MatrixDataset:
    if not samples:
        raise ValueError("samples must not be empty")
    feature_names = samples[0].feature_names
    for sample in samples:
        if sample.feature_names != feature_names:
            raise ValueError("all samples must share the same feature_names")
    return MatrixDataset(
        sample_ids=tuple(sample.sample_id for sample in samples),
        as_of_dates=tuple(sample.as_of_date for sample in samples),
        feature_names=feature_names,
        x=tuple(sample.features for sample in samples),
        y=tuple(sample.label for sample in samples),
        label_returns=tuple(sample.label_return for sample in samples),
    )


def time_split_samples(
    samples: Sequence[ResearchSample],
    train_end: date,
    valid_end: date,
    *,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> TimeSplitResult:
    if train_end >= valid_end:
        raise ValueError("train_end must be before valid_end")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")
    ordered = tuple(sorted(samples, key=lambda item: (item.as_of_date, item.stock_id)))
    train_cutoff = train_end - timedelta(days=purge_days)
    valid_start = train_end + timedelta(days=embargo_days)
    valid_cutoff = valid_end - timedelta(days=purge_days)
    test_start = valid_end + timedelta(days=embargo_days)
    return TimeSplitResult(
        train=tuple(
            sample
            for sample in ordered
            if sample.as_of_date <= train_cutoff
            and (purge_days == 0 or sample.label_end_date <= train_end)
        ),
        valid=tuple(
            sample
            for sample in ordered
            if valid_start < sample.as_of_date <= valid_cutoff
            and (purge_days == 0 or sample.label_end_date <= valid_end)
        ),
        test=tuple(sample for sample in ordered if sample.as_of_date > test_start),
    )


def train_l2_logistic(
    dataset: MatrixDataset,
    *,
    learning_rate: float = 0.1,
    epochs: int = 300,
    l2: float = 1e-3,
    seed: int = 0,
) -> LogisticModel:
    np = _require_numpy()
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    x = np.asarray(dataset.x, dtype=float)
    y = np.asarray(dataset.y, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("dataset.x must be a non-empty 2D matrix")
    if y.shape != (x.shape[0],):
        raise ValueError("dataset.y must align with dataset.x rows")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    x_standard = (x - mean) / scale

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x.shape[1])
    bias = 0.0
    losses: list[float] = []
    for _ in range(epochs):
        logits = x_standard @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        error = probabilities - y
        weights -= learning_rate * ((x_standard.T @ error) / x.shape[0] + l2 * weights)
        bias -= learning_rate * float(error.mean())
        loss = -np.mean(
            y * np.log(np.clip(probabilities, 1e-12, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - probabilities, 1e-12, 1.0))
        ) + 0.5 * l2 * float(weights @ weights)
        losses.append(float(loss))

    return LogisticModel(
        feature_names=dataset.feature_names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        l2=l2,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        losses=tuple(losses),
    )


def predict_proba(model: LogisticModel, x: Sequence[Sequence[float]]) -> tuple[float, ...]:
    probabilities: list[float] = []
    for row in x:
        if len(row) != len(model.weights):
            raise ValueError("feature row width must match model weights")
        logit = model.bias + sum(
            ((value - mean) / scale) * weight
            for value, mean, scale, weight in zip(
                row, model.mean, model.scale, model.weights, strict=True
            )
        )
        probabilities.append(_sigmoid(logit))
    return tuple(probabilities)


def evaluate_predictions(
    probabilities: Sequence[float],
    labels: Sequence[int],
    label_returns: Sequence[float] | None = None,
    threshold: float = 0.5,
) -> ResearchMetrics:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("at least one prediction is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    for probability in probabilities:
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must be finite values in [0, 1]")
    for label in labels:
        if label not in (0, 1):
            raise ValueError("labels must be binary integers")
    predictions = tuple(1 if probability >= threshold else 0 for probability in probabilities)
    correct = sum(
        1 for prediction, label in zip(predictions, labels, strict=True) if prediction == label
    )
    rank_ic = None
    if label_returns is not None:
        if len(label_returns) != len(probabilities):
            raise ValueError("label_returns must align with probabilities")
        rank_ic = _spearman(probabilities, label_returns)
    return ResearchMetrics(
        sample_count=len(probabilities),
        accuracy=correct / len(probabilities),
        positive_rate=sum(labels) / len(labels),
        mean_probability=sum(probabilities) / len(probabilities),
        rank_ic=rank_ic,
        auc=_auc(probabilities, labels),
    )


def _group_bars(bars: Iterable[DailyBar]) -> dict[str, tuple[DailyBar, ...]]:
    grouped: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.stock_id].append(bar)
    return {
        stock_id: tuple(sorted(stock_bars, key=lambda item: item.trade_date))
        for stock_id, stock_bars in grouped.items()
    }


def _group_events(
    events: Iterable[DragonTigerSummary],
) -> dict[tuple[str, date], tuple[DragonTigerSummary, ...]]:
    grouped: dict[tuple[str, date], list[DragonTigerSummary]] = defaultdict(list)
    for event in events:
        grouped[(normalize_stock_id(event.stock_id), event.trade_date)].append(event)
    return {key: tuple(value) for key, value in grouped.items()}


def _bar_index(bars: Sequence[DailyBar], trade_date: date) -> int | None:
    for index, bar in enumerate(bars):
        if bar.trade_date == trade_date:
            return index
    return None


def _features_from_lookback(
    events: Sequence[DragonTigerSummary],
    lookback: Sequence[DailyBar],
) -> tuple[float, ...]:
    current = lookback[-1]
    closes = tuple(float(bar.close_price) for bar in lookback)
    volumes = tuple(float(bar.volume) for bar in lookback)
    highs = tuple(float(bar.high_price) for bar in lookback)
    lows = tuple(float(bar.low_price) for bar in lookback)
    returns = tuple(
        closes[index] / closes[index - 1] - 1.0 for index in range(1, len(closes))
    )
    mean_volume = sum(volumes) / len(volumes)
    return (
        float(sum(event.buy_amount for event in events)),
        float(sum(event.sell_amount for event in events)),
        float(sum(event.net_amount for event in events)),
        float(len(events)),
        _decimal_ratio(current.close_price, current.open_price) - 1.0,
        closes[-1] / closes[-2] - 1.0,
        closes[-1] / closes[-6] - 1.0,
        closes[-1] / closes[0] - 1.0,
        volumes[-1] / mean_volume if mean_volume > 0 else 0.0,
        (max(highs) - min(lows)) / closes[-1] if closes[-1] > 0 else 0.0,
        _std(returns),
    )


def _decimal_ratio(numerator: Decimal, denominator: Decimal) -> float:
    if denominator <= 0:
        raise ValueError("price denominator must be positive")
    return float(numerator / denominator)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = tuple(value - mean_left for value in left)
    centered_right = tuple(value - mean_right for value in right)
    denom = sqrt(sum(value * value for value in centered_left)) * sqrt(
        sum(value * value for value in centered_right)
    )
    if denom == 0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(centered_left, centered_right, strict=True)
    ) / denom


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return tuple(ranks)


def _auc(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _ranks(probabilities)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "numpy is required for train_l2_logistic; install seatflow[ml]"
        ) from error
    return np
