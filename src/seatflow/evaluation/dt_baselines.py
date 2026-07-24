from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from math import expm1, isfinite

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.research import (
    MatrixDataset,
    ResearchMetrics,
    evaluate_predictions,
    predict_proba,
    train_l2_logistic,
)

DT_BASELINE_NAME = "dt_event_seat_profile_logistic"
EVENT_FEATURE_NAMES: tuple[str, ...] = (
    "event_buy_amount",
    "event_sell_amount",
    "event_net_amount",
    "event_count",
)
PROFILE_FEATURE_NAMES: tuple[str, ...] = (
    "seat_profile_available",
    "seat_profile_event_count",
    "seat_profile_buy_share",
    "seat_profile_sell_share",
    "seat_profile_positive_rate",
    "seat_profile_avg_signed_return",
    "seat_profile_amount_weighted_signed_return",
    "seat_profile_avg_horizon",
)
DT_BASELINE_FEATURE_NAMES = EVENT_FEATURE_NAMES + PROFILE_FEATURE_NAMES


@dataclass(frozen=True)
class DTBaselineScore:
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
            "model": self.baseline_name,
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
class DTBaselineEvaluation:
    baseline_name: str
    seed: int
    fold_id: str
    scores: tuple[DTBaselineScore, ...]
    metrics: ResearchMetrics
    top_k: int
    top_k_mean_return: float
    max_drawdown: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.baseline_name,
            "baseline_name": self.baseline_name,
            "seed": self.seed,
            "fold": self.fold_id,
            "metrics": self.metrics.to_dict(),
            "top_k": self.top_k,
            "top_k_mean_return": round(self.top_k_mean_return, 12),
            "max_drawdown": round(self.max_drawdown, 12),
            "scores": [score.to_dict() for score in self.scores],
        }


def run_dt_graph_baseline(
    train_samples: Sequence[TensorizedWindowSample],
    test_samples: Sequence[TensorizedWindowSample],
    *,
    seed: int,
    fold_id: str = "",
    top_k: int = 5,
    epochs: int = 300,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> DTBaselineEvaluation:
    """Fit the fixed event/profile baseline on the DT label and graph contract."""

    if not train_samples or not test_samples:
        raise ValueError("DT baseline train and test samples must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    train = _matrix(train_samples)
    test = _matrix(test_samples)
    model = train_l2_logistic(
        train,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    probabilities = predict_proba(model, test.x)
    scores = tuple(
        DTBaselineScore(
            baseline_name=DT_BASELINE_NAME,
            seed=seed,
            sample_id=sample.sample_id,
            as_of_date=_sample_date(sample),
            stock_id=sample.target_stock_id,
            score=score,
            label=sample.labels[1],
            label_return=sample.returns[1],
        )
        for sample, score in zip(test_samples, probabilities, strict=True)
    )
    metrics = evaluate_predictions(
        probabilities,
        test.y,
        test.label_returns,
    )
    return DTBaselineEvaluation(
        baseline_name=DT_BASELINE_NAME,
        seed=seed,
        fold_id=fold_id,
        scores=scores,
        metrics=metrics,
        top_k=top_k,
        top_k_mean_return=_top_k_mean_return(scores, top_k),
        max_drawdown=_max_drawdown(scores, top_k),
    )


def _matrix(samples: Sequence[TensorizedWindowSample]) -> MatrixDataset:
    rows = tuple(_feature_vector(sample) for sample in samples)
    return MatrixDataset(
        sample_ids=tuple(sample.sample_id for sample in samples),
        as_of_dates=tuple(_sample_date(sample) for sample in samples),
        feature_names=DT_BASELINE_FEATURE_NAMES,
        x=rows,
        y=tuple(int(sample.labels[1]) for sample in samples),
        label_returns=tuple(float(sample.returns[1]) for sample in samples),
    )


def _feature_vector(sample: TensorizedWindowSample) -> tuple[float, ...]:
    """Aggregate only edges present in the as-of graph into an auditable vector."""

    data = sample.graph.data
    metadata = sample.graph.metadata
    raw_schema = metadata.get("edge_attr_schema", {})
    edge_attr_schema = raw_schema if isinstance(raw_schema, dict) else {}
    profile_schema_raw = metadata.get("feature_schema", {})
    profile_schema = (
        profile_schema_raw.get("seat", [])
        if isinstance(profile_schema_raw, dict)
        else []
    )
    profile_indexes = {
        name: index
        for index, name in enumerate(profile_schema)
        if isinstance(name, str)
    }
    event_buy = 0.0
    event_sell = 0.0
    event_count = 0.0
    profile_sums = {name: 0.0 for name in PROFILE_FEATURE_NAMES}
    profile_weight = 0.0
    target_index = int(sample.target_node_index)

    for edge_type in data.edge_types:
        source_type, relation, target_type = edge_type
        if target_type != "stock" or source_type != "seat":
            continue
        store = data[edge_type]
        edge_index = store.edge_index.detach().cpu()
        edge_attr = store.edge_attr.detach().cpu()
        schema = edge_attr_schema.get("::".join(edge_type), ())
        if not isinstance(schema, (list, tuple)):
            raise ValueError(f"missing edge attribute schema for {edge_type}")
        indexes = {str(name): index for index, name in enumerate(schema)}
        direction_index = indexes.get("direction")
        amount_index = indexes.get("amount")
        availability_index = indexes.get("seat_profile_available")
        gate_index = indexes.get("graph_edge_weight", indexes.get("edge_gate_input"))
        for row in range(int(edge_index.size(1))):
            if int(edge_index[1, row]) != target_index:
                continue
            direction = (
                float(edge_attr[row, direction_index])
                if direction_index is not None
                else (1.0 if relation == "buys" else -1.0)
            )
            amount = (
                _raw_amount(float(edge_attr[row, amount_index]))
                if amount_index is not None
                else 1.0
            )
            signed_amount = amount if direction > 0 else -amount
            event_buy += max(signed_amount, 0.0)
            event_sell += max(-signed_amount, 0.0)
            event_count += 1.0

            source_index = int(edge_index[0, row])
            available = (
                float(edge_attr[row, availability_index])
                if availability_index is not None
                else 0.0
            )
            weight = (
                abs(float(edge_attr[row, gate_index]))
                if gate_index is not None
                else 1.0
            )
            if available <= 0.0 or "seat" not in data.node_types:
                continue
            seat_features = data["seat"].x.detach().cpu()[source_index]
            profile_weight += max(weight, 1e-6)
            profile_sums["seat_profile_available"] += available * max(weight, 1e-6)
            for name in PROFILE_FEATURE_NAMES[1:]:
                index = profile_indexes.get(name)
                if index is not None and index < int(seat_features.numel()):
                    profile_sums[name] += float(seat_features[index]) * max(weight, 1e-6)

    profile = tuple(
        value / profile_weight if profile_weight > 0 else 0.0
        for value in (profile_sums[name] for name in PROFILE_FEATURE_NAMES)
    )
    values = (event_buy, event_sell, event_buy - event_sell, event_count) + profile
    if any(not isfinite(value) for value in values):
        raise ValueError(f"non-finite DT baseline features for {sample.sample_id}")
    return values


def _raw_amount(tensor_value: float) -> float:
    """Invert tensorization's signed-log amount before cross-edge aggregation."""

    return expm1(abs(tensor_value))


def _sample_date(sample: TensorizedWindowSample) -> date:
    if sample.as_of_date is None:
        raise ValueError(f"sample {sample.sample_id} is missing as_of_date")
    return sample.as_of_date


def _top_k_mean_return(scores: Sequence[DTBaselineScore], top_k: int) -> float:
    daily = _daily_selected_returns(scores, top_k)
    if not daily:
        raise ValueError("scores must contain at least one date")
    return sum(daily) / len(daily)


def _max_drawdown(scores: Sequence[DTBaselineScore], top_k: int) -> float:
    equity = 1.0
    peak = equity
    worst = 0.0
    for value in _daily_selected_returns(scores, top_k):
        equity *= 1.0 + value
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def _daily_selected_returns(scores: Sequence[DTBaselineScore], top_k: int) -> tuple[float, ...]:
    by_date: dict[date, list[DTBaselineScore]] = {}
    for score in scores:
        by_date.setdefault(score.as_of_date, []).append(score)
    returns: list[float] = []
    for day in sorted(by_date):
        selected = sorted(
            by_date[day], key=lambda item: (-item.score, item.sample_id)
        )[:top_k]
        returns.append(sum(item.label_return for item in selected) / len(selected))
    return tuple(returns)
