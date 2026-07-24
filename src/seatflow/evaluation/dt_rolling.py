from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, timedelta
from hashlib import sha256
from math import exp
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TypedDict

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.evaluation.dt_baselines import (
    DT_BASELINE_NAME,
    DTBaselineEvaluation,
    run_dt_graph_baseline,
)
from seatflow.evaluation.portfolio_costs import ranked_portfolio_cost_summary
from seatflow.evaluation.rolling import DEFAULT_SEEDS
from seatflow.graph_transforms import (
    FoldGraphTransform,
    GraphTransformError,
    fit_apply_fold_graph_transform,
)
from seatflow.research import ResearchMetrics, evaluate_predictions
from seatflow.return_transforms import (
    ReturnTransformState,
    fit_return_transform,
    return_transform_diagnostics,
)
from seatflow.training.dtseathgt import (
    DTSeatHGTTrainingConfig,
    prediction_snapshot_for_samples,
    restore_dt_seat_hgt_seed,
    train_dt_seat_hgt,
)


class DTScore(TypedDict):
    model: str
    seed: int
    fold: str
    sample_id: str
    as_of_date: str
    stock_id: str
    score: float
    direction_probability: float
    expected_return: float
    selection_score: float
    selection_score_policy: str
    label: int
    label_return: float


@dataclass(frozen=True)
class DTRollingFold:
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train: tuple[TensorizedWindowSample, ...]
    test: tuple[TensorizedWindowSample, ...]

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
class DTSeedFoldEvaluation:
    seed: int
    fold_id: str
    checkpoint_path: Path
    metrics: ResearchMetrics
    top_k: int
    top_k_mean_return: float
    scores: tuple[DTScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": "dt_seathgt",
            "seed": self.seed,
            "fold": self.fold_id,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics": self.metrics.to_dict(),
            "top_k": self.top_k,
            "top_k_mean_return": round(self.top_k_mean_return, 12),
            "scores": list(self.scores),
        }


@dataclass(frozen=True)
class DTRollingEvaluation:
    selection_horizon: int
    seeds: tuple[int, ...]
    folds: tuple[DTRollingFold, ...]
    results: tuple[DTSeedFoldEvaluation, ...]
    ablations: tuple[dict[str, object], ...]
    baseline_results: tuple[DTBaselineEvaluation, ...] = ()

    def to_dict(self) -> dict[str, object]:
        paired_rows = self._paired_rows()
        return {
            "model": "dt_seathgt",
            "selection_horizon": self.selection_horizon,
            "seeds": list(self.seeds),
            "folds": [fold.to_dict() for fold in self.folds],
            "results": [result.to_dict() for result in self.results],
            "baseline_results": [result.to_dict() for result in self.baseline_results],
            "ablations": list(self.ablations),
            "rows": [
                {
                    "model": "dt_seathgt",
                    "seed": result.seed,
                    "fold": result.fold_id,
                    "auc": result.metrics.auc,
                    "rank_ic": result.metrics.rank_ic,
                    "top_k_mean_return": result.top_k_mean_return,
                }
                for result in self.results
            ],
            "paired_rows": paired_rows,
        }

    def _paired_rows(self) -> list[dict[str, object]]:
        if not self.baseline_results:
            return []
        baseline_by_key = {
            (result.seed, result.fold_id): result
            for result in self.baseline_results
        }
        rows: list[dict[str, object]] = []
        for result in self.results:
            baseline = baseline_by_key.get((result.seed, result.fold_id))
            if baseline is None:
                raise ValueError(
                    f"missing DT baseline pair for seed={result.seed}, fold={result.fold_id}"
                )
            rows.append(
                _candidate_metric_row(result, baseline)
            )
            rows.append(
                {
                    "model": DT_BASELINE_NAME,
                    "seed": baseline.seed,
                    "fold": result.fold_id,
                    "auc": baseline.metrics.auc,
                    "rank_ic": baseline.metrics.rank_ic,
                    "topk_excess_return_after_cost": 0.0,
                    "max_drawdown": baseline.max_drawdown,
                }
            )
        return rows


def _validate_seed_policy(seeds: tuple[int, ...], *, diagnostic_e1: bool) -> None:
    if len(set(seeds)) != len(seeds):
        raise ValueError("DT-SeatHGT rolling evaluation requires unique seeds")
    expected_count = 1 if diagnostic_e1 else 5
    if len(seeds) != expected_count:
        mode = "E1 diagnostic" if diagnostic_e1 else "DT-SeatHGT rolling evaluation"
        raise ValueError(f"{mode} must use exactly {expected_count} unique seeds")


def save_dt_rolling_result(
    path: str | Path,
    evaluation: DTRollingEvaluation,
    *,
    audit: dict[str, object],
    split_status: str,
) -> Path:
    paired_rows = evaluation._paired_rows()
    payload = {
        "schema_version": 1,
        "status": "evaluated_fail_closed",
        "candidate_model": "dt_seathgt",
        "strong_baseline": DT_BASELINE_NAME,
        "model_valid": False,
        "negative_result": True,
        "audited": audit.get("verified") is True,
        "audit": audit,
        "split_status": split_status,
        "final_blind_manifest": None,
        "metrics": paired_rows,
        **evaluation.to_dict(),
    }
    return _atomic_json_write(Path(path), payload)


def save_dt_rolling_failure(
    path: str | Path,
    *,
    error: BaseException,
    audit: dict[str, object],
    split_status: str,
) -> Path:
    payload = {
        "schema_version": 1,
        "status": "failed_closed",
        "candidate_model": "dt_seathgt",
        "model_valid": False,
        "negative_result": True,
        "audited": False,
        "audit": audit,
        "split_status": split_status,
        "final_blind_manifest": None,
        "failure_type": type(error).__name__,
        "failure_reason": str(error),
        "rows": [],
        "ablations": [],
    }
    return _atomic_json_write(Path(path), payload)


def build_dt_rolling_folds(
    samples: Sequence[TensorizedWindowSample],
    *,
    selection_horizon: int,
    n_folds: int = 3,
    min_train_dates: int = 3,
    purge_days: int = 0,
    embargo_days: int = 0,
) -> tuple[DTRollingFold, ...]:
    """Create expanding folds using only labels fully known before each test window."""

    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    if min_train_dates < 1:
        raise ValueError("min_train_dates must be positive")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")
    ordered = tuple(sorted(samples, key=lambda item: (_as_of_date(item), item.sample_id)))
    if not ordered:
        raise ValueError("DT-SeatHGT rolling evaluation requires samples")
    _require_horizon(ordered, selection_horizon)
    max_declared_horizon = max(
        horizon for sample in ordered for horizon in sample.label_end_dates
    )
    effective_purge_days = max(purge_days, max_declared_horizon)
    effective_embargo_days = max(embargo_days, max_declared_horizon)
    dates = tuple(sorted({_as_of_date(sample) for sample in ordered}))
    if len(dates) < min_train_dates + n_folds:
        raise ValueError("not enough unique dates for requested rolling folds")

    folds: list[DTRollingFold] = []
    for index, test_dates in enumerate(_contiguous_chunks(dates[min_train_dates:], n_folds)):
        test_start = test_dates[0]
        test_end = test_dates[-1]
        train_as_of_cutoff = test_start - timedelta(days=effective_embargo_days)
        train_label_cutoff = test_start - timedelta(days=effective_purge_days)
        train = tuple(
            sample
            for sample in ordered
            if _as_of_date(sample) < train_as_of_cutoff
            and _label_end_date(sample, selection_horizon) < train_label_cutoff
        )
        test = tuple(
            sample for sample in ordered if test_start <= _as_of_date(sample) <= test_end
        )
        if not train:
            raise ValueError(f"rolling fold {index} has empty train partition")
        if not test:
            raise ValueError(f"rolling fold {index} has empty test partition")
        folds.append(
            DTRollingFold(
                fold_id=f"fold_{index}",
                train_start=_as_of_date(train[0]),
                train_end=max(_as_of_date(sample) for sample in train),
                test_start=test_start,
                test_end=test_end,
                train=train,
                test=test,
            )
        )
    return tuple(folds)


def run_dt_seathgt_rolling_evaluation(
    samples: Sequence[TensorizedWindowSample],
    config: DTSeatHGTTrainingConfig,
    output_root: str | Path,
    *,
    selection_horizon: int = 1,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_folds: int = 3,
    min_train_dates: int = 3,
    purge_days: int = 0,
    embargo_days: int = 0,
    top_k: int = 5,
    diagnostic_e1: bool = False,
) -> DTRollingEvaluation:
    _validate_seed_policy(seeds, diagnostic_e1=diagnostic_e1)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if selection_horizon not in config.model.horizons:
        raise ValueError("selection_horizon must be declared by the DT-SeatHGT model")
    if config.model.direction_classes != 2:
        raise ValueError(
            "V2 three-way direction labels require the dedicated V2 rolling evaluator"
        )
    max_horizon = max(config.model.horizons)
    effective_purge_days = max(purge_days, max_horizon)
    effective_embargo_days = max(embargo_days, max_horizon)
    folds = build_dt_rolling_folds(
        samples,
        selection_horizon=selection_horizon,
        n_folds=n_folds,
        min_train_dates=min_train_dates,
        purge_days=effective_purge_days,
        embargo_days=effective_embargo_days,
    )
    return run_dt_seathgt_folds_evaluation(
        folds,
        config,
        output_root,
        selection_horizon=selection_horizon,
        seeds=seeds,
        top_k=top_k,
        diagnostic_e1=diagnostic_e1,
    )


def run_dt_seathgt_folds_evaluation(
    folds: Sequence[DTRollingFold],
    config: DTSeatHGTTrainingConfig,
    output_root: str | Path,
    *,
    selection_horizon: int = 1,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    top_k: int = 5,
    diagnostic_e1: bool = False,
) -> DTRollingEvaluation:
    """Evaluate already-materialized folds without retaining unrelated folds."""

    _validate_seed_policy(seeds, diagnostic_e1=diagnostic_e1)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if selection_horizon not in config.model.horizons:
        raise ValueError("selection_horizon must be declared by the DT-SeatHGT model")
    if not folds:
        raise ValueError("DT-SeatHGT rolling evaluation requires folds")
    root = Path(output_root)
    results: list[DTSeedFoldEvaluation] = []
    ablations: list[dict[str, object]] = []
    baseline_results: list[DTBaselineEvaluation] = []
    transform_root, diagnostics_root = _fold_artifact_roots(root)
    for fold in folds:
        transformed, validation_transformed = _transform_fold_with_validation(fold)
        return_transform = fit_return_transform(transformed.train)
        return_diagnostics: dict[str, object] = {
            "train": return_transform_diagnostics(transformed.train, return_transform),
            "validation": return_transform_diagnostics(
                validation_transformed, return_transform
            ),
            "test": return_transform_diagnostics(transformed.test, return_transform),
        }
        return_diagnostics["diagnostics_hash"] = sha256(
            json.dumps(return_diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _atomic_json_write(
            transform_root / fold.fold_id / "feature_transform.json",
            transformed.state.to_dict(),
        )
        _atomic_json_write(
            diagnostics_root / fold.fold_id / "feature_distribution.json",
            transformed.diagnostics,
        )
        _atomic_json_write(
            transform_root / fold.fold_id / "return_transform.json",
            return_transform.to_dict(),
        )
        _atomic_json_write(
            diagnostics_root / fold.fold_id / "return_distribution.json",
            return_diagnostics,
        )
        for seed in seeds:
            seed_config = replace(config, seeds=(seed,))
            training = train_dt_seat_hgt(
                transformed.train,
                seed_config,
                root,
                run_id=f"dtseathgt-{fold.fold_id}-seed-{seed}",
                resume_existing=True,
                feature_transform_state=transformed.state,
                feature_transform_diagnostics=transformed.diagnostics,
                return_transform_state=return_transform,
                return_transform_diagnostics=return_diagnostics,
                validation_samples=validation_transformed,
            )
            restored = restore_dt_seat_hgt_seed(
                training.checkpoint_path,
                transformed.test[0],
                seed,
                feature_transform_state=transformed.state,
                return_transform_state=return_transform,
            )
            snapshot = prediction_snapshot_for_samples(
                restored.model,
                transformed.test,
                config.model.horizons,
                return_transform_state=return_transform,
            )
            scores = _scores(transformed.test, snapshot, selection_horizon, seed, fold.fold_id)
            probabilities = tuple(float(score["direction_probability"]) for score in scores)
            labels = tuple(int(score["label"]) for score in scores)
            returns = tuple(float(score["label_return"]) for score in scores)
            full_metrics = evaluate_predictions(probabilities, labels, returns)
            results.append(
                DTSeedFoldEvaluation(
                    seed=seed,
                    fold_id=fold.fold_id,
                    checkpoint_path=training.checkpoint_path,
                    metrics=full_metrics,
                    top_k=top_k,
                    top_k_mean_return=_top_k_mean_return(scores, top_k),
                    scores=scores,
                )
            )
            baseline_results.append(
                run_dt_graph_baseline(
                    transformed.train,
                    transformed.test,
                    seed=seed,
                    fold_id=fold.fold_id,
                    top_k=top_k,
                )
            )
            for group in ("seat_profile", "time_edges", "direction_edges"):
                ablated_snapshot = _prediction_snapshot_with_ablation(
                    restored.model,
                    transformed.test,
                    config.model.horizons,
                    group,
                    return_transform,
                )
                ablated_scores = _scores(
                    transformed.test,
                    ablated_snapshot,
                    selection_horizon,
                    seed,
                    fold.fold_id,
                )
                ablated_metrics = evaluate_predictions(
                    tuple(score["direction_probability"] for score in ablated_scores),
                    tuple(score["label"] for score in ablated_scores),
                    tuple(score["label_return"] for score in ablated_scores),
                )
                ablations.append(
                    _ablation_row(
                        group,
                        seed,
                        fold.fold_id,
                        full_metrics,
                        ablated_metrics,
                        scores,
                        ablated_scores,
                    )
                )
    return DTRollingEvaluation(
        selection_horizon=selection_horizon,
        seeds=seeds,
        folds=tuple(folds),
        results=tuple(results),
        ablations=tuple(ablations),
        baseline_results=tuple(baseline_results),
    )


def _fold_artifact_roots(root: Path) -> tuple[Path, Path]:
    """Honor canonical artifact locations for normal diagnostics runs."""
    if root.parent.name == "diagnostics":
        artifacts_root = root.parent.parent
        return artifacts_root / "transforms" / root.name, root
    return root / "transforms", root / "diagnostics"


def _candidate_metric_row(
    result: DTSeedFoldEvaluation,
    baseline: DTBaselineEvaluation,
) -> dict[str, object]:
    candidate_costs = ranked_portfolio_cost_summary(
        result.scores,
        top_k=result.top_k,
        date_field="as_of_date",
        stock_field="stock_id",
        score_field="selection_score",
    )
    baseline_rows = [score.to_dict() for score in baseline.scores]
    baseline_costs = ranked_portfolio_cost_summary(
        baseline_rows,
        top_k=baseline.top_k,
        date_field="as_of_date",
        stock_field="stock_id",
        score_field="score",
    )
    candidate_net = result.top_k_mean_return - candidate_costs["total_cost"]
    baseline_net = baseline.top_k_mean_return - baseline_costs["total_cost"]
    return {
        "model": "dt_seathgt",
        "seed": result.seed,
        "fold": result.fold_id,
        "auc": result.metrics.auc,
        "rank_ic": result.metrics.rank_ic,
        "topk_excess_return_after_cost": candidate_net - baseline_net,
        "candidate_net_return": candidate_net,
        "baseline_net_return": baseline_net,
        "candidate_costs": candidate_costs,
        "baseline_costs": baseline_costs,
        "max_drawdown": _max_drawdown(result.scores, result.top_k),
        "baseline_max_drawdown": baseline.max_drawdown,
    }


def _max_drawdown(scores: Sequence[DTScore], top_k: int) -> float:
    by_date: dict[str, list[DTScore]] = {}
    for score in scores:
        by_date.setdefault(str(score["as_of_date"]), []).append(score)
    equity = 1.0
    peak = equity
    worst = 0.0
    for day in sorted(by_date):
        selected = sorted(
            by_date[day],
            key=lambda item: (-float(item["selection_score"]), str(item["sample_id"])),
        )[:top_k]
        daily_return = sum(float(item["label_return"]) for item in selected) / len(selected)
        equity *= 1.0 + daily_return
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, equity / peak - 1.0)
    return abs(worst)


def _scores(
    samples: Sequence[TensorizedWindowSample],
    snapshot: dict[str, object],
    horizon: int,
    seed: int,
    fold_id: str,
) -> tuple[DTScore, ...]:
    scores: list[DTScore] = []
    for sample in samples:
        raw = snapshot.get(sample.sample_id)
        if not isinstance(raw, dict):
            raise ValueError(f"DT-SeatHGT prediction snapshot is missing {sample.sample_id}")
        direction_logits = raw.get("direction_logits")
        if not isinstance(direction_logits, dict):
            raise ValueError("DT-SeatHGT prediction snapshot is malformed")
        return_predictions = raw.get("return_predictions")
        if not isinstance(return_predictions, dict):
            raise ValueError("DT-SeatHGT prediction snapshot lacks return head for selection")
        logit = direction_logits.get(str(horizon))
        expected_return = return_predictions.get(str(horizon))
        if not isinstance(logit, float):
            raise ValueError("DT-SeatHGT prediction snapshot lacks selection horizon")
        if not isinstance(expected_return, float):
            raise ValueError("DT-SeatHGT prediction snapshot lacks return head for selection")
        direction_probability = _sigmoid(logit)
        scores.append(
            {
                "model": "dt_seathgt",
                "seed": seed,
                "fold": fold_id,
                "sample_id": sample.sample_id,
                "as_of_date": _as_of_date(sample).isoformat(),
                "stock_id": sample.target_stock_id,
                "score": direction_probability,
                "direction_probability": direction_probability,
                "expected_return": expected_return,
                "selection_score": expected_return,
                "selection_score_policy": "return_head_expected_return",
                "label": sample.labels[horizon],
                "label_return": sample.returns[horizon],
            }
        )
    return tuple(scores)


def _top_k_mean_return(scores: Sequence[DTScore], top_k: int) -> float:
    if not scores:
        raise ValueError("scores must not be empty")
    daily: dict[str, list[DTScore]] = {}
    for score in scores:
        daily.setdefault(str(score["as_of_date"]), []).append(score)
    returns: list[float] = []
    for value in daily.values():
        selected = sorted(
            value, key=lambda item: (-item["selection_score"], item["sample_id"])
        )[:top_k]
        returns.append(sum(item["label_return"] for item in selected) / len(selected))
    return sum(returns) / len(returns)


def _prediction_snapshot_with_ablation(
    model: Any,
    samples: Sequence[TensorizedWindowSample],
    horizons: Sequence[int],
    group: str,
    return_transform_state: ReturnTransformState | None = None,
) -> dict[str, object]:
    if group not in {"seat_profile", "time_edges", "direction_edges"}:
        raise ValueError(f"unsupported DT-SeatHGT ablation group: {group}")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("DT-SeatHGT ablation requires PyTorch") from error
    snapshot: dict[str, object] = {}
    model.eval()
    with _temporarily_ablated(samples, group):
        with torch.no_grad():
            for batch in _group_samples_by_graph(samples):
                outputs = model(batch[0].graph.data)
                for sample in batch:
                    snapshot[sample.sample_id] = {
                        "direction_logits": {
                            str(horizon): float(
                                outputs["direction_logits"][horizon][sample.target_node_index]
                                .detach()
                                .cpu()
                            )
                            for horizon in horizons
                        },
                        "return_predictions": {
                            str(horizon): _inverse_return_prediction(
                                float(
                                    outputs["return_predictions"][horizon][
                                        sample.target_node_index
                                    ]
                                    .detach()
                                    .cpu()
                                ),
                                horizon,
                                return_transform_state,
                            )
                            for horizon in horizons
                        },
                    }
    return snapshot


def _inverse_return_prediction(
    value: float, horizon: int, state: ReturnTransformState | None
) -> float:
    return state.inverse_transform(horizon, value) if state is not None else value


@contextmanager
def _temporarily_ablated(
    samples: Sequence[TensorizedWindowSample], group: str
) -> Iterator[None]:
    originals: list[tuple[Any, str, Any]] = []
    seen_graphs: set[int] = set()
    try:
        for batch in _group_samples_by_graph(samples):
            graph = batch[0].graph
            if id(graph) in seen_graphs:
                continue
            seen_graphs.add(id(graph))
            data = graph.data
            attribute_names: tuple[str, ...]
            if group == "seat_profile":
                _ablate_seat_node_profile(data, graph.metadata, originals)
                attribute_names = ("seat_profile_available",)
            elif group == "time_edges":
                attribute_names = ("delta_days", "time_decay")
            else:
                attribute_names = ("direction",)
            _ablate_edge_attributes(data, graph.metadata, attribute_names, originals)
        yield
    finally:
        for owner, attribute, value in reversed(originals):
            setattr(owner, attribute, value)


def _ablate_seat_node_profile(
    data: Any,
    metadata: dict[str, object],
    originals: list[tuple[Any, str, Any]],
) -> None:
    feature_schema = metadata.get("feature_schema")
    if not isinstance(feature_schema, dict):
        raise ValueError("tensorized graph metadata is missing feature_schema")
    seat_schema = feature_schema.get("seat")
    if not isinstance(seat_schema, list):
        raise ValueError("tensorized graph metadata is missing seat feature schema")
    indexes = [
        index for index, name in enumerate(seat_schema) if str(name).startswith("seat_profile")
    ]
    if not indexes:
        return
    originals.append((data["seat"], "x", data["seat"].x))
    ablated = data["seat"].x.clone()
    ablated[:, indexes] = 0.0
    data["seat"].x = ablated


def _ablate_edge_attributes(
    data: Any,
    metadata: dict[str, object],
    attribute_names: Sequence[str],
    originals: list[tuple[Any, str, Any]],
) -> None:
    schema_by_edge = metadata.get("edge_attr_schema")
    if not isinstance(schema_by_edge, dict):
        raise ValueError("tensorized graph metadata is missing edge_attr_schema")
    for edge_type in data.edge_types:
        key = "::".join(edge_type)
        schema = schema_by_edge.get(key)
        if not isinstance(schema, list):
            raise ValueError(f"tensorized graph metadata is missing edge schema for {key}")
        indexes = [index for index, name in enumerate(schema) if name in attribute_names]
        if not indexes:
            continue
        store = data[edge_type]
        originals.append((store, "edge_attr", store.edge_attr))
        ablated = store.edge_attr.clone()
        ablated[:, indexes] = 0.0
        store.edge_attr = ablated


def _ablation_row(
    group: str,
    seed: int,
    fold_id: str,
    full: ResearchMetrics,
    ablated: ResearchMetrics,
    full_scores: Sequence[DTScore],
    ablated_scores: Sequence[DTScore],
) -> dict[str, object]:
    if [score["sample_id"] for score in full_scores] != [
        score["sample_id"] for score in ablated_scores
    ]:
        raise ValueError("DT-SeatHGT ablation scores do not align with full scores")
    score_deltas = tuple(
        abs(full_score["score"] - ablated_score["score"])
        for full_score, ablated_score in zip(
            full_scores, ablated_scores, strict=True
        )
    )
    return {
        "group": group,
        "seed": seed,
        "fold": fold_id,
        "full_auc": full.auc,
        "ablated_auc": ablated.auc,
        "auc_contribution": _metric_contribution(full.auc, ablated.auc),
        "full_rank_ic": full.rank_ic,
        "ablated_rank_ic": ablated.rank_ic,
        "rank_ic_contribution": _metric_contribution(full.rank_ic, ablated.rank_ic),
        "mean_abs_score_delta": sum(score_deltas) / len(score_deltas),
        "max_abs_score_delta": max(score_deltas),
    }


def _metric_contribution(full: float | None, ablated: float | None) -> float | None:
    if full is None or ablated is None:
        return None
    return full - ablated


def _group_samples_by_graph(
    samples: Sequence[TensorizedWindowSample],
) -> tuple[tuple[TensorizedWindowSample, ...], ...]:
    grouped: dict[int, list[TensorizedWindowSample]] = {}
    for sample in samples:
        grouped.setdefault(id(sample.graph), []).append(sample)
    return tuple(tuple(batch) for batch in grouped.values())


def _require_horizon(samples: Sequence[TensorizedWindowSample], horizon: int) -> None:
    for sample in samples:
        if horizon not in sample.labels or horizon not in sample.returns:
            raise ValueError(f"sample {sample.sample_id} is missing selection horizon labels")
        _label_end_date(sample, horizon)


def _as_of_date(sample: TensorizedWindowSample) -> date:
    if sample.as_of_date is None:
        raise ValueError(f"sample {sample.sample_id} is missing as_of_date")
    return sample.as_of_date


def _split_fold_validation(
    samples: Sequence[TensorizedWindowSample],
    validation_date_count: int = 1,
) -> tuple[tuple[TensorizedWindowSample, ...], tuple[TensorizedWindowSample, ...]]:
    """Reserve the latest chronological training dates for model monitoring."""

    ordered_dates = tuple(sorted({_as_of_date(sample) for sample in samples}))
    if len(ordered_dates) < 2:
        raise ValueError("a rolling fold needs at least two training dates for validation")
    if validation_date_count < 1 or validation_date_count >= len(ordered_dates):
        raise ValueError("validation date count must leave at least one training date")
    validation_dates = set(ordered_dates[-validation_date_count:])
    train = tuple(sample for sample in samples if _as_of_date(sample) not in validation_dates)
    validation = tuple(sample for sample in samples if _as_of_date(sample) in validation_dates)
    if not train or not validation:
        raise ValueError("fold-local validation split must contain training and validation samples")
    return train, validation


def _transform_fold_with_validation(
    fold: DTRollingFold,
) -> tuple[FoldGraphTransform, tuple[TensorizedWindowSample, ...]]:
    """Pick the latest validation window whose schema is supported by earlier data."""

    date_count = len({_as_of_date(sample) for sample in fold.train})
    failures: list[str] = []
    for held_out_dates in range(1, date_count):
        training_samples, validation_samples = _split_fold_validation(
            fold.train, held_out_dates
        )
        holdout = (*validation_samples, *fold.test)
        try:
            transformed_holdout = fit_apply_fold_graph_transform(training_samples, holdout)
        except GraphTransformError as error:
            failures.append(str(error))
            continue
        validation_count = len(validation_samples)
        return (
            FoldGraphTransform(
                state=transformed_holdout.state,
                train=transformed_holdout.train,
                test=transformed_holdout.test[validation_count:],
                diagnostics=transformed_holdout.diagnostics,
            ),
            transformed_holdout.test[:validation_count],
        )
    detail = failures[-1] if failures else "no chronological validation candidate"
    raise ValueError(f"fold {fold.fold_id} has no schema-supported validation split: {detail}")


def _label_end_date(sample: TensorizedWindowSample, horizon: int) -> date:
    value = sample.label_end_dates.get(horizon)
    if not isinstance(value, date):
        raise ValueError(
            f"sample {sample.sample_id} is missing label_end_date for horizon {horizon}"
        )
    return value


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def _atomic_json_write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
    ).encode("utf-8")
    with NamedTemporaryFile(
        "wb", delete=False, dir=path.parent, prefix=f".{path.name}."
    ) as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
    return path
