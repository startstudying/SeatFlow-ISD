from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from seatflow.evaluation.ablations import AblationMatrix, evaluate_ablation_matrix


@dataclass(frozen=True)
class PromotionThresholds:
    min_auc_lift: float = 0.010
    min_rank_ic_lift: float = 0.015
    min_positive_fold_fraction: float = 2.0 / 3.0
    max_drawdown_degradation: float = 0.0
    min_seed_count: int = 5
    min_fold_count: int = 3


@dataclass(frozen=True)
class MetricSummary:
    candidate_model: str
    baseline_model: str
    seed_count: int
    fold_count: int
    average_auc_lift: float
    average_rank_ic_lift: float
    average_topk_excess_return_after_cost: float
    positive_fold_fraction: float
    max_drawdown_degradation: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_model": self.candidate_model,
            "baseline_model": self.baseline_model,
            "seed_count": self.seed_count,
            "fold_count": self.fold_count,
            "average_auc_lift": self.average_auc_lift,
            "average_rank_ic_lift": self.average_rank_ic_lift,
            "average_topk_excess_return_after_cost": self.average_topk_excess_return_after_cost,
            "positive_fold_fraction": self.positive_fold_fraction,
            "max_drawdown_degradation": self.max_drawdown_degradation,
        }


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    model_valid: bool
    backtest_valid: bool
    competition_ready: bool
    final_blind: bool
    release_contract_verified: bool
    negative_result: bool
    blocked_reasons: tuple[str, ...]
    competition_blocked_reasons: tuple[str, ...]
    metrics: MetricSummary | None
    ablations: AblationMatrix

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "model_valid": self.model_valid,
            "backtest_valid": self.backtest_valid,
            "competition_ready": self.competition_ready,
            "final_blind": self.final_blind,
            "release_contract_verified": self.release_contract_verified,
            "negative_result": self.negative_result,
            "blocked_reasons": list(self.blocked_reasons),
            "competition_blocked_reasons": list(self.competition_blocked_reasons),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "ablations": self.ablations.to_dict(),
        }


def evaluate_promotion_decision(
    raw_metrics: Mapping[str, Any],
    *,
    candidate_model: str | None = None,
    baseline_model: str | None = None,
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    """Produce a fail-closed promotion decision from audited rolling-fold metrics."""

    threshold_values = thresholds or PromotionThresholds()
    blocked: list[str] = []
    if not _is_audited(raw_metrics):
        blocked.append("raw_metrics_not_audited")

    candidate = candidate_model or _optional_text(raw_metrics, "candidate_model") or "dt"
    baseline = (
        baseline_model
        or _optional_text(raw_metrics, "strong_baseline")
        or _optional_text(raw_metrics, "baseline_model")
        or "strong_baseline"
    )
    final_blind = _is_final_blind(raw_metrics.get("final_blind_manifest", raw_metrics))
    if not final_blind:
        blocked.append("missing_sealed_independent_final_blind_manifest")
    if _is_contaminated(raw_metrics.get("final_blind_manifest", raw_metrics)):
        blocked.append("contaminated_oos_cannot_promote")
    release_contract_verified = _is_verified_release_contract(raw_metrics.get("release_contract"))

    try:
        rows = _metric_rows(raw_metrics)
        metrics = _summarize_metrics(rows, candidate, baseline)
    except ValueError as error:
        metrics = None
        blocked.append(str(error))

    ablations = _safe_ablation_matrix(raw_metrics.get("ablations"))
    blocked.extend(ablations.blocked_reasons)

    if metrics is not None:
        if metrics.seed_count < threshold_values.min_seed_count:
            blocked.append("single_or_missing_seed")
        if metrics.fold_count < threshold_values.min_fold_count:
            blocked.append("single_or_missing_rolling_fold")
        if metrics.average_auc_lift < threshold_values.min_auc_lift:
            blocked.append("auc_lift_below_threshold")
        if metrics.average_rank_ic_lift < threshold_values.min_rank_ic_lift:
            blocked.append("rank_ic_lift_below_threshold")
        if metrics.average_topk_excess_return_after_cost <= 0:
            blocked.append("topk_cost_adjusted_excess_return_not_positive")
        if metrics.positive_fold_fraction < threshold_values.min_positive_fold_fraction:
            blocked.append("fewer_than_two_thirds_positive_folds")
        if metrics.max_drawdown_degradation > threshold_values.max_drawdown_degradation:
            blocked.append("max_drawdown_significantly_worse")

    promoted = not blocked
    competition_ready = promoted and release_contract_verified
    competition_blocked = (
        () if competition_ready else ("missing_verified_release_contract",)
    )
    return PromotionDecision(
        promoted=promoted,
        model_valid=promoted,
        backtest_valid=promoted,
        competition_ready=competition_ready,
        final_blind=final_blind,
        release_contract_verified=release_contract_verified,
        negative_result=not promoted,
        blocked_reasons=tuple(dict.fromkeys(blocked)),
        competition_blocked_reasons=competition_blocked,
        metrics=metrics,
        ablations=ablations,
    )


def save_promotion_decision(path: str | Path, decision: PromotionDecision) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        decision.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    with NamedTemporaryFile(
        "wb", delete=False, dir=output_path.parent, prefix=f".{output_path.name}."
    ) as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    return output_path


def _summarize_metrics(
    rows: Sequence[Mapping[str, Any]],
    candidate_model: str,
    baseline_model: str,
) -> MetricSummary:
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        model = _text(row, "model")
        if model not in {candidate_model, baseline_model}:
            continue
        seed = str(row.get("seed", ""))
        fold = str(row.get("fold", row.get("fold_id", row.get("window", ""))))
        if not seed or not fold:
            raise ValueError("metric row missing seed or fold")
        models = by_key.setdefault((seed, fold), {})
        if model in models:
            raise ValueError("duplicate_model_seed_fold_metric")
        models[model] = row
    seeds = {seed for seed, _ in by_key}
    folds = {fold for _, fold in by_key}
    if not seeds or not folds:
        raise ValueError("missing_candidate_or_baseline_metric_pairs")
    missing_pairs: list[str] = []
    paired: list[tuple[tuple[str, str], Mapping[str, Any], Mapping[str, Any]]] = []
    for seed in sorted(seeds):
        for fold in sorted(folds):
            models = by_key.get((seed, fold), {})
            candidate = models.get(candidate_model)
            baseline = models.get(baseline_model)
            if candidate is None or baseline is None:
                missing_pairs.append(f"{seed}:{fold}")
                continue
            paired.append(((seed, fold), candidate, baseline))
    if missing_pairs:
        raise ValueError("incomplete_seed_fold_metric_pairs")
    if not paired:
        raise ValueError("missing_candidate_or_baseline_metric_pairs")
    auc_lifts = tuple(
        _number(candidate, "auc") - _number(baseline, "auc")
        for _, candidate, baseline in paired
    )
    rank_lifts = tuple(
        _number(candidate, "rank_ic") - _number(baseline, "rank_ic")
        for _, candidate, baseline in paired
    )
    topk_values = tuple(
        _first_number(
            candidate,
            (
                "topk_excess_return_after_cost",
                "top_k_excess_return_after_cost",
                "cost_adjusted_excess_return",
            ),
        )
        for _, candidate, _ in paired
    )
    drawdown_degradation = max(
        abs(_first_number(candidate, ("max_drawdown", "maximum_drawdown")))
        - abs(_first_number(baseline, ("max_drawdown", "maximum_drawdown")))
        for _, candidate, baseline in paired
    )
    positive_count = sum(1 for value in topk_values if value > 0)
    return MetricSummary(
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        seed_count=len(seeds),
        fold_count=len(folds),
        average_auc_lift=sum(auc_lifts) / len(auc_lifts),
        average_rank_ic_lift=sum(rank_lifts) / len(rank_lifts),
        average_topk_excess_return_after_cost=sum(topk_values) / len(topk_values),
        positive_fold_fraction=positive_count / len(topk_values),
        max_drawdown_degradation=drawdown_degradation,
    )


def _metric_rows(raw_metrics: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    rows = raw_metrics.get("rows", raw_metrics.get("metrics"))
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return tuple(_require_mapping(row, "metric row") for row in rows)
    models = raw_metrics.get("models")
    if isinstance(models, Mapping):
        expanded: list[Mapping[str, Any]] = []
        for model, value in models.items():
            if not isinstance(value, Mapping):
                continue
            model_rows = value.get("rows", value.get("metrics", value.get("folds")))
            if not isinstance(model_rows, Sequence) or isinstance(
                model_rows, (str, bytes, bytearray)
            ):
                continue
            for row in model_rows:
                item = dict(_require_mapping(row, "metric row"))
                item.setdefault("model", model)
                expanded.append(item)
        if expanded:
            return tuple(expanded)
    raise ValueError("raw_metrics_missing_rows")


def _safe_ablation_matrix(raw_ablations: object) -> AblationMatrix:
    try:
        if raw_ablations is None:
            return evaluate_ablation_matrix(None)
        if isinstance(raw_ablations, Mapping):
            return evaluate_ablation_matrix(raw_ablations)
        if isinstance(raw_ablations, Sequence) and not isinstance(
            raw_ablations, (str, bytes, bytearray)
        ):
            rows = tuple(_require_mapping(row, "ablation row") for row in raw_ablations)
            return evaluate_ablation_matrix(rows)
    except ValueError as error:
        return AblationMatrix(False, (), (str(error),))
    return AblationMatrix(False, (), ("invalid_ablation_matrix",))


def _is_audited(raw_metrics: Mapping[str, Any]) -> bool:
    audit = raw_metrics.get("audit")
    return raw_metrics.get("audited") is True or (
        isinstance(audit, Mapping) and audit.get("verified") is True
    )


def _is_final_blind(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    split_status = value.get("split_status", value.get("blind_status"))
    return (
        split_status == "future_blind"
        and value.get("verified") is True
        and value.get("sealed") is True
        and value.get("independent") is True
    )


def _is_contaminated(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("split_status") == "contaminated_oos"


def _is_verified_release_contract(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("verified") is True


def _optional_text(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metric row missing {key}")
    return value


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric row {key} must be numeric")
    return float(value)


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    for key in keys:
        if key in row:
            return _number(row, key)
    raise ValueError(f"metric row missing one of {', '.join(keys)}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
