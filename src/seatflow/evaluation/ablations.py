from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REQUIRED_ABLATION_GROUPS = ("seat_profile", "time_edges", "direction_edges")
_GROUP_ALIASES = {
    "seat": "seat_profile",
    "seat_profile": "seat_profile",
    "seat_profiles": "seat_profile",
    "profile": "seat_profile",
    "time": "time_edges",
    "time_edge": "time_edges",
    "time_edges": "time_edges",
    "temporal": "time_edges",
    "direction": "direction_edges",
    "direction_edge": "direction_edges",
    "direction_edges": "direction_edges",
    "directional": "direction_edges",
}


@dataclass(frozen=True)
class AblationThresholds:
    min_auc_contribution: float = 0.0
    min_rank_ic_contribution: float = 0.0
    min_positive_fold_fraction: float = 2.0 / 3.0


@dataclass(frozen=True)
class AblationResult:
    group: str
    passed: bool
    average_auc_contribution: float
    average_rank_ic_contribution: float
    positive_fold_fraction: float
    fold_count: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "passed": self.passed,
            "average_auc_contribution": self.average_auc_contribution,
            "average_rank_ic_contribution": self.average_rank_ic_contribution,
            "positive_fold_fraction": self.positive_fold_fraction,
            "fold_count": self.fold_count,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AblationMatrix:
    passed: bool
    results: tuple[AblationResult, ...]
    blocked_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
            "blocked_reasons": list(self.blocked_reasons),
        }


def evaluate_ablation_matrix(
    raw_ablations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    required_groups: Sequence[str] = REQUIRED_ABLATION_GROUPS,
    thresholds: AblationThresholds | None = None,
) -> AblationMatrix:
    """Validate fixed ablations for stable contribution across rolling folds."""

    threshold_values = thresholds or AblationThresholds()
    if raw_ablations is None:
        return AblationMatrix(False, (), ("missing_ablation_matrix",))
    try:
        rows = _ablation_rows(raw_ablations)
    except ValueError as error:
        return AblationMatrix(False, (), (str(error),))
    results: list[AblationResult] = []
    blocked: list[str] = []
    for group in required_groups:
        canonical = _canonical_group(group)
        try:
            group_rows = tuple(
                row for row in rows if _canonical_group(_text(row, "group")) == canonical
            )
        except ValueError as error:
            return AblationMatrix(False, (), (str(error),))
        result = _evaluate_group(canonical, group_rows, threshold_values)
        results.append(result)
        if not result.passed:
            blocked.extend(result.reasons)
    return AblationMatrix(
        passed=bool(results) and not blocked,
        results=tuple(results),
        blocked_reasons=tuple(blocked),
    )


def _evaluate_group(
    group: str,
    rows: Sequence[Mapping[str, Any]],
    thresholds: AblationThresholds,
) -> AblationResult:
    if not rows:
        return AblationResult(group, False, 0.0, 0.0, 0.0, 0, (f"missing_ablation:{group}",))
    try:
        auc_values = tuple(_contribution(row, "auc") for row in rows)
        rank_values = tuple(_contribution(row, "rank_ic") for row in rows)
    except ValueError as error:
        return AblationResult(group, False, 0.0, 0.0, 0.0, 0, (str(error),))
    fold_count = len({_fold_key(row) for row in rows})
    positive_count = sum(
        1
        for auc_delta, rank_delta in zip(auc_values, rank_values, strict=True)
        if auc_delta > 0 and rank_delta > 0
    )
    positive_fraction = positive_count / len(rows)
    avg_auc = sum(auc_values) / len(auc_values)
    avg_rank = sum(rank_values) / len(rank_values)
    reasons: list[str] = []
    if fold_count < 2:
        reasons.append(f"single_or_missing_ablation_fold:{group}")
    if avg_auc < thresholds.min_auc_contribution:
        reasons.append(f"ablation_auc_contribution_below_threshold:{group}")
    if avg_rank < thresholds.min_rank_ic_contribution:
        reasons.append(f"ablation_rank_ic_contribution_below_threshold:{group}")
    if positive_fraction < thresholds.min_positive_fold_fraction:
        reasons.append(f"ablation_not_stable:{group}")
    return AblationResult(
        group=group,
        passed=not reasons,
        average_auc_contribution=avg_auc,
        average_rank_ic_contribution=avg_rank,
        positive_fold_fraction=positive_fraction,
        fold_count=fold_count,
        reasons=tuple(reasons),
    )


def _ablation_rows(
    raw_ablations: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_ablations, Mapping):
        rows = raw_ablations.get("rows", raw_ablations.get("ablations"))
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            return tuple(_require_mapping(row, "ablation row") for row in rows)
        expanded: list[Mapping[str, Any]] = []
        for group, value in raw_ablations.items():
            if group in {"rows", "ablations"}:
                continue
            if isinstance(value, Mapping):
                row = dict(value)
                row.setdefault("group", group)
                expanded.append(row)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    row = dict(_require_mapping(item, "ablation row"))
                    row.setdefault("group", group)
                    expanded.append(row)
        return tuple(expanded)
    return tuple(_require_mapping(row, "ablation row") for row in raw_ablations)


def _contribution(row: Mapping[str, Any], metric: str) -> float:
    for key in (f"{metric}_contribution", f"delta_{metric}", f"{metric}_delta"):
        if key in row:
            return _number(row, key)
    full_key = f"full_{metric}"
    ablated_key = f"ablated_{metric}"
    if full_key in row and ablated_key in row:
        return _number(row, full_key) - _number(row, ablated_key)
    raise ValueError(f"ablation row missing {metric} contribution")


def _canonical_group(group: str) -> str:
    return _GROUP_ALIASES.get(group.strip().lower().replace("-", "_"), group)


def _fold_key(row: Mapping[str, Any]) -> str:
    return str(row.get("fold", row.get("fold_id", row.get("window", ""))))


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ablation row missing {key}")
    return value


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"ablation row {key} must be numeric")
    return float(value)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
