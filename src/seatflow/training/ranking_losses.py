"""Date-local ranking losses for SeatRankLite."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any


class SmoothL1ResidualLoss:
    def __init__(self, beta: float = 1.0) -> None:
        if beta <= 0:
            raise ValueError("SmoothL1 beta must be positive")
        self.beta = beta

    def __call__(self, predictions: Any, targets: Any) -> Any:
        import torch.nn.functional as functional

        return functional.smooth_l1_loss(predictions, targets, beta=self.beta)


class DirectionBCELoss:
    def __call__(self, logits: Any, targets: Any) -> Any:
        import torch.nn.functional as functional

        return functional.binary_cross_entropy_with_logits(logits, targets)


class SampledPairwiseRankLoss:
    def __init__(self, max_pairs_per_date: int = 1024, seed: int = 0) -> None:
        if max_pairs_per_date <= 0:
            raise ValueError("max_pairs_per_date must be positive")
        self.max_pairs_per_date = max_pairs_per_date
        self.seed = seed
        self.last_pair_count_by_date: dict[str, int] = {}

    def __call__(
        self,
        predictions: Any,
        targets: Any,
        dates: Sequence[Hashable],
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
            raise ValueError("pairwise predictions and targets must be aligned 1D tensors")
        if len(dates) != int(predictions.numel()):
            raise ValueError("pairwise dates must align with predictions")
        losses: list[Any] = []
        self.last_pair_count_by_date = {}
        for date_value, indices in _group_indices(dates).items():
            pairs = _sample_pairs(
                len(indices),
                self.max_pairs_per_date,
                seed=self.seed + _stable_date_seed(date_value),
            )
            if not pairs:
                self.last_pair_count_by_date[str(date_value)] = 0
                continue
            left = torch.tensor(
                [indices[pair[0]] for pair in pairs],
                dtype=torch.long,
                device=predictions.device,
            )
            right = torch.tensor(
                [indices[pair[1]] for pair in pairs],
                dtype=torch.long,
                device=predictions.device,
            )
            target_delta = targets.index_select(0, left) - targets.index_select(0, right)
            non_ties = target_delta != 0
            if not bool(non_ties.any()):
                self.last_pair_count_by_date[str(date_value)] = 0
                continue
            prediction_delta = (
                predictions.index_select(0, left) - predictions.index_select(0, right)
            )[non_ties]
            signs = torch.sign(target_delta[non_ties])
            losses.append(functional.softplus(-signs * prediction_delta).mean())
            self.last_pair_count_by_date[str(date_value)] = int(non_ties.sum().item())
        return torch.stack(losses).mean() if losses else predictions.sum() * 0.0


class LambdaRankAtKLoss:
    """Sparse LambdaRank loss focused on true leaders and the predicted K boundary."""

    def __init__(
        self,
        k: int = 5,
        max_pairs_per_date: int = 1024,
        true_top_fraction: float = 0.20,
        seed: int = 0,
    ) -> None:
        if k <= 0 or max_pairs_per_date <= 0:
            raise ValueError("LambdaRank k and pair limit must be positive")
        if not 0 < true_top_fraction < 1:
            raise ValueError("true_top_fraction must be between zero and one")
        self.k = k
        self.max_pairs_per_date = max_pairs_per_date
        self.true_top_fraction = true_top_fraction
        self.seed = seed
        self.last_pair_count_by_date: dict[str, int] = {}

    def __call__(
        self,
        predictions: Any,
        ranking_targets: Any,
        dates: Sequence[Hashable],
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if (
            predictions.ndim != 1
            or ranking_targets.ndim != 1
            or predictions.shape != ranking_targets.shape
        ):
            raise ValueError("LambdaRank predictions and targets must be aligned 1D tensors")
        if len(dates) != int(predictions.numel()):
            raise ValueError("LambdaRank dates must align with predictions")
        losses: list[Any] = []
        self.last_pair_count_by_date = {}
        for date_value, indices in _group_indices(dates).items():
            index = torch.tensor(indices, dtype=torch.long, device=predictions.device)
            scores = predictions.index_select(0, index)
            targets = ranking_targets.index_select(0, index)
            pairs = _sparse_lambda_pairs(
                scores,
                targets,
                k=self.k,
                maximum=self.max_pairs_per_date,
                true_top_fraction=self.true_top_fraction,
                seed=self.seed + _stable_date_seed(date_value),
            )
            self.last_pair_count_by_date[str(date_value)] = len(pairs)
            if not pairs:
                continue
            left = torch.tensor([pair[0] for pair in pairs], device=scores.device)
            right = torch.tensor([pair[1] for pair in pairs], device=scores.device)
            target_delta = targets.index_select(0, left) - targets.index_select(0, right)
            signs = torch.sign(target_delta)
            prediction_delta = scores.index_select(0, left) - scores.index_select(0, right)
            weights = _delta_ndcg_weights(scores, targets, left, right, self.k)
            losses.append((functional.softplus(-signs * prediction_delta) * weights).mean())
        return torch.stack(losses).mean() if losses else predictions.sum() * 0.0


class PearsonCorrelationLoss:
    def __call__(
        self,
        predictions: Any,
        targets: Any,
        dates: Sequence[Hashable],
    ) -> Any:
        import torch

        if predictions.ndim != 1 or targets.shape != predictions.shape:
            raise ValueError("Pearson predictions and targets must be aligned 1D tensors")
        if len(dates) != int(predictions.numel()):
            raise ValueError("Pearson dates must align with predictions")
        losses: list[Any] = []
        for indices in _group_indices(dates).values():
            if len(indices) < 2:
                continue
            index = torch.tensor(indices, dtype=torch.long, device=predictions.device)
            predicted = predictions.index_select(0, index)
            actual = targets.index_select(0, index)
            predicted = predicted - predicted.mean()
            actual = actual - actual.mean()
            denominator = predicted.square().sum().sqrt() * actual.square().sum().sqrt()
            if bool(denominator > 0):
                losses.append(1.0 - (predicted * actual).sum() / denominator.clamp_min(1e-8))
        return torch.stack(losses).mean() if losses else predictions.sum() * 0.0


@dataclass(frozen=True)
class RankingLossWeights:
    smooth_l1: float = 0.45
    pairwise_rank: float = 0.30
    pearson: float = 0.15
    direction: float = 0.10

    def __post_init__(self) -> None:
        values = (self.smooth_l1, self.pairwise_rank, self.pearson, self.direction)
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("ranking loss weights must be non-negative and sum to one")


def compute_ranking_loss(
    outputs: dict[str, Any],
    residual_targets: Any,
    direction_targets: Any,
    dates: Sequence[Hashable],
    *,
    weights: RankingLossWeights | None = None,
    max_pairs_per_date: int = 1024,
    objective: str = "L0",
    ranking_targets: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    objective = objective.upper()
    if objective not in {"L0", "L1", "L2"}:
        raise ValueError("ranking objective must be L0, L1, or L2")
    if weights is None:
        weights = {
            "L0": RankingLossWeights(0.45, 0.30, 0.15, 0.10),
            "L1": RankingLossWeights(0.35, 0.40, 0.15, 0.10),
            "L2": RankingLossWeights(0.30, 0.45, 0.15, 0.10),
        }[objective]
    residual_predictions = outputs["residual_return"].float()
    rank_predictions = outputs["rank_score"].float()
    direction_logits = outputs["direction_logit"].float()
    residual_targets = residual_targets.float()
    direction_targets = direction_targets.float()
    smooth = SmoothL1ResidualLoss()(residual_predictions, residual_targets)
    rank_targets = residual_targets if ranking_targets is None else ranking_targets.float()
    pairwise = (
        LambdaRankAtKLoss(k=5, max_pairs_per_date=max_pairs_per_date)(
            rank_predictions, rank_targets, dates
        )
        if objective == "L2"
        else SampledPairwiseRankLoss(max_pairs_per_date)(
            rank_predictions, rank_targets, dates
        )
    )
    pearson = PearsonCorrelationLoss()(rank_predictions, rank_targets, dates)
    direction = DirectionBCELoss()(direction_logits, direction_targets)
    total = (
        weights.smooth_l1 * smooth
        + weights.pairwise_rank * pairwise
        + weights.pearson * pearson
        + weights.direction * direction
    )
    return total, {
        "smooth_l1": smooth,
        "pairwise_rank": pairwise,
        "pearson": pearson,
        "direction": direction,
    }


def _group_indices(dates: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, value in enumerate(dates):
        grouped[value].append(index)
    return grouped


def _sample_pairs(size: int, maximum: int, *, seed: int) -> tuple[tuple[int, int], ...]:
    if size < 2:
        return ()
    total = size * (size - 1) // 2
    if total <= maximum:
        return tuple(combinations(range(size), 2))
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected: set[tuple[int, int]] = set()
    while len(selected) < maximum:
        draws = torch.randint(
            0,
            size,
            (max(32, (maximum - len(selected)) * 3), 2),
            generator=generator,
        ).tolist()
        for left, right in draws:
            if left == right:
                continue
            selected.add((min(left, right), max(left, right)))
            if len(selected) == maximum:
                break
    return tuple(sorted(selected))


def _stable_date_seed(value: Hashable) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(str(value)))


def _sparse_lambda_pairs(
    scores: Any,
    targets: Any,
    *,
    k: int,
    maximum: int,
    true_top_fraction: float,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    import torch

    size = int(scores.numel())
    if size < 2:
        return ()
    true_top_count = max(1, int((size * true_top_fraction) + 0.999999))
    true_order = targets.detach().argsort(descending=True).cpu().tolist()
    predicted_order = scores.detach().argsort(descending=True).cpu().tolist()
    target_values = targets.detach().cpu().tolist()
    pairs: set[tuple[int, int]] = set()
    boundary_left = predicted_order[: min(k, size)]
    boundary_right = predicted_order[min(k, size) : min(2 * k, size)]
    for left in boundary_left:
        for right in boundary_right:
            if target_values[left] != target_values[right]:
                pairs.add((left, right))
    leaders, rest = true_order[:true_top_count], true_order[true_top_count:]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    attempts = 0
    while rest and len(pairs) < maximum and attempts < maximum * 20:
        left = leaders[int(torch.randint(len(leaders), (), generator=generator))]
        right = rest[int(torch.randint(len(rest), (), generator=generator))]
        if target_values[left] != target_values[right]:
            pairs.add((left, right))
        attempts += 1
    return tuple(sorted(pairs))[:maximum]


def _delta_ndcg_weights(
    scores: Any,
    targets: Any,
    left: Any,
    right: Any,
    k: int,
) -> Any:
    import torch

    positions = torch.empty_like(scores.argsort())
    order = scores.detach().argsort(descending=True)
    positions[order] = torch.arange(scores.numel(), device=scores.device)
    left_positions = positions.index_select(0, left)
    right_positions = positions.index_select(0, right)
    left_discount = torch.where(
        left_positions < k,
        1.0 / torch.log2(left_positions.float() + 2.0),
        torch.zeros_like(left_positions, dtype=torch.float32),
    )
    right_discount = torch.where(
        right_positions < k,
        1.0 / torch.log2(right_positions.float() + 2.0),
        torch.zeros_like(right_positions, dtype=torch.float32),
    )
    gains = torch.pow(2.0, targets.detach().float()) - 1.0
    gain_delta = (
        gains.index_select(0, left) - gains.index_select(0, right)
    ).abs()
    ideal = targets.detach().argsort(descending=True)[:k]
    ideal_discounts = 1.0 / torch.log2(
        torch.arange(ideal.numel(), device=scores.device, dtype=torch.float32) + 2.0
    )
    ideal_gains = (
        torch.pow(2.0, targets.detach().float().index_select(0, ideal)) - 1.0
    )
    ideal_dcg = (ideal_gains * ideal_discounts).sum()
    normalized = (
        gain_delta
        * (left_discount - right_discount).abs()
        / ideal_dcg.clamp_min(1e-8)
    )
    return normalized.detach()
