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
) -> tuple[Any, dict[str, Any]]:
    weights = RankingLossWeights() if weights is None else weights
    residual_predictions = outputs["residual_return"].float()
    rank_predictions = outputs["rank_score"].float()
    direction_logits = outputs["direction_logit"].float()
    residual_targets = residual_targets.float()
    direction_targets = direction_targets.float()
    smooth = SmoothL1ResidualLoss()(residual_predictions, residual_targets)
    pairwise = SampledPairwiseRankLoss(max_pairs_per_date)(
        rank_predictions, residual_targets, dates
    )
    pearson = PearsonCorrelationLoss()(rank_predictions, residual_targets, dates)
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
