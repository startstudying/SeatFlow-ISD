"""Frozen equal-weight seed ensemble for R2."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

R2_SEEDS = (2026, 2027, 2028, 2029, 2030)


def equal_weight_seed_scores(
    predictions_by_seed: Mapping[int, Mapping[Hashable, float]],
    *,
    required_seeds: tuple[int, ...] = R2_SEEDS,
) -> dict[Hashable, float]:
    if tuple(sorted(predictions_by_seed)) != tuple(sorted(required_seeds)):
        raise ValueError("R2 ensemble requires every frozen seed exactly once")
    first = predictions_by_seed[required_seeds[0]]
    keys = set(first)
    if any(set(predictions_by_seed[seed]) != keys for seed in required_seeds):
        raise ValueError("R2 seed predictions must cover identical keys")
    divisor = float(len(required_seeds))
    return {
        key: sum(float(predictions_by_seed[seed][key]) for seed in required_seeds) / divisor
        for key in sorted(keys, key=str)
    }
