from __future__ import annotations

from math import isfinite, log


def binary_cross_entropy(probabilities: tuple[float, ...], labels: tuple[int, ...]) -> float:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("at least one prediction is required")

    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        if label not in (0, 1):
            raise ValueError("labels must be binary integers")
        if not isfinite(probability) or not 0 < probability < 1:
            raise ValueError("probabilities must be finite values strictly inside (0, 1)")
        total += -(label * log(probability) + (1 - label) * log(1 - probability))
    return round(total / len(probabilities), 6)
