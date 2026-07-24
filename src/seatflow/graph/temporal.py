from __future__ import annotations

from math import exp


def time_decay(base_weight: float, delta_days: int, decay_lambda: float) -> float:
    if base_weight < 0:
        raise ValueError("base_weight must be non-negative")
    if delta_days < 0:
        raise ValueError("delta_days must be non-negative")
    if decay_lambda < 0:
        raise ValueError("decay_lambda must be non-negative")
    return base_weight * exp(-decay_lambda * delta_days)
