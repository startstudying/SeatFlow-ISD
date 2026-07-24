from __future__ import annotations

from dataclasses import dataclass

from seatflow.graph.types import HeteroGraphSnapshot


@dataclass(frozen=True)
class ModelInput:
    graph: HeteroGraphSnapshot
    target_stock_id: str


@dataclass(frozen=True)
class PredictionResult:
    target_stock_id: str
    rise_probability: float
    expected_returns: dict[int, float]
    key_seats: tuple[str, ...]
    propagation_paths: tuple[tuple[str, ...], ...]
    risk_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.rise_probability <= 1:
            raise ValueError("rise_probability must be in [0, 1]")
