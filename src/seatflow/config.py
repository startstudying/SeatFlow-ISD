from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    lookback_days: int
    provider: str
    available_after_close: time


@dataclass(frozen=True)
class GraphConfig:
    decay_lambda: float
    max_edge_age_days: int
    prediction_horizons: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    seat_profile_types: int


@dataclass(frozen=True)
class TrainingConfig:
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    data: DataConfig
    graph: GraphConfig
    model: ModelConfig
    training: TrainingConfig


def _positive_int(value: Any, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _parse_time(value: Any, field: str) -> time:
    try:
        return time.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{field} must use HH:MM or HH:MM:SS") from error


def _parse_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD") from error


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    project = raw["project"]
    data = raw["data"]
    graph = raw["graph"]
    model = raw["model"]
    training = raw["training"]

    decay_lambda = float(graph["decay_lambda"])
    if decay_lambda < 0:
        raise ValueError("graph.decay_lambda must be non-negative")

    horizons = tuple(
        _positive_int(item, "graph.prediction_horizons") for item in graph["prediction_horizons"]
    )
    if not horizons:
        raise ValueError("graph.prediction_horizons must not be empty")

    dropout = float(model["dropout"])
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")

    raw_dir = Path(data["raw_dir"])
    processed_dir = Path(data["processed_dir"])
    if raw_dir.parent != processed_dir.parent:
        raise ValueError("data.raw_dir and data.processed_dir must share the same parent")

    return ProjectConfig(
        name=str(project["name"]),
        seed=int(project["seed"]),
        data=DataConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            lookback_days=_positive_int(data["lookback_days"], "data.lookback_days"),
            provider=str(data["provider"]),
            available_after_close=_parse_time(
                data["available_after_close"], "data.available_after_close"
            ),
        ),
        graph=GraphConfig(
            decay_lambda=decay_lambda,
            max_edge_age_days=_positive_int(graph["max_edge_age_days"], "graph.max_edge_age_days"),
            prediction_horizons=horizons,
        ),
        model=ModelConfig(
            hidden_dim=_positive_int(model["hidden_dim"], "model.hidden_dim"),
            num_heads=_positive_int(model["num_heads"], "model.num_heads"),
            num_layers=_positive_int(model["num_layers"], "model.num_layers"),
            dropout=dropout,
            seat_profile_types=_positive_int(
                model["seat_profile_types"], "model.seat_profile_types"
            ),
        ),
        training=TrainingConfig(
            train_start=_parse_date(training["train_start"], "training.train_start"),
            train_end=_parse_date(training["train_end"], "training.train_end"),
            valid_start=_parse_date(training["valid_start"], "training.valid_start"),
            valid_end=_parse_date(training["valid_end"], "training.valid_end"),
            test_start=_parse_date(training["test_start"], "training.test_start"),
            test_end=_parse_date(training["test_end"], "training.test_end"),
        ),
    )
