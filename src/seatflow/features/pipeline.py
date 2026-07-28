"""Configurable feature pipeline for E0-E4 experiments.

This module provides a unified interface for feature computation across different
experiment configurations, enabling consistent feature engineering for:
- E0: Stock temporal + market environment (baseline)
- E1: E0 + industry/stock relations (handled at graph level)
- E2: E1 + seat aggregate features
- E3: E1 + seat residual graph (handled at graph level)
- E4: E3 + SeatStyleEncoder (handled at model level)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import IntEnum
from typing import Any

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord
from seatflow.features.market_env import (
    MARKET_E0,
    MARKET_ENV_FEATURES,
    MarketEnvFeatureComputer,
)
from seatflow.features.seat_aggregates import (
    SEAT_AGGREGATE_FEATURES,
    SEAT_E2,
    SeatAggregateFeatureComputer,
)
from seatflow.features.stock_temporal import (
    STOCK_TEMPORAL_E0,
    STOCK_TEMPORAL_FEATURES,
    StockTemporalFeatureComputer,
)


class ExperimentLevel(IntEnum):
    """Experiment configuration levels.

    E0: Stock temporal + static features (baseline)
    E1: E0 + industry/stock relations (graph-level)
    E2: E1 + seat aggregate features
    E3: E1 + seat residual graph (graph-level)
    E4: E3 + SeatStyleEncoder (model-level)
    """

    E0 = 0  # Stock temporal + market env
    E1 = 1  # E0 + industry/stock relations (graph)
    E2 = 2  # E1 + seat aggregate features
    E3 = 3  # E1 + seat residual graph (graph)
    E4 = 4  # E3 + SeatStyleEncoder (model)


# Feature set by experiment level
FEATURE_CONFIG: dict[ExperimentLevel, dict[str, tuple[str, ...]]] = {
    ExperimentLevel.E0: {
        "stock_temporal": STOCK_TEMPORAL_E0,
        "market_env": MARKET_E0,
        "seat_aggregate": (),  # No seat features in E0
    },
    ExperimentLevel.E1: {
        "stock_temporal": STOCK_TEMPORAL_E0,
        "market_env": MARKET_E0,
        "seat_aggregate": (),  # Seat features still not included
    },
    ExperimentLevel.E2: {
        "stock_temporal": STOCK_TEMPORAL_E0,
        "market_env": MARKET_E0,
        "seat_aggregate": SEAT_E2,
    },
    ExperimentLevel.E3: {
        "stock_temporal": STOCK_TEMPORAL_E0,
        "market_env": MARKET_E0,
        "seat_aggregate": (),  # E3 uses seat graph, not aggregates
    },
    ExperimentLevel.E4: {
        "stock_temporal": STOCK_TEMPORAL_E0,
        "market_env": MARKET_E0,
        "seat_aggregate": (),  # E4 uses SeatStyleEncoder
    },
}


@dataclass(frozen=True)
class FeatureBundle:
    """Bundled features for a single stock-date."""

    trade_date: str
    stock_id: str
    stock_temporal: dict[str, float]
    market_env: dict[str, float]
    seat_aggregate: dict[str, float]
    has_seat_data: bool

    def all_features(self) -> dict[str, float]:
        """Merge all features into a single dict."""
        features: dict[str, float] = {}
        features.update(self.stock_temporal)
        features.update(self.market_env)
        features.update(self.seat_aggregate)
        return features

    def model_features(self, include_missing_marker: bool = True) -> dict[str, float]:
        """Return features ready for model input."""
        features = self.all_features()
        if include_missing_marker:
            features["missing_seat_event"] = 0.0 if self.has_seat_data else 1.0
        return features


@dataclass
class FeaturePipeline:
    """Configurable feature pipeline for experiment levels.

    This pipeline computes all features for a given experiment configuration,
    handling the coordination between stock temporal, market environment, and
    seat aggregate feature computers.
    """

    experiment_level: ExperimentLevel
    stock_temporal_computer: StockTemporalFeatureComputer
    market_env_computer: MarketEnvFeatureComputer
    seat_aggregate_computer: SeatAggregateFeatureComputer
    _feature_cache: dict[tuple[str, str], FeatureBundle] = field(default_factory=dict)

    @classmethod
    def from_experiment_level(
        cls,
        level: ExperimentLevel,
        min_history_days: int = 20,
        include_missing_marker: bool = True,
    ) -> FeaturePipeline:
        """Create a feature pipeline for a specific experiment level.

        Args:
            level: Experiment level (E0-E4)
            min_history_days: Minimum history required for temporal features
            include_missing_marker: Whether to include missing seat event marker

        Returns:
            Configured FeaturePipeline
        """
        config = FEATURE_CONFIG[level]

        stock_temporal = StockTemporalFeatureComputer(
            features=config["stock_temporal"],
            min_history_days=min_history_days,
        )

        market_env = MarketEnvFeatureComputer(
            features=config["market_env"],
        )

        seat_aggregate = SeatAggregateFeatureComputer(
            features=config["seat_aggregate"] if config["seat_aggregate"] else None,
            include_missing_marker=include_missing_marker,
        )

        return cls(
            experiment_level=level,
            stock_temporal_computer=stock_temporal,
            market_env_computer=market_env,
            seat_aggregate_computer=seat_aggregate,
        )

    def compute_all_features(
        self,
        bars: Sequence[DailyBar],
        records: Sequence[DragonTigerRecord],
        as_of_date: date,
        stock_ids: Sequence[str] | None = None,
        cutoff_time: datetime | None = None,
    ) -> tuple[FeatureBundle, ...]:
        """Compute all features for the given experiment level.

        Args:
            bars: Daily bar data
            records: DragonTiger seat records
            as_of_date: Date to compute features as of
            stock_ids: Optional stock IDs to compute for
            cutoff_time: Optional cutoff time for PIT contract

        Returns:
            Tuple of FeatureBundle for each stock-date
        """
        self._feature_cache.clear()

        # Normalize stock IDs
        target_stocks = {normalize_stock_id(sid) for sid in (stock_ids or [])}

        # Compute stock temporal features
        temporal_rows = self.stock_temporal_computer.compute(bars, as_of_date=as_of_date)
        temporal_by_stock: dict[str, dict[str, float]] = defaultdict(dict)
        for row in temporal_rows:
            if not target_stocks or row.stock_id in target_stocks:
                temporal_by_stock[row.stock_id] = row.features

        # Compute market environment features (stock-level)
        market_rows = self.market_env_computer.compute_stock_market_features(
            bars, as_of_date=as_of_date
        )
        market_by_stock: dict[str, dict[str, float]] = defaultdict(dict)
        for market_row in market_rows:
            if not target_stocks or market_row.stock_id in target_stocks:
                market_by_stock[market_row.stock_id] = market_row.features

        # Compute seat aggregate features
        seat_rows = self.seat_aggregate_computer.compute(
            records,
            as_of_date,
            stock_ids=list(target_stocks) if target_stocks else None,
            cutoff_time=cutoff_time,
            historical_records=records,
        )
        seat_by_stock: dict[str, tuple[dict[str, float], bool]] = {}
        for seat_row in seat_rows:
            seat_by_stock[seat_row.stock_id] = (
                seat_row.features,
                not seat_row.missing_seat_event,
            )

        # Build feature bundles
        bundles: list[FeatureBundle] = []
        all_stocks = target_stocks or set(temporal_by_stock.keys())

        for stock_id in sorted(all_stocks):
            temporal = temporal_by_stock.get(stock_id, {})
            market = market_by_stock.get(stock_id, {})
            seat_data = seat_by_stock.get(stock_id, ({}, False))
            seat_feat, has_seat = seat_data

            if temporal or market or seat_feat:  # At least some features exist
                bundles.append(
                    FeatureBundle(
                        trade_date=as_of_date.isoformat(),
                        stock_id=stock_id,
                        stock_temporal=temporal,
                        market_env=market,
                        seat_aggregate=seat_feat,
                        has_seat_data=has_seat,
                    )
                )

        return tuple(bundles)

    def get_feature_order(self) -> tuple[str, ...]:
        """Return ordered feature names for this experiment level."""
        order: list[str] = []
        order.extend(self.stock_temporal_computer.feature_order)
        order.extend(self.market_env_computer.feature_order)

        seat_features = self.seat_aggregate_computer.feature_order
        for feat in seat_features:
            if feat not in order:
                order.append(feat)

        # Add missing marker if seat aggregates are configured
        if self.experiment_level >= ExperimentLevel.E2:
            if "missing_seat_event" not in order:
                order.append("missing_seat_event")

        return tuple(order)

    def get_feature_count(self) -> int:
        """Return the number of features for this experiment level."""
        return len(self.get_feature_order())


def build_feature_pipeline(
    experiment: str | ExperimentLevel,
    min_history_days: int = 20,
) -> FeaturePipeline:
    """Build a feature pipeline from experiment identifier.

    Args:
        experiment: Experiment name (e.g., "E0", "E1") or ExperimentLevel
        min_history_days: Minimum history for temporal features

    Returns:
        Configured FeaturePipeline
    """
    if isinstance(experiment, str):
        level = ExperimentLevel[experiment.upper()]
    else:
        level = experiment

    return FeaturePipeline.from_experiment_level(
        level,
        min_history_days=min_history_days,
    )


def get_features_for_experiment(
    experiment: str | ExperimentLevel,
) -> dict[str, tuple[str, ...]]:
    """Get feature names for a specific experiment.

    Args:
        experiment: Experiment name or level

    Returns:
        Dict mapping feature category to feature names
    """
    if isinstance(experiment, str):
        level = ExperimentLevel[experiment.upper()]
    else:
        level = experiment

    return dict(FEATURE_CONFIG[level])


def get_all_supported_features() -> dict[str, tuple[str, ...]]:
    """Get all supported features across all experiments.

    Returns:
        Dict mapping feature category to all supported feature names
    """
    return {
        "stock_temporal": STOCK_TEMPORAL_FEATURES,
        "market_env": MARKET_ENV_FEATURES,
        "seat_aggregate": SEAT_AGGREGATE_FEATURES,
    }


def normalize_stock_id(stock_id: str) -> str:
    """Normalize stock ID to canonical form."""
    from seatflow.data.contracts import normalize_stock_id as _normalize

    return _normalize(stock_id)


@dataclass
class ExperimentFeatureSummary:
    """Summary of features for an experiment."""

    experiment_level: ExperimentLevel
    stock_temporal_features: tuple[str, ...]
    market_env_features: tuple[str, ...]
    seat_aggregate_features: tuple[str, ...]
    total_feature_count: int
    includes_seat_aggregates: bool

    @classmethod
    def from_pipeline(cls, pipeline: FeaturePipeline) -> ExperimentFeatureSummary:
        """Create summary from a feature pipeline."""
        config = FEATURE_CONFIG[pipeline.experiment_level]
        seat_features = config["seat_aggregate"]
        return cls(
            experiment_level=pipeline.experiment_level,
            stock_temporal_features=config["stock_temporal"],
            market_env_features=config["market_env"],
            seat_aggregate_features=seat_features,
            total_feature_count=pipeline.get_feature_count(),
            includes_seat_aggregates=bool(seat_features),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "experiment_level": self.experiment_level.name,
            "stock_temporal_features": list(self.stock_temporal_features),
            "market_env_features": list(self.market_env_features),
            "seat_aggregate_features": list(self.seat_aggregate_features),
            "total_feature_count": self.total_feature_count,
            "includes_seat_aggregates": self.includes_seat_aggregates,
        }
