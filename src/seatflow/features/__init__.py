"""Feature engineering for SeatFlow experiments.

This package provides configurable feature engineering supporting E0-E4 experiments:
- E0: Stock temporal + static features (baseline)
- E1: E0 + industry/stock relations
- E2: E1 + seat aggregate features
- E3: E1 + seat residual graph
- E4: E3 + SeatStyleEncoder

Feature modules:
- stock_temporal: Stock time-series features (returns, volatility, momentum)
- seat_aggregates: DragonTiger seat aggregation features
- market_env: Market-wide environment features (CSI300, market median)
- pipeline: Configurable feature pipeline for different experiments
"""

from seatflow.features.basic import compute_basic_features
from seatflow.features.market_env import (
    MARKET_ENV_FEATURES,
    MarketEnvFeatureComputer,
    compute_market_env_features,
)
from seatflow.features.pipeline import (
    ExperimentLevel,
    FeaturePipeline,
    build_feature_pipeline,
    get_features_for_experiment,
)
from seatflow.features.seat_aggregates import (
    SEAT_AGGREGATE_FEATURES,
    SeatAggregateFeatureComputer,
    compute_seat_aggregate_features,
)
from seatflow.features.stock_temporal import (
    STOCK_TEMPORAL_FEATURES,
    StockTemporalFeatureComputer,
    compute_stock_temporal_features,
)

__all__ = [
    # Legacy basic features
    "compute_basic_features",
    # Stock temporal features
    "STOCK_TEMPORAL_FEATURES",
    "StockTemporalFeatureComputer",
    "compute_stock_temporal_features",
    # Seat aggregate features
    "SEAT_AGGREGATE_FEATURES",
    "SeatAggregateFeatureComputer",
    "compute_seat_aggregate_features",
    # Market environment features
    "MARKET_ENV_FEATURES",
    "MarketEnvFeatureComputer",
    "compute_market_env_features",
    # Feature pipeline
    "ExperimentLevel",
    "FeaturePipeline",
    "build_feature_pipeline",
    "get_features_for_experiment",
]
