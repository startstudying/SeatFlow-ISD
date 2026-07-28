"""Stock temporal sequence module for R5.

This module provides:
- 20-day stock sequence tensors for temporal encoding
- Train-only scaler fitting (no cross-fold leakage)
- Sequence masking for missing days and unavailable features
- Frozen schema for feature availability tracking
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from seatflow.data.contracts import DailyBar

if TYPE_CHECKING:
    import torch


# Default reliable features (no external data required)
DEFAULT_SEQUENCE_FEATURES = [
    "overnight_return",
    "intraday_return",
    "close_return",
    "high_low_volatility",
    "volume_change",
]

# Sequence feature dimensions
SEQUENCE_LENGTH = 20
SEQUENCE_FEATURES = DEFAULT_SEQUENCE_FEATURES
SEQUENCE_FEATURE_DIM = len(SEQUENCE_FEATURES)


@dataclass(frozen=True)
class StockSequenceFeatureInfo:
    """Information about a single sequence feature."""

    name: str
    available: bool
    default_value: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class StockSequenceSchema:
    """Frozen schema for stock sequence features.

    This schema tracks which features are enabled, which are missing,
    and their availability status.
    """

    enabled_features: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_SEQUENCE_FEATURES)
    )
    missing_features: tuple[str, ...] = field(default_factory=tuple)
    feature_availability: dict[str, bool] = field(default_factory=dict)

    # Feature metadata
    feature_descriptions: dict[str, str] = field(default_factory=dict)

    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return feature_name in self.enabled_features

    def is_available(self, feature_name: str) -> bool:
        """Check if a feature has real data (not placeholder)."""
        return self.feature_availability.get(feature_name, False)

    def get_missing_count(self) -> int:
        """Number of missing/unavailable features."""
        return len(self.missing_features)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled_features": list(self.enabled_features),
            "missing_features": list(self.missing_features),
            "feature_availability": self.feature_availability,
        }


@dataclass
class StockSequenceTransformState:
    """State for stock sequence normalization.

    This state must only be fitted on training data and reused
    for validation/test transformations.
    """

    # Per-feature statistics (mean, std)
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)

    # Fitted flag
    is_fitted: bool = False

    # Cutoff date for this fit
    cutoff_date: date | None = None

    # Available time cutoff
    available_time_cutoff: date | None = None

    # Schema tracking
    schema: StockSequenceSchema = field(default_factory=StockSequenceSchema)

    def fit(
        self,
        sequences: Mapping[str, list[dict[str, float]]],
        cutoff_date: date,
        available_times: Mapping[str, date] | None = None,
        feature_availability: Mapping[str, bool] | None = None,
    ) -> None:
        """Fit transform state from stock sequences.

        Args:
            sequences: Mapping of stock_id -> list of daily feature dicts
            cutoff_date: The cutoff date (only sequences up to this date are used)
            available_times: Mapping of stock_id -> latest available time
            feature_availability: Mapping of feature_name -> True if real data exists
        """
        import numpy as np

        self.is_fitted = True
        self.cutoff_date = cutoff_date
        self.available_time_cutoff = cutoff_date

        # Build feature availability dict
        if feature_availability:
            self.schema = StockSequenceSchema(
                enabled_features=tuple(SEQUENCE_FEATURES),
                missing_features=tuple(
                    f for f in SEQUENCE_FEATURES
                    if not feature_availability.get(f, False)
                ),
                feature_availability=dict(feature_availability),
            )

        # Collect all valid values per feature (respecting cutoff dates)
        feature_values: dict[str, list[float]] = {
            feat: [] for feat in SEQUENCE_FEATURES
        }

        for stock_id, daily_features in sequences.items():
            for _, day_features in enumerate(daily_features):
                # Check available time if provided
                if available_times and stock_id in available_times:
                    if available_times[stock_id] > cutoff_date:
                        continue  # Skip post-cutoff data

                for feat in SEQUENCE_FEATURES:
                    value = day_features.get(feat)
                    # Only include if feature is actually available
                    if (value is not None
                            and not (value != value)  # Not NaN
                            and self.schema.is_enabled(feat)):
                        feature_values[feat].append(value)

        # Compute statistics only for enabled features
        for feat in SEQUENCE_FEATURES:
            if self.schema.is_enabled(feat):
                values = feature_values[feat]
                if values:
                    self.feature_means[feat] = float(np.mean(values))
                    std = float(np.std(values, ddof=1))
                    self.feature_stds[feat] = std if std > 1e-8 else 1.0
                else:
                    self.feature_means[feat] = 0.0
                    self.feature_stds[feat] = 1.0
            else:
                self.feature_means[feat] = 0.0
                self.feature_stds[feat] = 1.0

    def transform(
        self,
        sequences: Mapping[str, list[dict[str, float]]],
        stock_id_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Transform sequences using fitted statistics.

        Args:
            sequences: Mapping of stock_id -> list of daily feature dicts
            stock_id_order: Optional explicit order for stock IDs.
                           If None, uses sorted keys.

        Returns:
            dict with tensors:
            - stock_id: list of stock IDs matching output order
            - sequence: [N_stock, 20, F] float32 tensor
            - sequence_mask: [N_stock, 20] float32 tensor (1.0 = valid, 0.0 = masked)
            - feature_mask: [N_stock, 20, F] float32 tensor (1.0 = available, 0.0 = unavailable)
        """
        import numpy as np
        import torch

        if not self.is_fitted:
            raise ValueError("StockSequenceTransformState must be fitted before transform")

        # Determine stock ID order
        if stock_id_order is not None:
            stock_ids = stock_id_order
        else:
            stock_ids = sorted(sequences.keys())

        n_stock = len(stock_ids)
        n_features = len(SEQUENCE_FEATURES)

        # Initialize output tensors
        seq_tensor = np.zeros((n_stock, SEQUENCE_LENGTH, n_features), dtype=np.float32)
        mask_tensor = np.zeros((n_stock, SEQUENCE_LENGTH), dtype=np.float32)
        feature_mask_tensor = np.zeros((n_stock, SEQUENCE_LENGTH, n_features), dtype=np.float32)

        for i, stock_id in enumerate(stock_ids):
            daily_features = sequences.get(stock_id, [])
            n_days = min(len(daily_features), SEQUENCE_LENGTH)

            for day_idx in range(n_days):
                day_features = daily_features[day_idx]
                mask_tensor[i, day_idx] = 1.0  # Valid day

                for feat_idx, feat in enumerate(SEQUENCE_FEATURES):
                    if self.schema.is_enabled(feat):
                        value = day_features.get(feat)
                        if value is not None and not (value != value):  # Not NaN
                            # Mark as available
                            feature_mask_tensor[i, day_idx, feat_idx] = 1.0

                            # Normalize
                            mean = self.feature_means.get(feat, 0.0)
                            std = self.feature_stds.get(feat, 1.0)
                            seq_tensor[i, day_idx, feat_idx] = (value - mean) / std
                        # else: stays at 0.0 (masked)
                    # else: stays at 0.0 (feature not enabled)

        return {
            "stock_id": stock_ids,
            "sequence": torch.from_numpy(seq_tensor),
            "sequence_mask": torch.from_numpy(mask_tensor),
            "feature_mask": torch.from_numpy(feature_mask_tensor),
        }

    def transform_single(
        self,
        stock_id: str,
        daily_features: Sequence[dict[str, float]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform a single stock's sequence.

        Args:
            stock_id: Stock identifier (for dict key)
            daily_features: List of daily feature dicts (newest last)

        Returns:
            Tuple of (sequence, mask, feature_mask) tensors
        """
        result = self.transform({stock_id: list(daily_features)})
        return (
            result["sequence"][0],
            result["sequence_mask"][0],
            result["feature_mask"][0],
        )

    def inverse_transform_sequence(
        self,
        normalized_value: float,
        feature_name: str,
    ) -> float:
        """Inverse transform a normalized value back to original scale.

        Args:
            normalized_value: Normalized value
            feature_name: Feature name

        Returns:
            Original scale value
        """
        mean = self.feature_means.get(feature_name, 0.0)
        std = self.feature_stds.get(feature_name, 1.0)
        return normalized_value * std + mean

    def to_dict(self) -> dict[str, object]:
        """Serialize transform state for checkpoint."""
        return {
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "is_fitted": self.is_fitted,
            "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
            "schema": self.schema.to_dict(),
        }

    def fingerprint(self) -> str:
        """Generate hash of transform state for checkpoint binding."""
        payload = {
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def validate(self) -> None:
        """Validate transform state is properly fitted."""
        if not self.is_fitted:
            raise ValueError("StockSequenceTransformState is not fitted")
        if self.cutoff_date is None:
            raise ValueError("StockSequenceTransformState cutoff_date is not set")


def extract_stock_sequences(
    bars_by_stock: Mapping[str, Sequence[DailyBar]],
    as_of_date: date,
    market_index_bars: Mapping[str, Sequence[DailyBar]] | None = None,
    industry_bars_by_stock: Mapping[str, Sequence[DailyBar]] | None = None,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_availability: Mapping[str, bool] | None = None,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, bool]]:
    """Extract 20-day sequence features for each stock.

    Args:
        bars_by_stock: DailyBar instances grouped by stock_id
        as_of_date: The cutoff date for features
        market_index_bars: Market index bars (unused in v1 - real data required)
        industry_bars_by_stock: Industry bars (unused in v1 - real data required)
        sequence_length: Number of days in sequence (default 20)
        feature_availability: Dict of feature -> True if real data exists

    Returns:
        Tuple of (sequences_dict, feature_availability)
    """
    sequences: dict[str, list[dict[str, float]]] = {}
    detected_availability: dict[str, bool] = {
        feat: False for feat in SEQUENCE_FEATURES
    }

    for stock_id, bars in bars_by_stock.items():
        # Sort by date
        sorted_bars = sorted(bars, key=lambda b: b.trade_date)

        # Filter to date range
        valid_bars = [b for b in sorted_bars if b.trade_date <= as_of_date]
        if len(valid_bars) < 2:
            continue  # Need at least 2 days for returns

        # Take last N days
        recent_bars = valid_bars[-sequence_length:]

        # Compute daily features
        stock_sequences: list[dict[str, float]] = []
        closes = [float(b.close_price) for b in recent_bars]
        volumes = [float(b.volume) for b in recent_bars]
        highs = [float(b.high_price) for b in recent_bars]
        lows = [float(b.low_price) for b in recent_bars]
        opens = [float(b.open_price) for b in recent_bars]

        for i, _bar in enumerate(recent_bars):
            features: dict[str, float] = {}

            # Overnight return: previous close to today's open (available from second day)
            if i > 0 and closes[i - 1] > 0:
                features["overnight_return"] = (opens[i] / closes[i - 1]) - 1.0
                detected_availability["overnight_return"] = True
            else:
                features["overnight_return"] = 0.0

            # Intraday return: today's open to today's close
            if opens[i] > 0:
                features["intraday_return"] = (closes[i] / opens[i]) - 1.0
                detected_availability["intraday_return"] = True
            else:
                features["intraday_return"] = 0.0

            # Close return: today's close vs previous close (available from second day)
            if i > 0 and closes[i - 1] > 0:
                features["close_return"] = (closes[i] / closes[i - 1]) - 1.0
                detected_availability["close_return"] = True
            else:
                features["close_return"] = 0.0

            # High-low volatility: daily range normalized by close (always available)
            if closes[i] > 0:
                features["high_low_volatility"] = (highs[i] - lows[i]) / closes[i]
                detected_availability["high_low_volatility"] = True
            else:
                features["high_low_volatility"] = 0.0

            # Volume change: volume ratio vs previous day (available from second day)
            if i > 0 and volumes[i - 1] > 0:
                features["volume_change"] = (volumes[i] / volumes[i - 1]) - 1.0
                detected_availability["volume_change"] = True
            else:
                features["volume_change"] = 0.0

            # NOTE: turnover_rate, market_excess_return, industry_excess_return
            # are NOT included in DEFAULT_SEQUENCE_FEATURES.
            # They require external data sources and should not be placeholders.

            stock_sequences.append(features)

        if stock_sequences:
            sequences[stock_id] = stock_sequences

    # Merge with provided availability
    if feature_availability:
        detected_availability.update(feature_availability)

    return sequences, detected_availability


def attach_sequences_to_graph(
    graph_data: Any,
    stock_sequences: torch.Tensor,
    stock_sequence_mask: torch.Tensor,
    stock_feature_mask: torch.Tensor,
    stock_id_order: list[str],
) -> Any:
    """Attach sequence tensors to a PyG graph data object.

    Args:
        graph_data: PyG HeteroData object
        stock_sequences: [N_stock, 20, F] tensor
        stock_sequence_mask: [N_stock, 20] tensor
        stock_feature_mask: [N_stock, 20, F] tensor
        stock_id_order: List of stock IDs matching tensor order.
                       Must match graph node mapping order.

    Returns:
        Modified graph_data with sequence tensors attached
    """
    # Verify order matches node mapping
    if hasattr(graph_data, "metadata"):
        node_mapping = graph_data.metadata()
        if "node_mapping" in node_mapping and "stock" in node_mapping["node_mapping"]:
            graph_stock_ids = node_mapping["node_mapping"]["stock"]
            if graph_stock_ids != stock_id_order:
                raise ValueError(
                    f"Stock ID order mismatch. Graph has {len(graph_stock_ids)} stocks, "
                    f"sequences have {len(stock_id_order)} stocks. "
                    f"First mismatch: graph={graph_stock_ids[:5]}, "
                    f"sequences={stock_id_order[:5]}"
                )

    graph_data["stock"].sequence = stock_sequences
    graph_data["stock"].sequence_mask = stock_sequence_mask
    graph_data["stock"].feature_mask = stock_feature_mask
    graph_data["stock"].sequence_stock_ids = stock_id_order

    return graph_data
