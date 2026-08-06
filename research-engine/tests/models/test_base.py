"""Tests for the formal Base daily-path contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from seatflow.features.registry import FeatureScope, FeatureSpec, SourceKind
from seatflow.models.base import (
    BASE_CUMULATIVE_HORIZONS,
    BaseFeatureContract,
    BaseLoss,
    BaseMarketModel,
    BasePrediction,
    cumulative_from_daily_path,
)


def _stock_spec(name: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        version="1.0",
        horizons=[5],
        dependencies=[],
        description=f"{name} fixture",
        scope=FeatureScope.STOCK_PIT,
        source=SourceKind.STOCK_PIT,
    )


def _market_spec(name: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        version="1.0",
        horizons=[5],
        dependencies=[],
        description=f"{name} fixture",
        scope=FeatureScope.MARKET_PIT,
        source=SourceKind.MARKET_PIT,
    )


def _base_specs() -> tuple[FeatureSpec, ...]:
    return (
        _stock_spec("stock_volatility_5d"),
        _market_spec("market_excess_return_1d"),
        _market_spec("market_excess_return_5d"),
    )


class TestBaseDailyPath:
    def test_forward_returns_exactly_five_daily_increments(self) -> None:
        model = BaseMarketModel(input_dim=3, hidden_dim=8, dropout=0.0)
        out = model(torch.randn(4, 3))
        assert out.shape == (4, 5)
        assert model.output_dim == 5

    def test_rejects_independent_cumulative_horizon_head_configuration(self) -> None:
        with pytest.raises(ValueError, match="five daily"):
            BaseMarketModel(input_dim=3, num_horizons=3)

    def test_cumulative_horizons_are_derived_from_one_daily_path(self) -> None:
        daily = torch.tensor([[0.01, -0.02, 0.03, 0.04, -0.05]])
        cumulative = cumulative_from_daily_path(daily)
        assert BASE_CUMULATIVE_HORIZONS == (1, 3, 5)
        assert torch.allclose(cumulative, torch.tensor([[0.01, 0.02, 0.01]]))

    @pytest.mark.parametrize(
        "daily",
        [
            torch.ones(5),
            torch.ones(2, 3),
            torch.tensor([[0.0, 1.0, float("nan"), 3.0, 4.0]]),
        ],
    )
    def test_cumulative_derivation_validates_shape_and_finiteness(self, daily: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="daily path"):
            cumulative_from_daily_path(daily)


class TestBaseFeatureContract:
    def test_accepts_only_registered_stock_and_market_pit_features(self) -> None:
        contract = BaseFeatureContract(_base_specs())
        x = torch.tensor([[0.2, 0.01, -0.03], [0.4, 0.02, 0.01]])
        validated = contract.validate_tensor(x, ["stock-a", "stock-b"])
        assert torch.equal(validated, x)

    @pytest.mark.parametrize(
        ("scope", "source", "current_event", "message"),
        [
            (FeatureScope.SEAT, SourceKind.SEAT, False, "not allowed"),
            (FeatureScope.EVENT_CURRENT, SourceKind.EVENT_CURRENT, False, "not allowed"),
            (FeatureScope.GRAPH_RELATION, SourceKind.GRAPH_RELATION, False, "not allowed"),
            (FeatureScope.IDENTITY, SourceKind.IDENTITY, False, "identity"),
            (FeatureScope.UNKNOWN, SourceKind.UNKNOWN, False, "UNKNOWN"),
            (FeatureScope.STOCK_PIT, SourceKind.STOCK_PIT, True, "current-event"),
        ],
    )
    def test_rejects_forbidden_scopes_and_current_event_features(
        self,
        scope: FeatureScope,
        source: SourceKind,
        current_event: bool,
        message: str,
    ) -> None:
        specs = (
            _stock_spec("stock_volatility_5d"),
            FeatureSpec(
                name=f"bad_{scope.value}",
                version="1.0",
                horizons=[1],
                dependencies=[],
                description="forbidden fixture",
                scope=scope,
                source=source,
                is_current_event=current_event,
            ),
        )
        with pytest.raises(ValueError, match=message):
            BaseFeatureContract(specs)

    @pytest.mark.parametrize("name", ["stock_id", "stock_id_hash", "id_hash"])
    def test_rejects_identity_and_hash_features_by_metadata(self, name: str) -> None:
        specs = (
            FeatureSpec(
                name=name,
                version="1.0",
                horizons=[1],
                dependencies=[],
                description="identity fixture",
                scope=FeatureScope.IDENTITY,
                source=SourceKind.IDENTITY,
            ),
        )
        with pytest.raises(ValueError, match="identity"):
            BaseFeatureContract(specs)

    def test_rejects_duplicate_and_unknown_features(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            BaseFeatureContract((_stock_spec("x"), _stock_spec("x")))

        contract = BaseFeatureContract(_base_specs())
        with pytest.raises(ValueError, match="unknown"):
            contract.validate_tensor(torch.ones(1, 3), ["stock-a"], feature_names=["missing"])

    def test_rejects_nonfinite_values_and_dimension_drift(self) -> None:
        contract = BaseFeatureContract(_base_specs())
        with pytest.raises(ValueError, match="finite"):
            contract.validate_tensor(torch.tensor([[1.0, float("inf"), 3.0]]), ["stock-a"])
        with pytest.raises(ValueError, match="dimension"):
            contract.validate_tensor(torch.ones(1, 2), ["stock-a"])
        with pytest.raises(ValueError, match="stock metadata"):
            contract.validate_tensor(torch.ones(2, 3), ["stock-a"])

    def test_schema_fingerprint_is_deterministic_and_order_sensitive(self) -> None:
        contract_a = BaseFeatureContract(_base_specs())
        contract_b = BaseFeatureContract(_base_specs())
        contract_reordered = BaseFeatureContract(tuple(reversed(_base_specs())))
        assert contract_a.schema_fingerprint == contract_b.schema_fingerprint
        assert contract_a.schema_fingerprint != contract_reordered.schema_fingerprint


class TestBasePredictionDTO:
    def test_prediction_derives_cumulative_and_keeps_stock_id_metadata_only(self) -> None:
        prediction = BasePrediction(
            stock_id="stock-a",
            prediction_time="2026-01-09",
            cutoff_time="2026-01-08",
            daily_path=(0.01, -0.02, 0.03, 0.04, -0.05),
            feature_schema_fingerprint="sha256:abc",
        )
        assert prediction.cumulative == pytest.approx((0.01, 0.02, 0.01))
        assert prediction.classification == "FORMAL_BASE_DAILY_PATH"
        assert prediction.provenance == "BASE_ONLY"
        assert prediction.stock_id == "stock-a"

        renamed = BasePrediction(
            stock_id="renamed-stock-a",
            prediction_time=prediction.prediction_time,
            cutoff_time=prediction.cutoff_time,
            daily_path=prediction.daily_path,
            feature_schema_fingerprint=prediction.feature_schema_fingerprint,
        )
        assert renamed.cumulative == prediction.cumulative

    def test_prediction_is_immutable_and_rejects_tampered_cumulative(self) -> None:
        prediction = BasePrediction(
            stock_id="stock-a",
            prediction_time="2026-01-09",
            cutoff_time="2026-01-08",
            daily_path=(0.01, 0.02, 0.03, 0.04, 0.05),
            feature_schema_fingerprint="sha256:abc",
        )
        with pytest.raises(FrozenInstanceError):
            prediction.stock_id = "changed"  # type: ignore[misc]

        with pytest.raises(ValueError, match="cumulative"):
            BasePrediction(
                stock_id="stock-a",
                prediction_time="2026-01-09",
                cutoff_time="2026-01-08",
                daily_path=(0.01, 0.02, 0.03, 0.04, 0.05),
                feature_schema_fingerprint="sha256:abc",
                cumulative=(0.01, 0.02, 0.03),
            )


class TestBaseLoss:
    def test_loss_uses_daily_paths_not_independent_cumulative_heads(self) -> None:
        loss_fn = BaseLoss(huber_delta=1.0)
        pred = torch.tensor([[0.1, -0.05, 0.2, 0.0, 0.01]])
        target = torch.tensor([[0.12, -0.03, 0.18, 0.0, 0.02]])
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_loss_rejects_non_daily_path_targets(self) -> None:
        loss_fn = BaseLoss()
        with pytest.raises(ValueError, match="daily path"):
            loss_fn(torch.zeros(4, 5), torch.zeros(4, 3))
