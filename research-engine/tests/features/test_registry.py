"""Tests for Feature registry and PIT feature construction (Task 10)."""

from __future__ import annotations

import pytest

from seatflow.features.registry import FeatureRegistry, FeatureSpec, PREDEFINED_SPECS


# --- FeatureSpec ------------------------------------------------------


class TestFeatureSpec:
    def test_spec_is_frozen(self):
        spec = FeatureSpec(
            name="test_feature",
            version="1.0",
            horizons=[5],
            dependencies=[],
            description="A test feature",
            pit_required=True,
        )
        with pytest.raises(Exception):
            spec.name = "changed"  # type: ignore[misc]

    def test_spec_has_version(self):
        spec = FeatureSpec(
            name="f1",
            version="2.0",
            horizons=[5, 20],
            dependencies=[],
            description="desc",
            pit_required=True,
        )
        assert spec.version == "2.0"

    def test_spec_has_horizons(self):
        spec = FeatureSpec(
            name="f1",
            version="1.0",
            horizons=[5, 20, 60],
            dependencies=[],
            description="desc",
            pit_required=True,
        )
        assert spec.horizons == [5, 20, 60]

    def test_spec_pit_required_default_true(self):
        spec = FeatureSpec(
            name="f1",
            version="1.0",
            horizons=[5],
            dependencies=[],
            description="desc",
        )
        assert spec.pit_required is True

    def test_spec_pit_required_can_be_false(self):
        spec = FeatureSpec(
            name="f1",
            version="1.0",
            horizons=[5],
            dependencies=[],
            description="desc",
            pit_required=False,
        )
        assert spec.pit_required is False


# --- FeatureRegistry --------------------------------------------------


class TestFeatureRegistry:
    def test_register_and_get(self):
        reg = FeatureRegistry()
        spec = FeatureSpec(
            name="my_feature",
            version="1.0",
            horizons=[5],
            dependencies=[],
            description="desc",
        )
        reg.register("my_feature", spec)
        assert reg.get("my_feature") is spec

    def test_get_unknown_returns_none(self):
        reg = FeatureRegistry()
        assert reg.get("nonexistent") is None

    def test_list_features_returns_names(self):
        reg = FeatureRegistry()
        reg.register("f1", FeatureSpec("f1", "1.0", [5], [], "d"))
        reg.register("f2", FeatureSpec("f2", "1.0", [20], [], "d"))
        names = reg.list_features()
        assert set(names) == {"f1", "f2"}

    def test_list_features_empty(self):
        assert FeatureRegistry().list_features() == []

    def test_register_duplicate_raises(self):
        reg = FeatureRegistry()
        reg.register("f1", FeatureSpec("f1", "1.0", [5], [], "d"))
        with pytest.raises(Exception):
            reg.register("f1", FeatureSpec("f1", "2.0", [5], [], "d"))

    def test_registry_is_iterable(self):
        reg = FeatureRegistry()
        reg.register("f1", FeatureSpec("f1", "1.0", [5], [], "d"))
        reg.register("f2", FeatureSpec("f2", "1.0", [20], [], "d"))
        names = [f.name for f in reg]
        assert set(names) == {"f1", "f2"}


# --- ID leakage prevention -------------------------------------------


class TestIdLeakagePrevention:
    def test_stock_id_rejected_as_feature(self):
        reg = FeatureRegistry()
        reg.register("stock_id", FeatureSpec("stock_id", "1.0", [5], [], "id"))
        with pytest.raises(Exception):
            reg.validate_no_id_leakage(["stock_id"])

    def test_seat_id_rejected_as_feature(self):
        reg = FeatureRegistry()
        reg.register("seat_id", FeatureSpec("seat_id", "1.0", [5], [], "id"))
        with pytest.raises(Exception):
            reg.validate_no_id_leakage(["seat_id"])

    def test_event_id_rejected_as_feature(self):
        reg = FeatureRegistry()
        reg.register("event_id", FeatureSpec("event_id", "1.0", [5], [], "id"))
        with pytest.raises(Exception):
            reg.validate_no_id_leakage(["event_id"])

    def test_cycle_id_rejected_as_feature(self):
        reg = FeatureRegistry()
        reg.register("cycle_id", FeatureSpec("cycle_id", "1.0", [5], [], "id"))
        with pytest.raises(Exception):
            reg.validate_no_id_leakage(["cycle_id"])

    def test_id_hash_rejected_as_feature(self):
        reg = FeatureRegistry()
        reg.register("id_hash", FeatureSpec("id_hash", "1.0", [5], [], "id"))
        with pytest.raises(Exception):
            reg.validate_no_id_leakage(["id_hash"])

    def test_legitimate_feature_accepted(self):
        reg = FeatureRegistry()
        reg.register("seat_buy_amount_5d", PREDEFINED_SPECS["seat_buy_amount_5d"])
        # Should not raise
        reg.validate_no_id_leakage(["seat_buy_amount_5d"])

    def test_empty_list_accepted(self):
        reg = FeatureRegistry()
        reg.validate_no_id_leakage([])


# --- Predefined specs -------------------------------------------------


class TestPredefinedSpecs:
    def test_all_predefined_specs_exist(self):
        expected = {
            "seat_buy_amount_5d",
            "seat_sell_amount_5d",
            "seat_net_amount_5d",
            "seat_lhb_count_5d",
            "seat_mature_return_mean",
            "seat_win_rate",
            "event_turnover_rate",
            "event_net_amount",
            "stock_volatility_5d",
            "stock_volatility_20d",
            "market_excess_return_1d",
            "market_excess_return_3d",
            "market_excess_return_5d",
            "relay_in_count",
            "relay_out_count",
            "coalition_score",
            "conflict_score",
        }
        assert set(PREDEFINED_SPECS.keys()) == expected

    def test_predefined_specs_have_version(self):
        for name, spec in PREDEFINED_SPECS.items():
            assert spec.version, f"{name} has empty version"

    def test_predefined_specs_have_horizons(self):
        for name, spec in PREDEFINED_SPECS.items():
            assert isinstance(spec.horizons, list), f"{name} horizons not a list"

    def test_predefined_specs_have_description(self):
        for name, spec in PREDEFINED_SPECS.items():
            assert spec.description, f"{name} has empty description"

    def test_predefined_specs_no_id_leakage(self):
        reg = FeatureRegistry()
        for name, spec in PREDEFINED_SPECS.items():
            reg.register(name, spec)
        # Validating all predefined feature names must not raise.
        reg.validate_no_id_leakage(list(PREDEFINED_SPECS.keys()))

    def test_predefined_specs_pit_required(self):
        for name, spec in PREDEFINED_SPECS.items():
            assert spec.pit_required is True, f"{name} should require PIT"
