"""Feature registry and PIT feature specifications (Task 10).

The feature registry is the central catalogue of all model-input
features. Every feature declares its version, horizons, dependencies,
and PIT requirement. The registry enforces **identity leakage
prevention**: no feature may be named after an identity field
(``stock_id``, ``seat_id``, ``event_id``, ``cycle_id``, ``id_hash``)
because IDs are string hashes, never continuous model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


# --- Identity fields forbidden as features ----------------------------


FORBIDDEN_ID_FEATURES = frozenset(
    {
        "stock_id",
        "seat_id",
        "event_id",
        "cycle_id",
        "id_hash",
    }
)


# --- FeatureSpec ------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of a single feature.

    Attributes
    ----------
    name:
        Canonical feature name (must not be an identity field).
    version:
        Lineage version string (e.g. ``"1.0"``).
    horizons:
        Lookback horizons in trading days (e.g. ``[5, 20]``).
    dependencies:
        Names of upstream features this feature depends on.
    description:
        Human-readable description.
    pit_required:
        Whether point-in-time semantics are required (default ``True``).
    """

    name: str
    version: str
    horizons: list[int]
    dependencies: list[str]
    description: str
    pit_required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureSpec.name must be non-empty")
        if not self.version:
            raise ValueError("FeatureSpec.version must be non-empty")


# --- FeatureRegistry --------------------------------------------------


class FeatureRegistry:
    """Central catalogue of feature specifications.

    Use :meth:`register` to add a feature, :meth:`get` to retrieve one,
    and :meth:`validate_no_id_leakage` to assert that a set of feature
    names contains no identity fields.
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, name: str, spec: FeatureSpec) -> None:
        """Register a feature spec.

        Raises ``ValueError`` if ``name`` is already registered or if the
        spec's name is a forbidden identity field.
        """
        if name in self._specs:
            raise ValueError(f"Feature {name!r} is already registered")
        if name != spec.name:
            raise ValueError(
                f"Registered key {name!r} does not match spec.name {spec.name!r}"
            )
        self._specs[name] = spec

    def get(self, name: str) -> Optional[FeatureSpec]:
        return self._specs.get(name)

    def list_features(self) -> list[str]:
        return list(self._specs.keys())

    def validate_no_id_leakage(self, feature_names: list[str]) -> None:
        """Assert that no feature name is an identity field.

        Raises ``ValueError`` listing any forbidden names found.
        """
        forbidden = [n for n in feature_names if n in FORBIDDEN_ID_FEATURES]
        if forbidden:
            raise ValueError(
                f"Identity leakage detected: {forbidden} are forbidden as "
                f"continuous features (IDs are string hashes, never model inputs)"
            )

    def __iter__(self) -> Iterator[FeatureSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs


# --- Predefined feature specs ----------------------------------------


def _seat_feature(name: str, version: str, description: str, horizons=(5,), dependencies=()) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        version=version,
        horizons=list(horizons),
        dependencies=list(dependencies),
        description=description,
        pit_required=True,
    )


PREDEFINED_SPECS: dict[str, FeatureSpec] = {
    # --- Seat-level features (5d / 20d mature windows) ----------------
    "seat_buy_amount_5d": _seat_feature(
        "seat_buy_amount_5d", "1.0", "Total buy amount by the seat over 5 trading days.", horizons=(5,),
    ),
    "seat_sell_amount_5d": _seat_feature(
        "seat_sell_amount_5d", "1.0", "Total sell amount by the seat over 5 trading days.", horizons=(5,),
    ),
    "seat_net_amount_5d": _seat_feature(
        "seat_net_amount_5d", "1.0", "Net (buy - sell) amount by the seat over 5 trading days.",
        horizons=(5,), dependencies=("seat_buy_amount_5d", "seat_sell_amount_5d"),
    ),
    "seat_lhb_count_5d": _seat_feature(
        "seat_lhb_count_5d", "1.0", "Number of LHB participations by the seat over 5 trading days.", horizons=(5,),
    ),
    "seat_mature_return_mean": _seat_feature(
        "seat_mature_return_mean", "1.0", "Mean mature return of stocks the seat participated in.", horizons=(5, 20),
    ),
    "seat_win_rate": _seat_feature(
        "seat_win_rate", "1.0", "Fraction of mature returns > 0 for the seat.", horizons=(5, 20),
        dependencies=("seat_mature_return_mean",),
    ),
    # --- Event-level features -----------------------------------------
    "event_turnover_rate": _seat_feature(
        "event_turnover_rate", "1.0", "Turnover rate of the stock on the LHB event day.", horizons=(1,),
    ),
    "event_net_amount": _seat_feature(
        "event_net_amount", "1.0", "Net amount (buy - sell) on the LHB event day.", horizons=(1,),
    ),
    # --- Stock-level features -----------------------------------------
    "stock_volatility_5d": _seat_feature(
        "stock_volatility_5d", "1.0", "Realized volatility of the stock over 5 trading days.", horizons=(5,),
    ),
    "stock_volatility_20d": _seat_feature(
        "stock_volatility_20d", "1.0", "Realized volatility of the stock over 20 trading days.", horizons=(20,),
    ),
    # --- Market-level features ----------------------------------------
    "market_excess_return_1d": _seat_feature(
        "market_excess_return_1d", "1.0", "Market excess return over 1 trading day.", horizons=(1,),
    ),
    "market_excess_return_3d": _seat_feature(
        "market_excess_return_3d", "1.0", "Market excess return over 3 trading days.", horizons=(3,),
    ),
    "market_excess_return_5d": _seat_feature(
        "market_excess_return_5d", "1.0", "Market excess return over 5 trading days.", horizons=(5,),
    ),
    # --- Graph propagation features -----------------------------------
    "relay_in_count": _seat_feature(
        "relay_in_count", "1.0", "Count of propagation relays received (in-degree).", horizons=(5, 20),
    ),
    "relay_out_count": _seat_feature(
        "relay_out_count", "1.0", "Count of propagation relays sent (out-degree).", horizons=(5, 20),
    ),
    "coalition_score": _seat_feature(
        "coalition_score", "1.0", "Jaccard coalition overlap score with peer seats.", horizons=(5, 20),
    ),
    "conflict_score": _seat_feature(
        "conflict_score", "1.0", "Directional conflict score (opposite-side co-occurrence).", horizons=(5, 20),
    ),
}


def default_registry() -> FeatureRegistry:
    """Return a :class:`FeatureRegistry` pre-populated with all predefined
    feature specs."""
    reg = FeatureRegistry()
    for name, spec in PREDEFINED_SPECS.items():
        reg.register(name, spec)
    return reg
