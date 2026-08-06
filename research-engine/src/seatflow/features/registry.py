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
from enum import Enum
from typing import Iterator, Optional


# --- Identity fields forbidden as features ----------------------------


FORBIDDEN_ID_FEATURES = frozenset(
    {
        "stock_id",
        "seat_id",
        "event_id",
        "cycle_id",
        "id_hash",
        "stock_id_hash",
        "seat_id_hash",
        "event_id_hash",
    }
)


class FeatureScope(str, Enum):
    """Semantic scope of a feature's source evidence."""

    STOCK_PIT = "STOCK_PIT"
    MARKET_PIT = "MARKET_PIT"
    SEAT = "SEAT"
    EVENT_CURRENT = "EVENT_CURRENT"
    GRAPH_RELATION = "GRAPH/RELATION"
    IDENTITY = "IDENTITY"
    UNKNOWN = "UNKNOWN"


class SourceKind(str, Enum):
    """Provenance kind for a feature source."""

    STOCK_PIT = "STOCK_PIT"
    MARKET_PIT = "MARKET_PIT"
    SEAT = "SEAT"
    EVENT_CURRENT = "EVENT_CURRENT"
    GRAPH_RELATION = "GRAPH/RELATION"
    IDENTITY = "IDENTITY"
    UNKNOWN = "UNKNOWN"


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
    scope:
        Explicit semantic scope. Base may only consume STOCK_PIT/MARKET_PIT.
    source:
        Explicit provenance source kind.
    is_current_event:
        Whether the feature is derived from the current event at or after cutoff.
    """

    name: str
    version: str
    horizons: list[int]
    dependencies: list[str]
    description: str
    pit_required: bool = True
    scope: FeatureScope = FeatureScope.UNKNOWN
    source: SourceKind = SourceKind.UNKNOWN
    is_current_event: bool = False

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


def _feature(
    name: str,
    version: str,
    description: str,
    *,
    scope: FeatureScope,
    source: SourceKind,
    horizons: tuple[int, ...] = (5,),
    dependencies: tuple[str, ...] = (),
    is_current_event: bool = False,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        version=version,
        horizons=list(horizons),
        dependencies=list(dependencies),
        description=description,
        pit_required=True,
        scope=scope,
        source=source,
        is_current_event=is_current_event,
    )


PREDEFINED_SPECS: dict[str, FeatureSpec] = {
    # --- Seat-level features (5d / 20d mature windows) ----------------
    "seat_buy_amount_5d": _feature(
        "seat_buy_amount_5d", "1.0", "Total buy amount by the seat over 5 trading days.",
        scope=FeatureScope.SEAT, source=SourceKind.SEAT, horizons=(5,),
    ),
    "seat_sell_amount_5d": _feature(
        "seat_sell_amount_5d", "1.0", "Total sell amount by the seat over 5 trading days.",
        scope=FeatureScope.SEAT, source=SourceKind.SEAT, horizons=(5,),
    ),
    "seat_net_amount_5d": _feature(
        "seat_net_amount_5d", "1.0", "Net (buy - sell) amount by the seat over 5 trading days.",
        scope=FeatureScope.SEAT, source=SourceKind.SEAT,
        horizons=(5,), dependencies=("seat_buy_amount_5d", "seat_sell_amount_5d"),
    ),
    "seat_lhb_count_5d": _feature(
        "seat_lhb_count_5d", "1.0", "Number of LHB participations by the seat over 5 trading days.",
        scope=FeatureScope.SEAT, source=SourceKind.SEAT, horizons=(5,),
    ),
    "seat_mature_return_mean": _feature(
        "seat_mature_return_mean", "1.0", "Mean mature return of stocks the seat participated in.",
        scope=FeatureScope.SEAT, source=SourceKind.SEAT, horizons=(5, 20),
    ),
    "seat_win_rate": _feature(
        "seat_win_rate", "1.0", "Fraction of mature returns > 0 for the seat.", horizons=(5, 20),
        scope=FeatureScope.SEAT, source=SourceKind.SEAT,
        dependencies=("seat_mature_return_mean",),
    ),
    # --- Event-level features -----------------------------------------
    "event_turnover_rate": _feature(
        "event_turnover_rate", "1.0", "Turnover rate of the stock on the LHB event day.",
        scope=FeatureScope.EVENT_CURRENT, source=SourceKind.EVENT_CURRENT, horizons=(1,),
        is_current_event=True,
    ),
    "event_net_amount": _feature(
        "event_net_amount", "1.0", "Net amount (buy - sell) on the LHB event day.",
        scope=FeatureScope.EVENT_CURRENT, source=SourceKind.EVENT_CURRENT, horizons=(1,),
        is_current_event=True,
    ),
    # --- Stock-level features -----------------------------------------
    "stock_volatility_5d": _feature(
        "stock_volatility_5d", "1.0", "Realized volatility of the stock over 5 trading days.",
        scope=FeatureScope.STOCK_PIT, source=SourceKind.STOCK_PIT, horizons=(5,),
    ),
    "stock_volatility_20d": _feature(
        "stock_volatility_20d", "1.0", "Realized volatility of the stock over 20 trading days.",
        scope=FeatureScope.STOCK_PIT, source=SourceKind.STOCK_PIT, horizons=(20,),
    ),
    # --- Market-level features ----------------------------------------
    "market_excess_return_1d": _feature(
        "market_excess_return_1d", "1.0", "Market excess return over 1 trading day.",
        scope=FeatureScope.MARKET_PIT, source=SourceKind.MARKET_PIT, horizons=(1,),
    ),
    "market_excess_return_3d": _feature(
        "market_excess_return_3d", "1.0", "Market excess return over 3 trading days.",
        scope=FeatureScope.MARKET_PIT, source=SourceKind.MARKET_PIT, horizons=(3,),
    ),
    "market_excess_return_5d": _feature(
        "market_excess_return_5d", "1.0", "Market excess return over 5 trading days.",
        scope=FeatureScope.MARKET_PIT, source=SourceKind.MARKET_PIT, horizons=(5,),
    ),
    # --- Graph propagation features -----------------------------------
    "relay_in_count": _feature(
        "relay_in_count", "1.0", "Count of propagation relays received (in-degree).",
        scope=FeatureScope.GRAPH_RELATION, source=SourceKind.GRAPH_RELATION, horizons=(5, 20),
    ),
    "relay_out_count": _feature(
        "relay_out_count", "1.0", "Count of propagation relays sent (out-degree).",
        scope=FeatureScope.GRAPH_RELATION, source=SourceKind.GRAPH_RELATION, horizons=(5, 20),
    ),
    "coalition_score": _feature(
        "coalition_score", "1.0", "Jaccard coalition overlap score with peer seats.",
        scope=FeatureScope.GRAPH_RELATION, source=SourceKind.GRAPH_RELATION, horizons=(5, 20),
    ),
    "conflict_score": _feature(
        "conflict_score", "1.0", "Directional conflict score (opposite-side co-occurrence).",
        scope=FeatureScope.GRAPH_RELATION, source=SourceKind.GRAPH_RELATION, horizons=(5, 20),
    ),
}


def default_registry() -> FeatureRegistry:
    """Return a :class:`FeatureRegistry` pre-populated with all predefined
    feature specs."""
    reg = FeatureRegistry()
    for name, spec in PREDEFINED_SPECS.items():
        reg.register(name, spec)
    return reg
