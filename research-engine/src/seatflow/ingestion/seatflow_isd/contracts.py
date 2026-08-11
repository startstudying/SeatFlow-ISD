"""V3 domain contracts for the SeatFlow-ISD V1.0 ingestion.

These frozen dataclasses are the authoritative shape that the rest of
the SeatFlow research engine consumes. The
:mod:`seatflow.ingestion.seatflow_isd.adapter` module is responsible
for producing them from the V1 JSONL records in
``bronze/silver/gold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class ISDMaturity(str, Enum):
    """Maturity classification of a SeatFlow-ISD V1 label row."""

    MATURE = "MATURE"
    UNMATURE = "UNMATURE"


@dataclass(frozen=True)
class ISDEventV3:
    """A single LHB event in V3 shape.

    Identity is ``(event_id, stock_id, trade_date, reason_codes)`` so
    that two distinct reasons on the same stock/day produce two
    deterministic events.
    """

    event_id: str
    stock_id: str
    trade_date: datetime
    available_time: datetime
    reason_codes: tuple[str, ...]
    total_buy: float
    total_sell: float
    net_amount: float
    turnover_rate: float
    event_confidence: float
    source_lineage: str
    source_version: str

    def __post_init__(self) -> None:
        if self.available_time < self.trade_date:
            raise ValueError(
                f"available_time ({self.available_time}) must be >= "
                f"trade_date ({self.trade_date})"
            )


@dataclass(frozen=True)
class ISDSeatParticipationV3:
    """A single seat's participation in an LHB event."""

    event_id: str
    stock_id: str
    trade_date: datetime
    seat_id: str
    side: str  # "BUY" or "SELL"
    amount: float
    rank_in_side: int
    identity_scope: str  # "named_seat" / "institutional_flow_proxy" / "unknown"
    available_time: datetime
    source_lineage: str
    source_version: str


@dataclass(frozen=True)
class ISDMarketBarV3:
    """A single daily OHLC bar."""

    stock_id: str
    trade_date: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float
    available_time: datetime
    source_lineage: str
    source_version: str

    def __post_init__(self) -> None:
        if self.open <= 0 or self.close <= 0 or self.high <= 0 or self.low <= 0:
            raise ValueError(
                f"market bar prices must be strictly positive for {self.stock_id}"
            )


@dataclass(frozen=True)
class ISDDailyBasicV3:
    """A single daily-basic row."""

    stock_id: str
    trade_date: datetime
    turnover_rate: float
    total_share: float
    float_share: float
    free_share: float
    total_mv: float
    circ_mv: float
    available_time: datetime
    source_lineage: str
    source_version: str


@dataclass(frozen=True)
class ISDCycleV3:
    """An event cycle produced by the SeatFlow-ISD V1 builder."""

    cycle_id: str
    stock_id: str
    cycle_start: datetime
    last_event_date: datetime
    cycle_end: datetime | None
    event_count: int
    status: str  # "ACTIVE" or "EXPIRED"


@dataclass(frozen=True)
class ISDSampleV3:
    """A single DIRECT prediction sample."""

    sample_id: str
    cycle_id: str
    stock_id: str
    prediction_date: datetime
    cutoff_time: datetime
    days_since_last_event: int
    pool_type: str  # "DIRECT" or "SPILLOVER" — V1 only emits DIRECT
    tradeability: str
    available_time: datetime
    is_tradable: bool
    is_backtest_eligible: bool


@dataclass(frozen=True)
class ISDLabelRowV3:
    """A 1/3/5-day raw return label with optional benchmark excess."""

    sample_id: str
    stock_id: str
    entry_date: datetime
    exit_1d_date: datetime
    exit_3d_date: datetime
    exit_5d_date: datetime
    horizon_end_date: datetime
    maturity: ISDMaturity
    exit_1d_return: float | None
    exit_3d_return: float | None
    exit_5d_return: float | None
    cum_excess_1d: float | None
    cum_excess_3d: float | None
    cum_excess_5d: float | None
    excess_unavailable_reason: str | None
    entry_price: float | None
    exit_1d_price: float | None
    exit_3d_price: float | None
    exit_5d_price: float | None
    materialized_at: datetime


@dataclass(frozen=True)
class ISDLhbUniverseV3:
    """The deterministic LHB stock universe derived from V1 events."""

    stock_ids: frozenset[str]
    first_seen_by_stock: Mapping[str, datetime]
    size: int


@dataclass(frozen=True)
class ISDBenchmarkPath:
    """Path to a benchmark source that can supply excess-return computation."""

    benchmark_id: str  # e.g. "CSI500"
    data_root: str  # absolute path to <root>/processed/tushare/...
    fingerprint: str  # SHA-256 over the verified benchmark manifest


@dataclass(frozen=True)
class ISDLoadResult:
    """Result of loading one SeatFlow-ISD V1 dataset version."""

    dataset_version: str
    dataset_root: str
    manifest_hash: str
    manifest_hash_verified: bool
    hash_mismatches: tuple[str, ...]
    events: tuple[ISDEventV3, ...]
    participations: tuple[ISDSeatParticipationV3, ...]
    market_bars: tuple[ISDMarketBarV3, ...]
    daily_basic: tuple[ISDDailyBasicV3, ...]
    cycles: tuple[ISDCycleV3, ...]
    samples: tuple[ISDSampleV3, ...]
    labels: tuple[ISDLabelRowV3, ...]
    universe: ISDLhbUniverseV3
    date_range: dict[str, str]
    row_counts: dict[str, int]
    source_inventory_fingerprint: str
    benchmarks: tuple[ISDBenchmarkPath, ...] = field(default_factory=tuple)
