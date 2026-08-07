"""Source-to-domain ISD adapter and label builder (Task 3).

Converts verified Tushare source snapshots into V3 domain records
(LHBEvent, SeatParticipation, market bars, daily-basic) while preserving:

- Timezone-aware UTC conversion of all availability timestamps from
  Asia/Shanghai.
- Deterministic event IDs derived from source lineage and the natural
  key ``(stock_id, trade_date, normalized reason)``.
- A separate ``identity_scope`` accounting for ``named_seat``,
  ``institutional_flow_proxy``, and unknown scopes — proxy rows remain
  usable engineering evidence but are never relabelled as named seats.
- The LHB-only stock universe boundary: stocks that never appear in the
  dragon-tiger summary are filtered out of market and daily-basic
  outputs and must never become dataset members.
- Mature label rows that derive 1/3/5-day raw returns from verified
  five-bar future windows. Tail samples without a full window remain
  present as ``UNMATURE`` with null returns.
- Market excess fields stay null with an explicit reason when no
  verified benchmark source is supplied; an equal-weight market is
  never fabricated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from seatflow.dataset.source_snapshots import (
    SnapshotRecord,
    discover_and_load_records,
    discover_snapshot_records,
)
from seatflow.domain.time import assert_aware


SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc


# ── Enums ──────────────────────────────────────────────────────────────


class SeatSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class IdentityScope(str, Enum):
    NAMED_SEAT = "named_seat"
    INSTITUTIONAL_FLOW_PROXY = "institutional_flow_proxy"
    UNKNOWN = "unknown"


class LabelMaturity(str, Enum):
    MATURE = "MATURE"
    UNMATURE = "UNMATURE"


# ── Adapter statistics ────────────────────────────────────────────────


@dataclass(frozen=True)
class AdapterStats:
    summary_row_count: int = 0
    event_count: int = 0
    participation_count: int = 0
    named_seat_count: int = 0
    institutional_flow_proxy_count: int = 0
    unknown_scope_count: int = 0
    unbound_seat_count: int = 0
    market_bar_count: int = 0
    daily_basic_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "summary_row_count": self.summary_row_count,
            "event_count": self.event_count,
            "participation_count": self.participation_count,
            "named_seat_count": self.named_seat_count,
            "institutional_flow_proxy_count": self.institutional_flow_proxy_count,
            "unknown_scope_count": self.unknown_scope_count,
            "unbound_seat_count": self.unbound_seat_count,
            "market_bar_count": self.market_bar_count,
            "daily_basic_count": self.daily_basic_count,
        }


# ── Domain dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class ISDEventRecord:
    event_id: str
    stock_id: str
    trade_date: datetime
    available_time: datetime
    reason_codes: list[str]
    total_buy: float
    total_sell: float
    net_amount: float
    turnover_rate: float
    event_confidence: float
    source_lineage: str
    source_version: str

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")
        if self.available_time < self.trade_date:
            raise ValueError(
                f"available_time ({self.available_time}) must be >= trade_date "
                f"({self.trade_date})"
            )


@dataclass(frozen=True)
class ISDSeatParticipation:
    event_id: str
    stock_id: str
    trade_date: datetime
    seat_id: str
    side: SeatSide
    amount: float
    rank_in_side: int
    identity_scope: IdentityScope
    available_time: datetime
    source_lineage: str
    source_version: str

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")


@dataclass(frozen=True)
class ISDMarketBar:
    stock_id: str
    trade_date: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float
    available_time: datetime
    source_lineage: str = "tushare"
    source_version: str = "tushare-pro-rest"

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")
        if self.open <= 0 or self.close <= 0 or self.high <= 0 or self.low <= 0:
            raise ValueError(
                f"market bar prices must be strictly positive for {self.stock_id} "
                f"on {self.trade_date.date().isoformat()}"
            )


@dataclass(frozen=True)
class ISDDailyBasic:
    stock_id: str
    trade_date: datetime
    turnover_rate: float
    total_share: float
    float_share: float
    free_share: float
    total_mv: float
    circ_mv: float
    available_time: datetime
    source_lineage: str = "tushare"
    source_version: str = "daily_basic"

    def __post_init__(self) -> None:
        assert_aware(self.trade_date, "trade_date")
        assert_aware(self.available_time, "available_time")


@dataclass(frozen=True)
class ISDIndustryMembership:
    stock_id: str
    industry_code: str
    industry_name: str
    available_time: datetime
    source_lineage: str = "tushare"
    source_version: str = "sw2021_l1_industry_membership"

    def __post_init__(self) -> None:
        assert_aware(self.available_time, "available_time")


@dataclass(frozen=True)
class LHBStockUniverse:
    stock_ids: frozenset[str]
    first_seen_by_stock: Mapping[str, date]

    def as_dict(self) -> dict[str, object]:
        return {
            "stock_ids": sorted(self.stock_ids),
            "first_seen_by_stock": {
                stock_id: when.isoformat()
                for stock_id, when in sorted(self.first_seen_by_stock.items())
            },
            "size": len(self.stock_ids),
        }


@dataclass(frozen=True)
class ISDInputs:
    events: tuple[ISDEventRecord, ...]
    participations: tuple[ISDSeatParticipation, ...]
    market_data: tuple[ISDMarketBar, ...]
    daily_basic: tuple[ISDDailyBasic, ...]
    industry_membership: tuple[ISDIndustryMembership, ...]
    trade_calendar: tuple[datetime, ...]
    source_inventory_fingerprint: str
    stats: AdapterStats


# ── Label dataclass ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ISDLabelRow:
    sample_id: str
    stock_id: str
    entry_date: datetime
    exit_1d_date: datetime
    exit_3d_date: datetime
    exit_5d_date: datetime
    horizon_end_date: datetime
    maturity: LabelMaturity
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

    def __post_init__(self) -> None:
        assert_aware(self.entry_date, "entry_date")
        assert_aware(self.exit_1d_date, "exit_1d_date")
        assert_aware(self.exit_3d_date, "exit_3d_date")
        assert_aware(self.exit_5d_date, "exit_5d_date")
        assert_aware(self.horizon_end_date, "horizon_end_date")
        assert_aware(self.materialized_at, "materialized_at")


# ── Time helpers ─────────────────────────────────────────────────────


def _parse_aware_utc(raw: str) -> datetime:
    """Parse an ISO-8601 string and convert to aware UTC."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {raw!r} is naive; offsets are required")
    return parsed.astimezone(UTC)


def _to_aware_utc_from_date(
    raw: str,
    *,
    default_timezone: ZoneInfo = SHANGHAI,
) -> datetime:
    """Parse a YYYY-MM-DD string as midnight in Shanghai, convert to UTC."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed.astimezone(UTC)


# ── Reason normalization ─────────────────────────────────────────────


def _normalize_reason(raw: object) -> str:
    """Return a canonical, deduped, comma-joined reason string."""
    if isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
    else:
        parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return ",".join(out)


# ── ID generation ───────────────────────────────────────────────────


def _event_id(stock_id: str, trade_date: datetime, reason: str) -> str:
    raw = f"{stock_id}|{trade_date.astimezone(UTC).isoformat()}|{reason}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"lhb-{digest}"


def _participation_id(event_id: str, seat_id: str, side: str, rank: int) -> str:
    raw = f"{event_id}|{seat_id}|{side}|{rank}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"prt-{digest}"


# ── Identity scope mapping ──────────────────────────────────────────


_IDENTITY_SCOPE_MAP: dict[str, IdentityScope] = {
    "named_seat": IdentityScope.NAMED_SEAT,
    "institutional_flow_proxy": IdentityScope.INSTITUTIONAL_FLOW_PROXY,
}


def _classify_identity_scope(raw: object) -> IdentityScope:
    if isinstance(raw, str):
        return _IDENTITY_SCOPE_MAP.get(raw, IdentityScope.UNKNOWN)
    return IdentityScope.UNKNOWN


# ── Source layout ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _SourceLayout:
    summary_path: Path
    seat_path: Path
    bar_path: Path
    basic_path: Path
    industry_path: Path


def _resolve_source_layout(source_root: Path) -> _SourceLayout:
    candidates = [
        _SourceLayout(
            summary_path=source_root / "processed/tushare/dragon_tiger_summary",
            seat_path=source_root / "processed/tushare/dragon_tiger_seat_event",
            bar_path=source_root / "processed/tushare/daily_bar",
            basic_path=source_root / "r1_daily_basic/tushare/daily_basic",
            industry_path=source_root / "r1_industry_membership/tushare/sw2021_l1_industry_membership",
        ),
        _SourceLayout(
            summary_path=source_root / "dragon_tiger_summary",
            seat_path=source_root / "dragon_tiger_seat_event",
            bar_path=source_root / "daily_bar",
            basic_path=source_root / "daily_basic",
            industry_path=source_root / "sw2021_l1_industry_membership",
        ),
    ]
    for layout in candidates:
        if layout.summary_path.exists():
            return layout
    return candidates[0]


def _load_records(path: Path) -> tuple[SnapshotRecord, ...]:
    if not path.exists():
        return ()
    return discover_snapshot_records(path.parent, Path(path.name))


def _load_and_filter_rows(
    path: Path, start: date, end: date
) -> tuple[list[dict[str, Any]], tuple[SnapshotRecord, ...]]:
    """Discover, verify, and load rows in a single pass.

    Unlike ``_load_records`` + ``load_verified_rows``, this reads each
    file only once, which is critical for large source windows.
    Returns the loaded rows *and* the matching SnapshotRecords.
    """
    if not path.exists():
        return [], ()
    pairs = discover_and_load_records(path.parent, Path(path.name))
    rows: list[dict[str, Any]] = []
    records: list[SnapshotRecord] = []
    for record, loaded in pairs:
        if start <= record.trade_date <= end:
            rows.extend(loaded)
            records.append(record)
    return rows, tuple(records)


def _date_filter(
    records: tuple[SnapshotRecord, ...],
    *,
    start_date: date,
    end_date: date,
) -> tuple[SnapshotRecord, ...]:
    return tuple(r for r in records if start_date <= r.trade_date <= end_date)


# ── Row builders ────────────────────────────────────────────────────


def _build_event(row: dict[str, Any]) -> ISDEventRecord:
    stock_id = str(row.get("stock_id", ""))
    trade_date = _to_aware_utc_from_date(str(row["trade_date"]))
    available_time = _parse_aware_utc(str(row["available_time"]))
    reason = _normalize_reason(row.get("reason", ""))
    if not reason:
        raise ValueError(
            f"summary row for {stock_id} on {trade_date.date().isoformat()} "
            f"has no reason code"
        )
    total_buy = float(row.get("buy_amount", 0.0) or 0.0)
    total_sell = float(row.get("sell_amount", 0.0) or 0.0)
    net_amount = float(
        row.get("net_amount", total_buy - total_sell) or total_buy - total_sell
    )
    turnover_rate = float(row.get("turnover_rate", 0.0) or 0.0)
    source_version = str(row.get("source_version", "top_list"))
    return ISDEventRecord(
        event_id=_event_id(stock_id, trade_date, reason),
        stock_id=stock_id,
        trade_date=trade_date,
        available_time=available_time,
        reason_codes=[reason],
        total_buy=total_buy,
        total_sell=total_sell,
        net_amount=net_amount,
        turnover_rate=turnover_rate,
        event_confidence=1.0,
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=source_version,
    )


def _build_participation(
    row: dict[str, Any], event_lookup: dict[tuple[str, datetime, str], str]
) -> ISDSeatParticipation:
    stock_id = str(row.get("stock_id", ""))
    trade_date = _to_aware_utc_from_date(str(row["trade_date"]))
    reason = _normalize_reason(row.get("event_reason", ""))
    available_time = _parse_aware_utc(str(row["available_time"]))
    side_raw = str(row.get("side", "")).strip().lower()
    side = SeatSide.BUY if side_raw == "buy" else SeatSide.SELL
    key = (stock_id, trade_date, reason)
    event_id = event_lookup.get(key, "")
    identity_scope = _classify_identity_scope(row.get("identity_scope"))
    rank = int(row.get("rank", 0) or 0)
    amount = float(row.get("amount", 0.0) or 0.0)
    return ISDSeatParticipation(
        event_id=event_id,
        stock_id=stock_id,
        trade_date=trade_date,
        seat_id=str(row.get("seat_id", "")),
        side=side,
        amount=amount,
        rank_in_side=rank,
        identity_scope=identity_scope,
        available_time=available_time,
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=str(row.get("source_version", "top_inst")),
    )


def _build_market_bar(row: dict[str, Any]) -> ISDMarketBar:
    stock_id = str(row.get("stock_id", ""))
    trade_date = _to_aware_utc_from_date(str(row["trade_date"]))
    available_time = _parse_aware_utc(str(row["available_time"]))
    open_ = float(row.get("open_price", 0.0) or 0.0)
    close = float(row.get("close_price", 0.0) or 0.0)
    high = float(row.get("high_price", max(open_, close)) or max(open_, close))
    low = float(row.get("low_price", min(open_, close)) or min(open_, close))
    volume = float(row.get("volume", 0.0) or 0.0)
    return ISDMarketBar(
        stock_id=stock_id,
        trade_date=trade_date,
        open=open_,
        close=close,
        high=high,
        low=low,
        volume=volume,
        available_time=available_time,
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=str(row.get("source_version", "tushare-pro-rest")),
    )


def _build_daily_basic(row: dict[str, Any]) -> ISDDailyBasic:
    stock_id = str(row.get("stock_id", ""))
    trade_date = _to_aware_utc_from_date(str(row["trade_date"]))
    available_time = _parse_aware_utc(str(row["available_time"]))
    return ISDDailyBasic(
        stock_id=stock_id,
        trade_date=trade_date,
        turnover_rate=float(row.get("turnover_rate", 0.0) or 0.0),
        total_share=float(row.get("total_share", 0.0) or 0.0),
        float_share=float(row.get("float_share", 0.0) or 0.0),
        free_share=float(row.get("free_share", 0.0) or 0.0),
        total_mv=float(row.get("total_mv", 0.0) or 0.0),
        circ_mv=float(row.get("circ_mv", 0.0) or 0.0),
        available_time=available_time,
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=str(row.get("source_version", "daily_basic")),
    )


def _build_industry_membership(row: dict[str, Any]) -> ISDIndustryMembership:
    available_time = _parse_aware_utc(str(row["available_time"]))
    return ISDIndustryMembership(
        stock_id=str(row.get("stock_id", "")),
        industry_code=str(row.get("industry_code", "")),
        industry_name=str(row.get("industry_name", "")),
        available_time=available_time,
        source_lineage=str(row.get("source_name", "tushare")),
        source_version=str(row.get("source_version", "sw2021_l1_industry_membership")),
    )


# ── Universe construction ──────────────────────────────────────────


def build_lhb_stock_universe(inputs: ISDInputs) -> LHBStockUniverse:
    """Derive the deterministic LHB stock universe from event records.

    Membership rule: a stock enters the universe the first time it
    appears in any verified LHB summary snapshot. The universe is the
    exclusive source of truth for the dataset's stock dimension —
    stocks that never appear in the LHB summary are never included
    even if they have market-data coverage.
    """
    first_seen: dict[str, date] = {}
    for event in inputs.events:
        existing = first_seen.get(event.stock_id)
        if existing is None or event.trade_date.date() < existing:
            first_seen[event.stock_id] = event.trade_date.date()
    return LHBStockUniverse(
        stock_ids=frozenset(first_seen.keys()),
        first_seen_by_stock=dict(sorted(first_seen.items())),
    )


# ── Main adapter entry point ───────────────────────────────────────


def build_isd_inputs(
    source_root: Path,
    *,
    start_date: str,
    end_date: str,
) -> ISDInputs:
    """Read verified source snapshots and produce V3 domain records.

    Source snapshots under ``source_root`` are re-validated against
    their manifest hashes by :func:`discover_and_load_records`; tampered or
    missing snapshots raise immediately.

    The function never emits SPILLOVER or propagation candidates: this
    build is LHB-only, and the returned domain records are the only
    authoritative inputs to the dataset builder.
    """
    layout = _resolve_source_layout(source_root)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Single-pass discovery + load: each file is read and verified only
    # once, avoiding doubled I/O on large source windows (870+ dates).
    summary_rows, summary_records = _load_and_filter_rows(
        layout.summary_path, start, end
    )
    seat_rows, seat_records = _load_and_filter_rows(
        layout.seat_path, start, end
    )
    bar_rows, bar_records = _load_and_filter_rows(
        layout.bar_path, start, end
    )
    basic_rows, basic_records = _load_and_filter_rows(
        layout.basic_path, start, end
    )
    industry_rows, industry_records = _load_and_filter_rows(
        layout.industry_path, start, end
    )

    # 1) Build the LHB stock universe from every LHB summary row.
    universe_first_seen: dict[str, date] = {}
    for row in summary_rows:
        if "trade_date" not in row:
            continue
        try:
            trade_date = _to_aware_utc_from_date(str(row["trade_date"])).date()
        except (KeyError, ValueError):
            continue
        stock_id = str(row.get("stock_id", ""))
        if not stock_id:
            continue
        existing = universe_first_seen.get(stock_id)
        if existing is None or trade_date < existing:
            universe_first_seen[stock_id] = trade_date
    universe_ids = frozenset(universe_first_seen.keys())

    # 2) Build events: one per (stock_id, trade_date, normalized reason).
    events: list[ISDEventRecord] = []
    seen_event_keys: set[tuple[str, datetime, str]] = set()
    for row in summary_rows:
        if "trade_date" not in row:
            continue
        try:
            ev = _build_event(row)
        except (KeyError, ValueError):
            continue
        key = (ev.stock_id, ev.trade_date, ev.reason_codes[0])
        if key in seen_event_keys:
            continue
        seen_event_keys.add(key)
        events.append(ev)
    events.sort(
        key=lambda ev: (ev.stock_id, ev.trade_date, ev.reason_codes[0], ev.event_id)
    )

    # 3) Build participations; only those that match an event are bound.
    event_lookup: dict[tuple[str, datetime, str], str] = {
        (ev.stock_id, ev.trade_date, ev.reason_codes[0]): ev.event_id for ev in events
    }
    participations: list[ISDSeatParticipation] = []
    unbound = 0
    named = proxy = unknown = 0
    for row in seat_rows:
        if "trade_date" not in row:
            continue
        try:
            p = _build_participation(row, event_lookup)
        except (KeyError, ValueError):
            continue
        if not p.event_id:
            unbound += 1
            continue
        participations.append(p)
        if p.identity_scope is IdentityScope.NAMED_SEAT:
            named += 1
        elif p.identity_scope is IdentityScope.INSTITUTIONAL_FLOW_PROXY:
            proxy += 1
        else:
            unknown += 1
    participations.sort(
        key=lambda p: (p.event_id, p.side.value, p.rank_in_side, p.seat_id)
    )

    # 4) Build market and daily-basic rows; filter to the LHB stock
    #    universe. Rows for stocks that never appeared in the LHB
    #    summary are dropped here, before the dataset is written.
    market_bars: list[ISDMarketBar] = []
    for row in bar_rows:
        if "trade_date" not in row:
            continue
        try:
            bar = _build_market_bar(row)
        except (KeyError, ValueError):
            continue
        if bar.stock_id not in universe_ids:
            continue
        market_bars.append(bar)
    market_bars.sort(key=lambda b: (b.stock_id, b.trade_date))

    basic_rows_built: list[ISDDailyBasic] = []
    for row in basic_rows:
        if "trade_date" not in row:
            continue
        try:
            basic = _build_daily_basic(row)
        except (KeyError, ValueError):
            continue
        if basic.stock_id not in universe_ids:
            continue
        basic_rows_built.append(basic)
    basic_rows_built.sort(key=lambda b: (b.stock_id, b.trade_date))

    industry_built: tuple[ISDIndustryMembership, ...] = tuple(
        sorted(
            (
                _build_industry_membership(row)
                for row in industry_rows
                if "available_time" in row
            ),
            key=lambda r: (r.stock_id, r.available_time),
        )
    )

    # 5) Calendar: every date present in either the market bars or the
    #    events, deduplicated and sorted.
    calendar_set: set[datetime] = set()
    for ev in events:
        calendar_set.add(ev.trade_date)
    for mb in market_bars:
        calendar_set.add(mb.trade_date)
    for bb in basic_rows_built:
        calendar_set.add(bb.trade_date)
    trade_calendar = tuple(sorted(calendar_set))

    inventory_fingerprint = _inventory_fingerprint(
        summary_records,
        seat_records,
        bar_records,
        basic_records,
        industry_records,
    )

    stats = AdapterStats(
        summary_row_count=len(summary_rows),
        event_count=len(events),
        participation_count=len(participations),
        named_seat_count=named,
        institutional_flow_proxy_count=proxy,
        unknown_scope_count=unknown,
        unbound_seat_count=unbound,
        market_bar_count=len(market_bars),
        daily_basic_count=len(basic_rows_built),
    )

    return ISDInputs(
        events=tuple(events),
        participations=tuple(participations),
        market_data=tuple(market_bars),
        daily_basic=tuple(basic_rows_built),
        industry_membership=industry_built,
        trade_calendar=trade_calendar,
        source_inventory_fingerprint=inventory_fingerprint,
        stats=stats,
    )


def _inventory_fingerprint(
    summary_records: tuple[SnapshotRecord, ...],
    seat_records: tuple[SnapshotRecord, ...],
    bar_records: tuple[SnapshotRecord, ...],
    basic_records: tuple[SnapshotRecord, ...],
    industry_records: tuple[SnapshotRecord, ...],
) -> str:
    """Compute a stable SHA-256 over the verified snapshot fingerprints."""
    payload = {
        "summary": [
            (r.trade_date.isoformat(), r.row_count, r.sha256) for r in summary_records
        ],
        "seat": [
            (r.trade_date.isoformat(), r.row_count, r.sha256) for r in seat_records
        ],
        "bar": [
            (r.trade_date.isoformat(), r.row_count, r.sha256) for r in bar_records
        ],
        "basic": [
            (r.trade_date.isoformat(), r.row_count, r.sha256) for r in basic_records
        ],
        "industry": [
            (r.trade_date.isoformat(), r.row_count, r.sha256) for r in industry_records
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Label builder ──────────────────────────────────────────────────


def build_isd_labels(
    *,
    sample_id: str,
    stock_id: str,
    entry_date: datetime,
    horizon_end_date: datetime,
    market_bars: Sequence[ISDMarketBar],
    benchmark_bars: Sequence[ISDMarketBar] | None,
    materialized_at: datetime,
) -> tuple[ISDLabelRow, ...]:
    """Build mature/unmature label rows for one sample.

    A label is mature only when a full 1/3/5 trading-day path of
    verified stock bars is observable at or before ``materialized_at``.
    Tail samples without a full path remain present as ``UNMATURE`` with
    null returns — zero-filling is forbidden.

    Market excess fields stay null with an explicit reason when no
    verified benchmark bars are supplied; an equal-weight market is
    never fabricated.
    """
    assert_aware(entry_date, "entry_date")
    assert_aware(horizon_end_date, "horizon_end_date")
    assert_aware(materialized_at, "materialized_at")

    stock_path = [b for b in market_bars if b.stock_id == stock_id]
    stock_path.sort(key=lambda b: b.trade_date)
    entry_bar = _find_bar(stock_path, entry_date)
    exit_1d_bar = _find_bar_n_after(stock_path, entry_date, 0)
    exit_3d_bar = _find_bar_n_after(stock_path, entry_date, 2)
    exit_5d_bar = _find_bar_n_after(stock_path, entry_date, 4)

    if benchmark_bars is None:
        excess_reason = "verified benchmark source not provided"
    else:
        excess_reason = "verified benchmark not bound in this build"

    mature_path = (
        entry_bar is not None
        and exit_1d_bar is not None
        and exit_3d_bar is not None
        and exit_5d_bar is not None
        and exit_5d_bar.trade_date <= materialized_at
    )
    if not mature_path:
        return (
            ISDLabelRow(
                sample_id=sample_id,
                stock_id=stock_id,
                entry_date=entry_date,
                exit_1d_date=exit_1d_bar.trade_date if exit_1d_bar else entry_date,
                exit_3d_date=exit_3d_bar.trade_date if exit_3d_bar else entry_date,
                exit_5d_date=exit_5d_bar.trade_date if exit_5d_bar else entry_date,
                horizon_end_date=horizon_end_date,
                maturity=LabelMaturity.UNMATURE,
                exit_1d_return=None,
                exit_3d_return=None,
                exit_5d_return=None,
                cum_excess_1d=None,
                cum_excess_3d=None,
                cum_excess_5d=None,
                excess_unavailable_reason=excess_reason,
                entry_price=entry_bar.open if entry_bar else None,
                exit_1d_price=exit_1d_bar.close if exit_1d_bar else None,
                exit_3d_price=exit_3d_bar.close if exit_3d_bar else None,
                exit_5d_price=exit_5d_bar.close if exit_5d_bar else None,
                materialized_at=materialized_at,
            ),
        )

    # The frozen contract: 1/3/5 day cumulative returns are measured
    # from the entry-day open (the "entry price"), not compounded from
    # the per-day path. day-1 leg is open-to-close; days 2..5 are
    # close-to-close.
    assert entry_bar is not None
    assert exit_1d_bar is not None
    assert exit_3d_bar is not None
    assert exit_5d_bar is not None
    entry_open = entry_bar.open
    exit_1d_return = exit_1d_bar.close / entry_open - 1.0
    exit_3d_return = exit_3d_bar.close / entry_open - 1.0
    exit_5d_return = exit_5d_bar.close / entry_open - 1.0

    return (
        ISDLabelRow(
            sample_id=sample_id,
            stock_id=stock_id,
            entry_date=entry_date,
            exit_1d_date=exit_1d_bar.trade_date,
            exit_3d_date=exit_3d_bar.trade_date,
            exit_5d_date=exit_5d_bar.trade_date,
            horizon_end_date=horizon_end_date,
            maturity=LabelMaturity.MATURE,
            exit_1d_return=exit_1d_return,
            exit_3d_return=exit_3d_return,
            exit_5d_return=exit_5d_return,
            cum_excess_1d=None,
            cum_excess_3d=None,
            cum_excess_5d=None,
            excess_unavailable_reason=excess_reason,
            entry_price=entry_open,
            exit_1d_price=exit_1d_bar.close,
            exit_3d_price=exit_3d_bar.close,
            exit_5d_price=exit_5d_bar.close,
            materialized_at=materialized_at,
        ),
    )


def _find_bar(bars: Sequence[ISDMarketBar], date: datetime) -> ISDMarketBar | None:
    for b in bars:
        if b.trade_date == date:
            return b
    return None


def _find_bar_n_after(
    bars: Sequence[ISDMarketBar],
    date: datetime,
    n: int,
) -> ISDMarketBar | None:
    """Return the n-th bar strictly at or after ``date`` (0-indexed)."""
    candidates = [b for b in bars if b.trade_date >= date]
    candidates.sort(key=lambda b: b.trade_date)
    if len(candidates) <= n:
        return None
    return candidates[n]


# ── Serialization helpers (used by builder) ────────────────────────


def _event_payload(ev: ISDEventRecord) -> dict[str, object]:
    return {
        "event_id": ev.event_id,
        "stock_id": ev.stock_id,
        "trade_date": ev.trade_date.isoformat(),
        "available_time": ev.available_time.isoformat(),
        "reason_codes": list(ev.reason_codes),
        "total_buy": ev.total_buy,
        "total_sell": ev.total_sell,
        "net_amount": ev.net_amount,
        "turnover_rate": ev.turnover_rate,
        "event_confidence": ev.event_confidence,
        "source_lineage": ev.source_lineage,
        "source_version": ev.source_version,
    }


def _participation_payload(p: ISDSeatParticipation) -> dict[str, object]:
    return {
        "participation_id": _participation_id(
            p.event_id, p.seat_id, p.side.value, p.rank_in_side
        ),
        "event_id": p.event_id,
        "stock_id": p.stock_id,
        "trade_date": p.trade_date.isoformat(),
        "seat_id": p.seat_id,
        "side": p.side.value,
        "amount": p.amount,
        "rank_in_side": p.rank_in_side,
        "identity_scope": p.identity_scope.value,
        "available_time": p.available_time.isoformat(),
        "source_lineage": p.source_lineage,
        "source_version": p.source_version,
    }


def _market_payload(b: ISDMarketBar) -> dict[str, object]:
    return {
        "stock_id": b.stock_id,
        "trade_date": b.trade_date.isoformat(),
        "open": b.open,
        "close": b.close,
        "high": b.high,
        "low": b.low,
        "volume": b.volume,
        "available_time": b.available_time.isoformat(),
        "source_lineage": b.source_lineage,
        "source_version": b.source_version,
    }


def _basic_payload(b: ISDDailyBasic) -> dict[str, object]:
    return {
        "stock_id": b.stock_id,
        "trade_date": b.trade_date.isoformat(),
        "turnover_rate": b.turnover_rate,
        "total_share": b.total_share,
        "float_share": b.float_share,
        "free_share": b.free_share,
        "total_mv": b.total_mv,
        "circ_mv": b.circ_mv,
        "available_time": b.available_time.isoformat(),
        "source_lineage": b.source_lineage,
        "source_version": b.source_version,
    }


def _label_payload(label: ISDLabelRow) -> dict[str, object]:
    return {
        "sample_id": label.sample_id,
        "stock_id": label.stock_id,
        "entry_date": label.entry_date.isoformat(),
        "exit_1d_date": label.exit_1d_date.isoformat(),
        "exit_3d_date": label.exit_3d_date.isoformat(),
        "exit_5d_date": label.exit_5d_date.isoformat(),
        "horizon_end_date": label.horizon_end_date.isoformat(),
        "maturity": label.maturity.value,
        "exit_1d_return": label.exit_1d_return,
        "exit_3d_return": label.exit_3d_return,
        "exit_5d_return": label.exit_5d_return,
        "cum_excess_1d": label.cum_excess_1d,
        "cum_excess_3d": label.cum_excess_3d,
        "cum_excess_5d": label.cum_excess_5d,
        "excess_unavailable_reason": label.excess_unavailable_reason,
        "entry_price": label.entry_price,
        "exit_1d_price": label.exit_1d_price,
        "exit_3d_price": label.exit_3d_price,
        "exit_5d_price": label.exit_5d_price,
        "materialized_at": label.materialized_at.isoformat(),
    }


__all__ = [
    "AdapterStats",
    "ISDInputs",
    "ISDLabelRow",
    "ISDMarketBar",
    "ISDDailyBasic",
    "ISDIndustryMembership",
    "ISDEventRecord",
    "ISDSeatParticipation",
    "IdentityScope",
    "LHBStockUniverse",
    "LabelMaturity",
    "SeatSide",
    "build_isd_inputs",
    "build_isd_labels",
    "build_lhb_stock_universe",
    "_normalize_reason",
    "_event_payload",
    "_participation_payload",
    "_market_payload",
    "_basic_payload",
    "_label_payload",
]

