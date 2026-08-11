"""V1 → V3 row conversion for SeatFlow-ISD.

Pure functions that convert the raw JSONL rows read by
:mod:`seatflow.ingestion.seatflow_isd.reader` into the frozen V3
contracts defined in :mod:`seatflow.ingestion.seatflow_isd.contracts`.

The adapter never re-validates hashes; the reader has already done
that. It only enforces the V3 row-level invariants (PIT, positive
prices, aware datetimes, etc.).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from seatflow.domain.time import assert_aware
from seatflow.ingestion.seatflow_isd.contracts import (
    ISDBenchmarkPath,
    ISDCycleV3,
    ISDDailyBasicV3,
    ISDEventV3,
    ISDLhbUniverseV3,
    ISDLabelRowV3,
    ISDLoadResult,
    ISDMaturity,
    ISDMarketBarV3,
    ISDSeatParticipationV3,
    ISDSampleV3,
)
from seatflow.ingestion.seatflow_isd.reader import load_isd_dataset


def _dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    assert_aware(dt, "datetime")
    return dt


def event_v3(row: Mapping[str, Any]) -> ISDEventV3:
    """Convert one V1 event row into the V3 contract."""
    return ISDEventV3(
        event_id=str(row["event_id"]),
        stock_id=str(row["stock_id"]),
        trade_date=_dt(row["trade_date"]),
        available_time=_dt(row["available_time"]),
        reason_codes=tuple(row.get("reason_codes", []) or []),
        total_buy=float(row.get("total_buy", 0.0) or 0.0),
        total_sell=float(row.get("total_sell", 0.0) or 0.0),
        net_amount=float(row.get("net_amount", 0.0) or 0.0),
        turnover_rate=float(row.get("turnover_rate", 0.0) or 0.0),
        event_confidence=float(row.get("event_confidence", 1.0) or 1.0),
        source_lineage=str(row.get("source_lineage", "tushare")),
        source_version=str(row.get("source_version", "top_list")),
    )


def participation_v3(row: Mapping[str, Any]) -> ISDSeatParticipationV3:
    """Convert one V1 seat participation row into the V3 contract.

    The ``event_id`` is taken directly from the V1 row — it was
    resolved against ``(stock_id, trade_date, normalized reason)``
    at build time and recorded on the row. Rows that don't resolve
    remain in the adapter statistics; they are never silently
    relabelled.
    """
    return ISDSeatParticipationV3(
        event_id=str(row.get("event_id", "")),
        stock_id=str(row.get("stock_id", "")),
        trade_date=_dt(row["trade_date"]),
        seat_id=str(row.get("seat_id", "")),
        side=str(row.get("side", "BUY")),
        amount=float(row.get("amount", 0.0) or 0.0),
        rank_in_side=int(row.get("rank_in_side", 0) or 0),
        identity_scope=str(row.get("identity_scope", "unknown")),
        available_time=_dt(row["available_time"]),
        source_lineage=str(row.get("source_lineage", "tushare")),
        source_version=str(row.get("source_version", "top_inst")),
    )


def market_bar_v3(row: Mapping[str, Any]) -> ISDMarketBarV3:
    return ISDMarketBarV3(
        stock_id=str(row["stock_id"]),
        trade_date=_dt(row["trade_date"]),
        open=float(row.get("open", 0.0) or 0.0),
        close=float(row.get("close", 0.0) or 0.0),
        high=float(row.get("high", 0.0) or 0.0),
        low=float(row.get("low", 0.0) or 0.0),
        volume=float(row.get("volume", 0.0) or 0.0),
        available_time=_dt(row["available_time"]),
        source_lineage=str(row.get("source_lineage", "tushare")),
        source_version=str(row.get("source_version", "tushare-pro-rest")),
    )


def daily_basic_v3(row: Mapping[str, Any]) -> ISDDailyBasicV3:
    return ISDDailyBasicV3(
        stock_id=str(row["stock_id"]),
        trade_date=_dt(row["trade_date"]),
        turnover_rate=float(row.get("turnover_rate", 0.0) or 0.0),
        total_share=float(row.get("total_share", 0.0) or 0.0),
        float_share=float(row.get("float_share", 0.0) or 0.0),
        free_share=float(row.get("free_share", 0.0) or 0.0),
        total_mv=float(row.get("total_mv", 0.0) or 0.0),
        circ_mv=float(row.get("circ_mv", 0.0) or 0.0),
        available_time=_dt(row["available_time"]),
        source_lineage=str(row.get("source_lineage", "tushare")),
        source_version=str(row.get("source_version", "daily_basic")),
    )


def cycle_v3(row: Mapping[str, Any]) -> ISDCycleV3:
    cycle_end_raw = row.get("cycle_end")
    cycle_end = _dt(cycle_end_raw) if cycle_end_raw else None
    return ISDCycleV3(
        cycle_id=str(row["cycle_id"]),
        stock_id=str(row["stock_id"]),
        cycle_start=_dt(row["cycle_start"]),
        last_event_date=_dt(row["last_event_date"]),
        cycle_end=cycle_end,
        event_count=int(row.get("event_count", 0) or 0),
        status=str(row.get("status", "EXPIRED")),
    )


def sample_v3(row: Mapping[str, Any]) -> ISDSampleV3:
    return ISDSampleV3(
        sample_id=str(row["sample_id"]),
        cycle_id=str(row["cycle_id"]),
        stock_id=str(row["stock_id"]),
        prediction_date=_dt(row["prediction_date"]),
        cutoff_time=_dt(row["cutoff_time"]),
        days_since_last_event=int(row.get("days_since_last_event", 0) or 0),
        pool_type=str(row.get("pool_type", "DIRECT")),
        tradeability=str(row.get("tradeability", "TRADABLE")),
        available_time=_dt(row["available_time"]),
        is_tradable=bool(row.get("is_tradable", False)),
        is_backtest_eligible=bool(row.get("is_backtest_eligible", False)),
    )


def label_v3(row: Mapping[str, Any]) -> ISDLabelRowV3:
    return ISDLabelRowV3(
        sample_id=str(row["sample_id"]),
        stock_id=str(row["stock_id"]),
        entry_date=_dt(row["entry_date"]),
        exit_1d_date=_dt(row["exit_1d_date"]),
        exit_3d_date=_dt(row["exit_3d_date"]),
        exit_5d_date=_dt(row["exit_5d_date"]),
        horizon_end_date=_dt(row["horizon_end_date"]),
        maturity=ISDMaturity(str(row.get("maturity", "UNMATURE"))),
        exit_1d_return=_opt_float(row.get("exit_1d_return")),
        exit_3d_return=_opt_float(row.get("exit_3d_return")),
        exit_5d_return=_opt_float(row.get("exit_5d_return")),
        cum_excess_1d=_opt_float(row.get("cum_excess_1d")),
        cum_excess_3d=_opt_float(row.get("cum_excess_3d")),
        cum_excess_5d=_opt_float(row.get("cum_excess_5d")),
        excess_unavailable_reason=row.get("excess_unavailable_reason"),
        entry_price=_opt_float(row.get("entry_price")),
        exit_1d_price=_opt_float(row.get("exit_1d_price")),
        exit_3d_price=_opt_float(row.get("exit_3d_price")),
        exit_5d_price=_opt_float(row.get("exit_5d_price")),
        materialized_at=_dt(row["materialized_at"]),
    )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def universe_v3(row: Mapping[str, Any]) -> ISDLhbUniverseV3:
    stock_ids = frozenset(row.get("stock_ids", []) or [])
    first_seen_raw = row.get("first_seen_by_stock", {}) or {}
    first_seen: dict[str, datetime] = {}
    for stock_id, when in first_seen_raw.items():
        try:
            first_seen[stock_id] = datetime.fromisoformat(when)
        except (ValueError, TypeError):
            continue
    return ISDLhbUniverseV3(
        stock_ids=stock_ids,
        first_seen_by_stock=dict(sorted(first_seen.items())),
        size=int(row.get("size", len(stock_ids))),
    )


def to_load_result(
    raw: Mapping[str, Any],
    benchmarks: tuple[ISDBenchmarkPath, ...] = (),
) -> ISDLoadResult:
    """Convert the raw reader output into a frozen V3 ``ISDLoadResult``."""
    events = tuple(event_v3(r) for r in raw["events"])
    participations = tuple(participation_v3(r) for r in raw["participations"])
    market_bars = tuple(market_bar_v3(r) for r in raw["market_bars"])
    daily_basic = tuple(daily_basic_v3(r) for r in raw["daily_basic"])
    cycles = tuple(cycle_v3(r) for r in raw["cycles"])
    samples = tuple(sample_v3(r) for r in raw["samples"])
    labels = tuple(label_v3(r) for r in raw["labels"])
    universe = universe_v3(raw["universe"])
    qr = raw.get("quality_report", {}) or {}
    date_range = qr.get("date_range", {}) or {}
    row_counts = qr.get("row_counts", {}) or {}
    source_inventory_fingerprint = str(qr.get("source_inventory_fingerprint", ""))
    return ISDLoadResult(
        dataset_version=str(raw["dataset_version"]),
        dataset_root=str(raw["dataset_root"]),
        manifest_hash=str(raw["manifest_hash"]),
        manifest_hash_verified=bool(raw.get("manifest_verified", False)),
        hash_mismatches=(),
        events=events,
        participations=participations,
        market_bars=market_bars,
        daily_basic=daily_basic,
        cycles=cycles,
        samples=samples,
        labels=labels,
        universe=universe,
        date_range=dict(date_range),
        row_counts=dict(row_counts),
        source_inventory_fingerprint=source_inventory_fingerprint,
        benchmarks=benchmarks,
    )


__all__ = [
    "cycle_v3",
    "daily_basic_v3",
    "event_v3",
    "label_v3",
    "market_bar_v3",
    "participation_v3",
    "sample_v3",
    "to_load_result",
    "universe_v3",
    "load_isd_dataset",
]
