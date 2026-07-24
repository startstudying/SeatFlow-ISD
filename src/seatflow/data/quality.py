from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from seatflow.data.contracts import DailyBar, DragonTigerSummary
from seatflow.data.records import DragonTigerRecord


@dataclass(frozen=True)
class DataQualityReport:
    dataset: str
    row_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "row_count": self.row_count,
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_daily_bars(bars: Iterable[DailyBar]) -> DataQualityReport:
    materialized = tuple(bars)
    seen: set[tuple[str, object]] = set()
    errors: list[str] = []
    warnings: list[str] = []
    for bar in materialized:
        key = (bar.stock_id, bar.trade_date)
        if key in seen:
            errors.append(f"duplicate daily bar: {bar.stock_id} {bar.trade_date.isoformat()}")
        seen.add(key)
        if bar.available_time is None:
            warnings.append(f"missing available_time: {bar.stock_id} {bar.trade_date.isoformat()}")
    return DataQualityReport("daily_bar", len(materialized), tuple(errors), tuple(warnings))


def validate_dragon_tiger_summaries(
    summaries: Iterable[DragonTigerSummary],
) -> DataQualityReport:
    materialized = tuple(summaries)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, object, str]] = set()
    for item in materialized:
        key = (item.stock_id, item.trade_date, item.reason)
        if key in seen:
            warnings.append(
                f"repeated dragon-tiger summary: {item.stock_id} {item.trade_date.isoformat()}"
            )
        seen.add(key)
        if item.available_time is None:
            warnings.append(
                f"missing available_time: {item.stock_id} {item.trade_date.isoformat()}"
            )
    return DataQualityReport(
        "dragon_tiger_summary", len(materialized), tuple(errors), tuple(warnings)
    )


def validate_dragon_tiger_records(
    records: Iterable[DragonTigerRecord], cutoff_time: datetime | None = None
) -> DataQualityReport:
    materialized = tuple(records)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str, object, str, str | None]] = set()
    for item in materialized:
        key = (
            item.identity_scope,
            item.seat_id,
            item.stock_id,
            item.trade_date,
            item.side.value,
            item.event_reason,
        )
        if key in seen:
            errors.append(
                "duplicate canonical seat event: "
                f"{item.seat_id} {item.stock_id} {item.trade_date.isoformat()} {item.side.value}"
            )
        seen.add(key)
        if item.available_time is None:
            warnings.append(
                f"missing available_time: {item.stock_id} {item.trade_date.isoformat()}"
            )
        elif cutoff_time is not None and item.available_time > cutoff_time:
            errors.append(
                f"record unavailable at cutoff: {item.stock_id} {item.trade_date.isoformat()}"
            )
    return DataQualityReport(
        "dragon_tiger_seat_event", len(materialized), tuple(errors), tuple(warnings)
    )
