"""Point-in-time record schema and future-blind eligibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")

POINT_IN_TIME_FIELDS: tuple[str, ...] = (
    "trade_date",
    "event_time",
    "source_publish_time",
    "source_available_time",
    "feature_generated_time",
    "prediction_generated_time",
    "earliest_trade_time",
)

MARKET_EXECUTION_FIELDS: tuple[str, ...] = (
    "is_suspended",
    "is_limit_up_at_entry",
    "is_limit_down_at_exit",
    "is_tradable_at_entry",
    "is_tradable_at_exit",
    "open_price",
    "close_price",
    "turnover_amount",
    "market_cap",
    "industry_id",
)


@dataclass(frozen=True)
class PointInTimeRecord:
    """Stock-date record carrying R4 point-in-time and execution fields.

    Legacy historical rows may lack precise intraday timestamps.  Such rows
    remain eligible for historical development analysis, but never for strict
    future-blind claims.
    """

    trade_date: date
    event_time: datetime | None
    source_publish_time: datetime | None
    source_available_time: datetime | None
    feature_generated_time: datetime | None
    prediction_generated_time: datetime | None
    earliest_trade_time: datetime | None
    is_suspended: bool
    is_limit_up_at_entry: bool
    is_limit_down_at_exit: bool
    is_tradable_at_entry: bool
    is_tradable_at_exit: bool
    open_price: float | None
    close_price: float | None
    turnover_amount: float | None
    market_cap: float | None
    industry_id: str | None
    future_blind_eligible: bool
    historical_development_eligible: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> PointInTimeRecord:
        """Build a record from explicit fields without fabricating timestamps."""

        _require_fields(row, POINT_IN_TIME_FIELDS + MARKET_EXECUTION_FIELDS)
        source_available_time = _optional_datetime(row["source_available_time"])
        prediction_generated_time = _optional_datetime(row["prediction_generated_time"])
        earliest_trade_time = _optional_datetime(row["earliest_trade_time"])
        future_blind_eligible = _is_future_blind_eligible(
            source_available_time=source_available_time,
            prediction_generated_time=prediction_generated_time,
            earliest_trade_time=earliest_trade_time,
        )
        return cls(
            trade_date=_date(row["trade_date"]),
            event_time=_optional_datetime(row["event_time"]),
            source_publish_time=_optional_datetime(row["source_publish_time"]),
            source_available_time=source_available_time,
            feature_generated_time=_optional_datetime(row["feature_generated_time"]),
            prediction_generated_time=prediction_generated_time,
            earliest_trade_time=earliest_trade_time,
            is_suspended=_bool(row["is_suspended"]),
            is_limit_up_at_entry=_bool(row["is_limit_up_at_entry"]),
            is_limit_down_at_exit=_bool(row["is_limit_down_at_exit"]),
            is_tradable_at_entry=_bool(row["is_tradable_at_entry"]),
            is_tradable_at_exit=_bool(row["is_tradable_at_exit"]),
            open_price=_optional_float(row["open_price"]),
            close_price=_optional_float(row["close_price"]),
            turnover_amount=_optional_float(row["turnover_amount"]),
            market_cap=_optional_float(row["market_cap"]),
            industry_id=_optional_str(row["industry_id"]),
            future_blind_eligible=future_blind_eligible,
            historical_development_eligible=True,
        )

    def validate(self) -> None:
        """Validate PIT ordering when precise timestamps are present."""

        if not self.future_blind_eligible:
            return
        if self.source_available_time is None:
            raise ValueError("source_available_time is required for future-blind rows")
        if self.prediction_generated_time is None:
            raise ValueError("prediction_generated_time is required for future-blind rows")
        if self.earliest_trade_time is None:
            raise ValueError("earliest_trade_time is required for future-blind rows")
        if self.source_available_time > self.prediction_generated_time:
            raise ValueError(
                "source_available_time must be <= prediction_generated_time"
            )
        if self.prediction_generated_time >= self.earliest_trade_time:
            raise ValueError("prediction_generated_time must be < earliest_trade_time")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_point_in_time_record(row: Mapping[str, object]) -> PointInTimeRecord:
    """Parse and validate a point-in-time record."""

    record = PointInTimeRecord.from_mapping(row)
    record.validate()
    return record


def _is_future_blind_eligible(
    *,
    source_available_time: datetime | None,
    prediction_generated_time: datetime | None,
    earliest_trade_time: datetime | None,
) -> bool:
    if (
        source_available_time is None
        or prediction_generated_time is None
        or earliest_trade_time is None
    ):
        return False
    if source_available_time > prediction_generated_time:
        raise ValueError("source_available_time must be <= prediction_generated_time")
    if prediction_generated_time >= earliest_trade_time:
        raise ValueError("prediction_generated_time must be < earliest_trade_time")
    return True


def _require_fields(row: Mapping[str, object], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"missing required PIT fields: {missing}")


def _date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("trade_date must be a date or ISO date string")


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamp fields must be timezone-aware")
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=SHANGHAI_TZ)
        return parsed
    raise TypeError("timestamp fields must be datetime, ISO datetime string, or None")


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("market execution boolean fields must be bool")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("numeric fields must not be bool")
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError("numeric fields must be int, float, or None")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
