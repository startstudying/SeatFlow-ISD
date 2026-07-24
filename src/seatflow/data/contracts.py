from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any

_STOCK_ID_PATTERN = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ|BJ)$")
_PREFIX_STOCK_ID_PATTERN = re.compile(r"^(?P<exchange>SH|SZ|BJ)\.(?P<code>\d{6})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUSPENSION_TYPES = frozenset({"S", "R"})


def parse_trade_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    normalized = str(value).strip()
    if len(normalized) == 8 and normalized.isdigit():
        return datetime.strptime(normalized, "%Y%m%d").date()
    return date.fromisoformat(normalized)


def parse_decimal(value: object, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def parse_optional_decimal(value: object | None, field: str) -> Decimal | None:
    if value is None or str(value).strip() in {"", "None", "nan", "NaN"}:
        return None
    return parse_decimal(value, field)


def parse_optional_datetime(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def normalize_stock_id(value: object) -> str:
    """Return one canonical A-share identifier such as 600000.SH."""

    raw = str(value).strip().upper()
    suffix_match = _STOCK_ID_PATTERN.fullmatch(raw)
    if suffix_match:
        code = suffix_match.group("code")
        exchange = suffix_match.group("exchange")
    else:
        prefix_match = _PREFIX_STOCK_ID_PATTERN.fullmatch(raw)
        if prefix_match:
            code = prefix_match.group("code")
            exchange = prefix_match.group("exchange")
        elif len(raw) == 6 and raw.isdigit():
            code = raw
            exchange = _infer_exchange(code)
        else:
            raise ValueError(f"invalid stock identifier: {value!r}")

    inferred = _infer_exchange(code)
    if exchange != inferred:
        raise ValueError(f"stock identifier exchange does not match code: {value!r}")
    return f"{code}.{exchange}"


def _infer_exchange(code: str) -> str:
    if code.startswith("6") or code.startswith("900"):
        return "SH"
    if code.startswith(("4", "8", "92")):
        return "BJ"
    if code.startswith(("0", "2", "3")):
        return "SZ"
    raise ValueError(f"cannot infer exchange from stock code: {code!r}")


def parse_suspension_timing(value: object) -> str | None:
    """Preserve Tushare's nullable/empty timing field without inventing a time."""

    if value is None or isinstance(value, str):
        return value
    raise ValueError("suspend_timing must be a string or null")


def canonical_suspension_row_payload(row: Mapping[str, object]) -> dict[str, str | None]:
    """Normalize the documented ``suspend_d`` fields before hashing their evidence."""

    try:
        raw_stock_id = row["ts_code"]
        raw_suspend_type = row["suspend_type"]
        raw_suspend_timing = row["suspend_timing"]
        if not isinstance(raw_stock_id, str):
            raise ValueError("ts_code must be a string")
        if not isinstance(raw_suspend_type, str):
            raise ValueError("suspend_type must be a string")
        stock_id = normalize_stock_id(raw_stock_id)
        trade_date = parse_trade_date(row["trade_date"])
        suspend_type = raw_suspend_type.strip().upper()
        suspend_timing = parse_suspension_timing(raw_suspend_timing)
    except KeyError as error:
        raise ValueError(f"suspension row is missing {error.args[0]}") from error
    if suspend_type not in _SUSPENSION_TYPES:
        raise ValueError("suspend_type must be one of: R, S")
    return {
        "stock_id": stock_id,
        "suspend_timing": suspend_timing,
        "suspend_type": suspend_type,
        "trade_date": trade_date.isoformat(),
    }


def suspension_row_payload_sha256(row: Mapping[str, object]) -> str:
    """Hash a normalized suspension row independently of Tushare field ordering."""

    payload = json.dumps(
        canonical_suspension_row_payload(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def suspension_daily_query_sha256(
    trade_date: date, rows: Iterable[Mapping[str, object]]
) -> str:
    """Hash a full ``suspend_d`` daily response after normalizing and sorting its rows."""

    if isinstance(trade_date, datetime) or not isinstance(trade_date, date):
        raise ValueError("trade_date must be a date")
    canonical_rows = sorted(
        (canonical_suspension_row_payload(row) for row in rows),
        key=_canonical_suspension_row_sort_key,
    )
    payload = json.dumps(
        {"trade_date": trade_date.isoformat(), "rows": canonical_rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _canonical_suspension_row_sort_key(
    row: Mapping[str, str | None],
) -> tuple[str, str, str, int, str]:
    timing = row["suspend_timing"]
    return (
        str(row["stock_id"]),
        str(row["trade_date"]),
        str(row["suspend_type"]),
        0 if timing is None else 1,
        timing or "",
    )


@dataclass(frozen=True)
class SuspensionRecord:
    """Immutable evidence for one documented Tushare ``suspend_d`` row."""

    stock_id: str
    trade_date: date
    suspend_type: str
    suspend_timing: str | None
    source: str
    retrieved_at: datetime
    row_payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.stock_id, str):
            raise ValueError("stock_id must be a string")
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if isinstance(self.trade_date, datetime) or not isinstance(self.trade_date, date):
            raise ValueError("trade_date must be a date")
        if not isinstance(self.suspend_type, str):
            raise ValueError("suspend_type must be a string")
        suspend_type = self.suspend_type.strip().upper()
        object.__setattr__(self, "suspend_type", suspend_type)
        if suspend_type not in _SUSPENSION_TYPES:
            raise ValueError("suspend_type must be one of: R, S")
        suspend_timing = parse_suspension_timing(self.suspend_timing)
        object.__setattr__(self, "suspend_timing", suspend_timing)
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")
        source = self.source.strip()
        object.__setattr__(self, "source", source)
        if not source:
            raise ValueError("source must not be empty")
        if not isinstance(self.retrieved_at, datetime) or self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be a timezone-aware datetime")
        if not isinstance(self.row_payload_sha256, str):
            raise ValueError("row_payload_sha256 must be a string")
        payload_sha256 = self.row_payload_sha256.strip().lower()
        object.__setattr__(self, "row_payload_sha256", payload_sha256)
        if not _SHA256_PATTERN.fullmatch(payload_sha256):
            raise ValueError("row_payload_sha256 must be a lowercase SHA-256 hex digest")

    @property
    def canonical_key(self) -> tuple[str, date, str, str | None]:
        return (self.stock_id, self.trade_date, self.suspend_type, self.suspend_timing)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "stock_id": self.stock_id,
            "trade_date": self.trade_date.isoformat(),
            "suspend_type": self.suspend_type,
            "suspend_timing": self.suspend_timing,
            "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
            "row_payload_sha256": self.row_payload_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SuspensionRecord:
        try:
            retrieved_at = datetime.fromisoformat(str(raw["retrieved_at"]))
            return cls(
                stock_id=str(raw["stock_id"]),
                trade_date=parse_trade_date(raw["trade_date"]),
                suspend_type=str(raw["suspend_type"]),
                suspend_timing=parse_suspension_timing(raw["suspend_timing"]),
                source=str(raw["source"]),
                retrieved_at=retrieved_at,
                row_payload_sha256=str(raw["row_payload_sha256"]),
            )
        except KeyError as error:
            raise ValueError(f"suspension record is missing {error.args[0]}") from error


@dataclass(frozen=True)
class SuspensionResponseEvidence:
    """A trade-date query and its fail-closed set of ``S`` suspension records."""

    trade_date: date
    source: str
    retrieved_at: datetime
    source_row_count: int
    daily_query_sha256: str
    records: tuple[SuspensionRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.trade_date, datetime) or not isinstance(self.trade_date, date):
            raise ValueError("trade_date must be a date")
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")
        source = self.source.strip()
        object.__setattr__(self, "source", source)
        if not source:
            raise ValueError("source must not be empty")
        if not isinstance(self.retrieved_at, datetime) or self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be a timezone-aware datetime")
        if self.source_row_count < len(self.records):
            raise ValueError("source_row_count must cover all suspension records")
        if not isinstance(self.daily_query_sha256, str):
            raise ValueError("daily_query_sha256 must be a string")
        daily_query_sha256 = self.daily_query_sha256.strip().lower()
        object.__setattr__(self, "daily_query_sha256", daily_query_sha256)
        if not _SHA256_PATTERN.fullmatch(daily_query_sha256):
            raise ValueError("daily_query_sha256 must be a lowercase SHA-256 hex digest")
        seen: set[tuple[str, date, str, str | None]] = set()
        for record in self.records:
            if record.trade_date != self.trade_date:
                raise ValueError("suspension record date does not match queried trade_date")
            if record.source != source or record.retrieved_at != self.retrieved_at:
                raise ValueError("suspension record provenance does not match response evidence")
            if record.suspend_type != "S":
                raise ValueError("suspension response evidence may only contain S records")
            if record.canonical_key in seen:
                raise ValueError(f"duplicate canonical suspension record: {record.canonical_key}")
            seen.add(record.canonical_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
            "source_row_count": self.source_row_count,
            "daily_query_sha256": self.daily_query_sha256,
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
        }


def serialize_suspension_evidence_jsonl(evidence: SuspensionResponseEvidence) -> str:
    """Return one canonical JSONL line for an upper-layer evidence writer."""

    serialized = json.dumps(
        evidence.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return serialized + "\n"


@dataclass(frozen=True)
class DailyBar:
    trade_date: date
    stock_id: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    amount: Decimal
    source_name: str
    available_time: datetime | None = None
    pre_close: Decimal | None = None
    source_version: str | None = None
    volume_unit: str = "unknown"
    amount_unit: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if min(self.open_price, self.high_price, self.low_price, self.close_price) < 0:
            raise ValueError("daily prices must be non-negative")
        if self.high_price < max(self.open_price, self.low_price, self.close_price):
            raise ValueError("high_price must be at least open, low, and close")
        if self.low_price > min(self.open_price, self.high_price, self.close_price):
            raise ValueError("low_price must be at most open, high, and close")
        if self.volume < 0 or self.amount < 0:
            raise ValueError("volume and amount must be non-negative")
        if not self.volume_unit.strip() or not self.amount_unit.strip():
            raise ValueError("volume_unit and amount_unit must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "stock_id": self.stock_id,
            "open_price": str(self.open_price),
            "high_price": str(self.high_price),
            "low_price": str(self.low_price),
            "close_price": str(self.close_price),
            "volume": str(self.volume),
            "amount": str(self.amount),
            "source_name": self.source_name,
            "available_time": self.available_time.isoformat() if self.available_time else None,
            "pre_close": str(self.pre_close) if self.pre_close is not None else None,
            "source_version": self.source_version,
            "volume_unit": self.volume_unit,
            "amount_unit": self.amount_unit,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DailyBar:
        return cls(
            trade_date=parse_trade_date(raw["trade_date"]),
            stock_id=str(raw["stock_id"]),
            open_price=parse_decimal(raw["open_price"], "open_price"),
            high_price=parse_decimal(raw["high_price"], "high_price"),
            low_price=parse_decimal(raw["low_price"], "low_price"),
            close_price=parse_decimal(raw["close_price"], "close_price"),
            volume=parse_decimal(raw["volume"], "volume"),
            amount=parse_decimal(raw["amount"], "amount"),
            source_name=str(raw["source_name"]),
            available_time=parse_optional_datetime(raw.get("available_time")),
            pre_close=parse_optional_decimal(raw.get("pre_close"), "pre_close"),
            source_version=raw.get("source_version"),
            volume_unit=str(raw.get("volume_unit") or "unknown"),
            amount_unit=str(raw.get("amount_unit") or "unknown"),
        )


@dataclass(frozen=True)
class AdjustmentFactorEvidence:
    """One source-hashed adjustment factor required by executable label V2."""

    stock_id: str
    trade_date: date
    adjustment_factor: Decimal
    source: str
    retrieved_at: datetime
    row_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if self.adjustment_factor <= 0:
            raise ValueError("adjustment_factor must be positive")
        if not self.source.strip():
            raise ValueError("adjustment factor source must not be empty")
        if not isinstance(self.retrieved_at, datetime) or self.retrieved_at.tzinfo is None:
            raise ValueError("adjustment factor retrieved_at must be timezone-aware")
        if not _SHA256_PATTERN.fullmatch(self.row_payload_sha256):
            raise ValueError("adjustment factor row hash must be lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "stock_id": self.stock_id,
            "trade_date": self.trade_date.isoformat(),
            "adjustment_factor": str(self.adjustment_factor),
            "source": self.source,
            "retrieved_at": self.retrieved_at.isoformat(),
            "row_payload_sha256": self.row_payload_sha256,
        }


@dataclass(frozen=True)
class AdjustmentFactorResponseEvidence:
    """Immutable query-level evidence for a bounded adjustment-factor response."""

    stock_id: str
    start_date: date
    end_date: date
    source: str
    retrieved_at: datetime
    query_sha256: str
    records: tuple[AdjustmentFactorEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if self.end_date < self.start_date:
            raise ValueError("adjustment factor end_date must not precede start_date")
        if not _SHA256_PATTERN.fullmatch(self.query_sha256):
            raise ValueError("adjustment factor query hash must be lowercase SHA-256")
        if any(record.stock_id != self.stock_id for record in self.records):
            raise ValueError("adjustment factor response records must match stock_id")


@dataclass(frozen=True)
class DragonTigerSummary:
    trade_date: date
    stock_id: str
    stock_name: str
    buy_amount: Decimal
    sell_amount: Decimal
    net_amount: Decimal
    reason: str
    source_name: str
    available_time: datetime | None = None
    source_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if min(self.buy_amount, self.sell_amount) < 0:
            raise ValueError("buy_amount and sell_amount must be non-negative")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "stock_id": self.stock_id,
            "stock_name": self.stock_name,
            "buy_amount": str(self.buy_amount),
            "sell_amount": str(self.sell_amount),
            "net_amount": str(self.net_amount),
            "reason": self.reason,
            "source_name": self.source_name,
            "available_time": self.available_time.isoformat() if self.available_time else None,
            "source_version": self.source_version,
        }


@dataclass(frozen=True)
class StockFeature:
    trade_date: date
    stock_id: str
    close_price: Decimal
    return_1d: Decimal | None
    ma_5: Decimal | None
    ma_20: Decimal | None
    volatility_5: Decimal | None
    source_name: str
    available_time: datetime | None

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "stock_id": self.stock_id,
            "close_price": str(self.close_price),
            "return_1d": str(self.return_1d) if self.return_1d is not None else None,
            "ma_5": str(self.ma_5) if self.ma_5 is not None else None,
            "ma_20": str(self.ma_20) if self.ma_20 is not None else None,
            "volatility_5": str(self.volatility_5) if self.volatility_5 is not None else None,
            "source_name": self.source_name,
            "available_time": self.available_time.isoformat() if self.available_time else None,
        }
