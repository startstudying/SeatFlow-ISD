from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from seatflow.data.contracts import normalize_stock_id


class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


NAMED_SEAT_IDENTITY_SCOPE = "named_seat"
INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE = "institutional_flow_proxy"
_IDENTITY_SCOPES = frozenset(
    {NAMED_SEAT_IDENTITY_SCOPE, INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE}
)


@dataclass(frozen=True)
class DragonTigerRecord:
    trade_date: date
    seat_id: str
    stock_id: str
    side: TradeSide
    amount: float
    industry: str | None = None
    concepts: tuple[str, ...] = ()
    event_reason: str | None = None
    available_time: datetime | None = None
    source_name: str = "unknown"
    source_version: str | None = None
    source_row_count: int = 1
    source_side_set: tuple[str, ...] = ()
    identity_scope: str = NAMED_SEAT_IDENTITY_SCOPE

    def __post_init__(self) -> None:
        if not self.seat_id.strip():
            raise ValueError("seat_id must not be empty")
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if self.amount <= 0:
            raise ValueError("amount must be greater than zero")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if self.source_row_count < 1:
            raise ValueError("source_row_count must be at least one")
        normalized_side_set = tuple(sorted({item.strip() for item in self.source_side_set}))
        if any(not item for item in normalized_side_set):
            raise ValueError("source_side_set must not contain empty values")
        if self.identity_scope not in _IDENTITY_SCOPES:
            raise ValueError("identity_scope is invalid")
        object.__setattr__(self, "source_side_set", normalized_side_set)

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "seat_id": self.seat_id,
            "stock_id": self.stock_id,
            "side": self.side.value,
            "amount": self.amount,
            "industry": self.industry,
            "concepts": list(self.concepts),
            "event_reason": self.event_reason,
            "available_time": self.available_time.isoformat() if self.available_time else None,
            "source_name": self.source_name,
            "source_version": self.source_version,
            "source_row_count": self.source_row_count,
            "source_side_set": list(self.source_side_set),
            "identity_scope": self.identity_scope,
        }
