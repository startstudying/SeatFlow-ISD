from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from seatflow.data.contracts import normalize_stock_id

OrderSide = Literal["buy", "sell"]
FillStatus = Literal["filled", "partial", "rejected"]


@dataclass(frozen=True)
class PaperTradingConfig:
    account_id: str = "seatflow-paper"
    initial_cash: Decimal = Decimal("1000000")
    top_n: int = 5
    minimum_score: Decimal = Decimal("0.5")
    cash_buffer_pct: Decimal = Decimal("0.05")
    max_position_pct: Decimal = Decimal("0.20")
    commission_bps: Decimal = Decimal("3")
    minimum_commission: Decimal = Decimal("5")
    stamp_tax_bps: Decimal = Decimal("5")
    slippage_bps: Decimal = Decimal("5")
    minimum_lot_size: int = 100

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("account_id must not be empty")
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.minimum_lot_size <= 0:
            raise ValueError("minimum_lot_size must be positive")
        if not self.minimum_score.is_finite():
            raise ValueError("minimum_score must be finite")
        if not self.cash_buffer_pct.is_finite() or not (
            Decimal("0") <= self.cash_buffer_pct < Decimal("1")
        ):
            raise ValueError("cash_buffer_pct must be in [0, 1)")
        if not self.max_position_pct.is_finite() or not (
            Decimal("0") < self.max_position_pct <= Decimal("1")
        ):
            raise ValueError("max_position_pct must be in (0, 1]")
        for field_name in (
            "commission_bps",
            "minimum_commission",
            "stamp_tax_bps",
            "slippage_bps",
        ):
            value = getattr(self, field_name)
            if not value.is_finite() or value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "initial_cash": str(self.initial_cash),
            "top_n": self.top_n,
            "minimum_score": str(self.minimum_score),
            "cash_buffer_pct": str(self.cash_buffer_pct),
            "max_position_pct": str(self.max_position_pct),
            "commission_bps": str(self.commission_bps),
            "minimum_commission": str(self.minimum_commission),
            "stamp_tax_bps": str(self.stamp_tax_bps),
            "slippage_bps": str(self.slippage_bps),
            "minimum_lot_size": self.minimum_lot_size,
        }


@dataclass(frozen=True)
class PaperSignal:
    stock_id: str
    score: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if not self.score.is_finite():
            raise ValueError("signal score must be finite")

    def to_dict(self) -> dict[str, object]:
        return {"stock_id": self.stock_id, "score": str(self.score)}


@dataclass(frozen=True)
class SignalBatch:
    as_of_date: date
    generated_at: datetime
    signals: tuple[PaperSignal, ...]

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must include a timezone offset")
        if self.generated_at.date() != self.as_of_date:
            raise ValueError("generated_at date must match as_of_date")
        stock_ids = [signal.stock_id for signal in self.signals]
        if len(stock_ids) != len(set(stock_ids)):
            raise ValueError("signal batch contains duplicate stock_id values")

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "signals": [signal.to_dict() for signal in self.signals],
        }


@dataclass(frozen=True)
class MarketQuote:
    stock_id: str
    price: Decimal
    tradable: bool = True
    limit_up: bool = False
    limit_down: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if not self.price.is_finite() or self.price <= 0:
            raise ValueError("quote price must be finite and positive")
        if self.limit_up and self.limit_down:
            raise ValueError("quote cannot be both limit_up and limit_down")

    def to_dict(self) -> dict[str, object]:
        return {
            "stock_id": self.stock_id,
            "price": str(self.price),
            "tradable": self.tradable,
            "limit_up": self.limit_up,
            "limit_down": self.limit_down,
        }


@dataclass(frozen=True)
class QuoteBatch:
    trade_date: date
    captured_at: datetime
    quotes: tuple[MarketQuote, ...]

    def __post_init__(self) -> None:
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must include a timezone offset")
        if self.captured_at.date() != self.trade_date:
            raise ValueError("captured_at date must match trade_date")
        stock_ids = [quote.stock_id for quote in self.quotes]
        if len(stock_ids) != len(set(stock_ids)):
            raise ValueError("quote batch contains duplicate stock_id values")

    def to_dict(self) -> dict[str, object]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "captured_at": self.captured_at.isoformat(),
            "quotes": [quote.to_dict() for quote in self.quotes],
        }


@dataclass(frozen=True)
class PaperOrder:
    stock_id: str
    side: OrderSide
    quantity: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))
        if self.side not in ("buy", "sell"):
            raise ValueError("order side must be buy or sell")
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if not self.reason.strip():
            raise ValueError("order reason must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "stock_id": self.stock_id,
            "side": self.side,
            "quantity": self.quantity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PaperFill:
    order_id: str
    order: PaperOrder
    status: FillStatus
    filled_quantity: int
    price: Decimal
    commission: Decimal
    stamp_tax: Decimal
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "order": self.order.to_dict(),
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "price": str(self.price),
            "commission": str(self.commission),
            "stamp_tax": str(self.stamp_tax),
            "message": self.message,
        }


@dataclass(frozen=True)
class PositionSnapshot:
    stock_id: str
    quantity: int
    sellable_quantity: int
    average_cost: Decimal
    mark_price: Decimal
    market_value: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "stock_id": self.stock_id,
            "quantity": self.quantity,
            "sellable_quantity": self.sellable_quantity,
            "average_cost": str(self.average_cost),
            "mark_price": str(self.mark_price),
            "market_value": str(self.market_value),
        }


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    cash: Decimal
    market_value: Decimal
    total_equity: Decimal
    positions: tuple[PositionSnapshot, ...]
    unpriced_positions: tuple[str, ...] = ()

    def position_map(self) -> dict[str, PositionSnapshot]:
        return {position.stock_id: position for position in self.positions}

    def to_dict(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "cash": str(self.cash),
            "market_value": str(self.market_value),
            "total_equity": str(self.total_equity),
            "positions": [position.to_dict() for position in self.positions],
            "unpriced_positions": list(self.unpriced_positions),
        }


@dataclass(frozen=True)
class OrderBatchResult:
    batch_id: str
    trade_date: date
    request_hash: str
    replayed: bool
    fills: tuple[PaperFill, ...]
    account: AccountSnapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "trade_date": self.trade_date.isoformat(),
            "request_hash": self.request_hash,
            "replayed": self.replayed,
            "fills": [fill.to_dict() for fill in self.fills],
            "account": self.account.to_dict(),
        }


@dataclass(frozen=True)
class PaperTradingReport:
    cycle_id: str
    cycle_request_hash: str
    trade_date: date
    selected_signals: tuple[PaperSignal, ...]
    account_before: AccountSnapshot
    sell_batch: OrderBatchResult
    buy_batch: OrderBatchResult
    account_after: AccountSnapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle_id": self.cycle_id,
            "cycle_request_hash": self.cycle_request_hash,
            "trade_date": self.trade_date.isoformat(),
            "selected_signals": [signal.to_dict() for signal in self.selected_signals],
            "account_before": self.account_before.to_dict(),
            "sell_batch": self.sell_batch.to_dict(),
            "buy_batch": self.buy_batch.to_dict(),
            "account_after": self.account_after.to_dict(),
        }
