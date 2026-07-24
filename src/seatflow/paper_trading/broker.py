from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast

from seatflow.data.contracts import normalize_stock_id, parse_decimal
from seatflow.paper_trading.models import (
    AccountSnapshot,
    FillStatus,
    MarketQuote,
    OrderBatchResult,
    PaperFill,
    PaperOrder,
    PaperTradingConfig,
    PositionSnapshot,
)


class PaperBrokerError(RuntimeError):
    """Raised when the local paper account cannot be read or updated safely."""


class PaperBroker(Protocol):
    def snapshot(
        self, trade_date: date, quotes: Mapping[str, MarketQuote]
    ) -> AccountSnapshot: ...

    def submit_market_orders(
        self,
        *,
        batch_id: str,
        trade_date: date,
        orders: Sequence[PaperOrder],
        quotes: Mapping[str, MarketQuote],
        request_context: Mapping[str, object] | None = None,
    ) -> OrderBatchResult: ...


class JsonPaperBroker:
    def __init__(self, state_path: str | Path, config: PaperTradingConfig) -> None:
        self.state_path = Path(state_path)
        self.config = config

    def initialize(self) -> AccountSnapshot:
        with _StateLock(self.state_path):
            if self.state_path.exists():
                raise PaperBrokerError(f"paper account already exists: {self.state_path}")
            state = self._new_state()
            self._write_state(state)
            return self._snapshot_from_state(state, date.min, {})

    def snapshot(
        self, trade_date: date, quotes: Mapping[str, MarketQuote]
    ) -> AccountSnapshot:
        return self._snapshot_from_state(self._read_state(), trade_date, quotes)

    def submit_market_orders(
        self,
        *,
        batch_id: str,
        trade_date: date,
        orders: Sequence[PaperOrder],
        quotes: Mapping[str, MarketQuote],
        request_context: Mapping[str, object] | None = None,
    ) -> OrderBatchResult:
        if not batch_id.strip():
            raise ValueError("batch_id must not be empty")
        request_hash = _batch_request_hash(trade_date, orders, quotes, request_context)
        with _StateLock(self.state_path):
            state = self._read_state()
            batches = _dict_value(state, "batches")
            prior = batches.get(batch_id)
            if isinstance(prior, dict):
                result = self._batch_result_from_dict(prior)
                if result.request_hash != request_hash:
                    raise PaperBrokerError(
                        f"batch_id was reused with a different request: {batch_id}"
                    )
                return OrderBatchResult(
                    batch_id=result.batch_id,
                    trade_date=result.trade_date,
                    request_hash=result.request_hash,
                    replayed=True,
                    fills=result.fills,
                    account=result.account,
                )
            last_trade_date = _optional_date_value(state, "last_trade_date")
            if last_trade_date is not None and trade_date < last_trade_date:
                raise PaperBrokerError(
                    "new paper-trading batches cannot execute before the last trade date"
                )

            fills: list[PaperFill] = []
            for index, order in enumerate(orders):
                fills.append(
                    self._execute_order(
                        state=state,
                        order_id=_order_id(batch_id, index, order),
                        trade_date=trade_date,
                        order=order,
                        quote=quotes.get(order.stock_id),
                    )
                )
            account = self._snapshot_from_state(state, trade_date, quotes)
            result = OrderBatchResult(
                batch_id=batch_id,
                trade_date=trade_date,
                request_hash=request_hash,
                replayed=False,
                fills=tuple(fills),
                account=account,
            )
            batches[batch_id] = result.to_dict()
            state["last_trade_date"] = trade_date.isoformat()
            self._write_state(state)
            return result

    def _execute_order(
        self,
        *,
        state: dict[str, object],
        order_id: str,
        trade_date: date,
        order: PaperOrder,
        quote: MarketQuote | None,
    ) -> PaperFill:
        zero = Decimal("0")
        if order.quantity % self.config.minimum_lot_size != 0:
            return PaperFill(
                order_id,
                order,
                "rejected",
                0,
                zero,
                zero,
                zero,
                "quantity_not_multiple_of_lot_size",
            )
        if quote is None:
            return PaperFill(
                order_id, order, "rejected", 0, zero, zero, zero, "quote_missing"
            )
        if not quote.tradable:
            return PaperFill(
                order_id, order, "rejected", 0, quote.price, zero, zero, "not_tradable"
            )
        if order.side == "buy":
            return self._execute_buy(state, order_id, trade_date, order, quote)
        return self._execute_sell(state, order_id, trade_date, order, quote)

    def _execute_buy(
        self,
        state: dict[str, object],
        order_id: str,
        trade_date: date,
        order: PaperOrder,
        quote: MarketQuote,
    ) -> PaperFill:
        zero = Decimal("0")
        if quote.limit_up:
            return PaperFill(
                order_id, order, "rejected", 0, quote.price, zero, zero, "limit_up_buy_blocked"
            )
        price = quote.price * (Decimal("1") + self.config.slippage_bps / Decimal("10000"))
        notional = price * Decimal(order.quantity)
        commission = _commission(notional, self.config)
        total_cost = notional + commission
        cash = parse_decimal(state.get("cash"), "cash")
        if total_cost > cash:
            return PaperFill(
                order_id, order, "rejected", 0, price, zero, zero, "insufficient_cash"
            )
        state["cash"] = str(cash - total_cost)
        positions = _dict_value(state, "positions")
        raw_lots = positions.setdefault(order.stock_id, [])
        if not isinstance(raw_lots, list):
            raise PaperBrokerError(f"invalid lots for {order.stock_id}")
        raw_lots.append(
            {
                "quantity": order.quantity,
                "acquired_date": trade_date.isoformat(),
                "cost_price": str(price),
            }
        )
        return PaperFill(
            order_id, order, "filled", order.quantity, price, commission, zero
        )

    def _execute_sell(
        self,
        state: dict[str, object],
        order_id: str,
        trade_date: date,
        order: PaperOrder,
        quote: MarketQuote,
    ) -> PaperFill:
        zero = Decimal("0")
        if quote.limit_down:
            return PaperFill(
                order_id,
                order,
                "rejected",
                0,
                quote.price,
                zero,
                zero,
                "limit_down_sell_blocked",
            )
        positions = _dict_value(state, "positions")
        raw_lots = positions.get(order.stock_id, [])
        if not isinstance(raw_lots, list):
            raise PaperBrokerError(f"invalid lots for {order.stock_id}")
        sellable = sum(
            _int_value(lot, "quantity")
            for lot in raw_lots
            if isinstance(lot, dict)
            and date.fromisoformat(_str_value(lot, "acquired_date")) < trade_date
        )
        filled_quantity = min(order.quantity, sellable)
        filled_quantity -= filled_quantity % self.config.minimum_lot_size
        if filled_quantity <= 0:
            return PaperFill(
                order_id, order, "rejected", 0, quote.price, zero, zero, "t_plus_one_sell_blocked"
            )
        self._remove_lots(raw_lots, filled_quantity, trade_date)
        if not raw_lots:
            positions.pop(order.stock_id, None)
        price = quote.price * (Decimal("1") - self.config.slippage_bps / Decimal("10000"))
        notional = price * Decimal(filled_quantity)
        commission = _commission(notional, self.config)
        stamp_tax = notional * self.config.stamp_tax_bps / Decimal("10000")
        cash = parse_decimal(state.get("cash"), "cash")
        state["cash"] = str(cash + notional - commission - stamp_tax)
        status: FillStatus = "filled" if filled_quantity == order.quantity else "partial"
        message = None if status == "filled" else "partially_blocked_by_t_plus_one"
        return PaperFill(
            order_id,
            order,
            status,
            filled_quantity,
            price,
            commission,
            stamp_tax,
            message,
        )

    @staticmethod
    def _remove_lots(raw_lots: list[object], quantity: int, trade_date: date) -> None:
        remaining = quantity
        for raw_lot in raw_lots:
            if remaining <= 0 or not isinstance(raw_lot, dict):
                continue
            acquired_date = date.fromisoformat(_str_value(raw_lot, "acquired_date"))
            if acquired_date >= trade_date:
                continue
            lot_quantity = _int_value(raw_lot, "quantity")
            sold = min(lot_quantity, remaining)
            raw_lot["quantity"] = lot_quantity - sold
            remaining -= sold
        raw_lots[:] = [
            raw_lot
            for raw_lot in raw_lots
            if isinstance(raw_lot, dict) and _int_value(raw_lot, "quantity") > 0
        ]

    def _snapshot_from_state(
        self,
        state: dict[str, object],
        trade_date: date,
        quotes: Mapping[str, MarketQuote],
    ) -> AccountSnapshot:
        positions = _dict_value(state, "positions")
        snapshots: list[PositionSnapshot] = []
        unpriced: list[str] = []
        market_value = Decimal("0")
        for raw_stock_id, raw_lots in sorted(positions.items()):
            stock_id = normalize_stock_id(raw_stock_id)
            if not isinstance(raw_lots, list):
                raise PaperBrokerError(f"invalid lots for {stock_id}")
            quantity = 0
            sellable_quantity = 0
            cost_value = Decimal("0")
            for raw_lot in raw_lots:
                if not isinstance(raw_lot, dict):
                    raise PaperBrokerError(f"invalid lot for {stock_id}")
                lot_quantity = _int_value(raw_lot, "quantity")
                acquired_date = date.fromisoformat(_str_value(raw_lot, "acquired_date"))
                cost_price = parse_decimal(raw_lot.get("cost_price"), "cost_price")
                quantity += lot_quantity
                cost_value += cost_price * Decimal(lot_quantity)
                if acquired_date < trade_date:
                    sellable_quantity += lot_quantity
            if quantity <= 0:
                continue
            average_cost = cost_value / Decimal(quantity)
            quote = quotes.get(stock_id)
            mark_price = quote.price if quote is not None else average_cost
            if quote is None:
                unpriced.append(stock_id)
            value = mark_price * Decimal(quantity)
            market_value += value
            snapshots.append(
                PositionSnapshot(
                    stock_id=stock_id,
                    quantity=quantity,
                    sellable_quantity=sellable_quantity,
                    average_cost=average_cost,
                    mark_price=mark_price,
                    market_value=value,
                )
            )
        cash = parse_decimal(state.get("cash"), "cash")
        return AccountSnapshot(
            account_id=_str_value(state, "account_id"),
            cash=cash,
            market_value=market_value,
            total_equity=cash + market_value,
            positions=tuple(snapshots),
            unpriced_positions=tuple(unpriced),
        )

    def _new_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "account_id": self.config.account_id,
            "initial_cash": str(self.config.initial_cash),
            "cash": str(self.config.initial_cash),
            "positions": {},
            "batches": {},
            "last_trade_date": None,
        }

    def _read_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            raise PaperBrokerError(
                f"paper account is not initialized: {self.state_path}; run paper-init first"
            )
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PaperBrokerError(f"cannot read paper account state: {self.state_path}") from error
        if not isinstance(raw, dict):
            raise PaperBrokerError("paper account state must be a JSON object")
        state = cast(dict[str, object], raw)
        if state.get("schema_version") != 1:
            raise PaperBrokerError("unsupported paper account state schema")
        if _str_value(state, "account_id") != self.config.account_id:
            raise PaperBrokerError("paper account_id does not match the configured account")
        _dict_value(state, "positions")
        _dict_value(state, "batches")
        _optional_date_value(state, "last_trade_date")
        parse_decimal(state.get("cash"), "cash")
        return state

    def _write_state(self, state: dict[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        with NamedTemporaryFile(
            "wb", delete=False, dir=self.state_path.parent, prefix=f".{self.state_path.name}."
        ) as stream:
            stream.write(content)
            temporary_path = Path(stream.name)
        temporary_path.replace(self.state_path)

    def _batch_result_from_dict(self, raw: dict[str, object]) -> OrderBatchResult:
        raw_fills = raw.get("fills")
        if not isinstance(raw_fills, list):
            raise PaperBrokerError("persisted batch fills must be a list")
        fills: list[PaperFill] = []
        for raw_fill in raw_fills:
            if not isinstance(raw_fill, dict):
                raise PaperBrokerError("persisted fill must be an object")
            raw_order = raw_fill.get("order")
            if not isinstance(raw_order, dict):
                raise PaperBrokerError("persisted fill order must be an object")
            order = PaperOrder(
                stock_id=_str_value(raw_order, "stock_id"),
                side=cast(Any, _str_value(raw_order, "side")),
                quantity=_int_value(raw_order, "quantity"),
                reason=_str_value(raw_order, "reason"),
            )
            fills.append(
                PaperFill(
                    order_id=_str_value(raw_fill, "order_id"),
                    order=order,
                    status=cast(Any, _str_value(raw_fill, "status")),
                    filled_quantity=_int_value(raw_fill, "filled_quantity"),
                    price=parse_decimal(raw_fill.get("price"), "price"),
                    commission=parse_decimal(raw_fill.get("commission"), "commission"),
                    stamp_tax=parse_decimal(raw_fill.get("stamp_tax"), "stamp_tax"),
                    message=cast(str | None, raw_fill.get("message")),
                )
            )
        raw_account = raw.get("account")
        if not isinstance(raw_account, dict):
            raise PaperBrokerError("persisted batch account must be an object")
        raw_positions = raw_account.get("positions")
        if not isinstance(raw_positions, list):
            raise PaperBrokerError("persisted account positions must be a list")
        account = AccountSnapshot(
            account_id=_str_value(raw_account, "account_id"),
            cash=parse_decimal(raw_account.get("cash"), "cash"),
            market_value=parse_decimal(raw_account.get("market_value"), "market_value"),
            total_equity=parse_decimal(raw_account.get("total_equity"), "total_equity"),
            positions=tuple(
                PositionSnapshot(
                    stock_id=_str_value(item, "stock_id"),
                    quantity=_int_value(item, "quantity"),
                    sellable_quantity=_int_value(item, "sellable_quantity"),
                    average_cost=parse_decimal(item.get("average_cost"), "average_cost"),
                    mark_price=parse_decimal(item.get("mark_price"), "mark_price"),
                    market_value=parse_decimal(item.get("market_value"), "market_value"),
                )
                for item in raw_positions
                if isinstance(item, dict)
            ),
            unpriced_positions=tuple(
                str(value) for value in raw_account.get("unpriced_positions", [])
            ),
        )
        return OrderBatchResult(
            batch_id=_str_value(raw, "batch_id"),
            trade_date=date.fromisoformat(_str_value(raw, "trade_date")),
            request_hash=_str_value(raw, "request_hash"),
            replayed=bool(raw.get("replayed", False)),
            fills=tuple(fills),
            account=account,
        )


class _StateLock:
    _STALE_AFTER_SECONDS = 300

    def __init__(self, state_path: Path) -> None:
        self.path = state_path.with_suffix(f"{state_path.suffix}.lock")
        self.file_descriptor: int | None = None

    def __enter__(self) -> _StateLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            if not self._remove_stale_lock():
                raise PaperBrokerError(
                    f"paper account is busy: lock exists at {self.path}"
                ) from error
            try:
                file_descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError as retry_error:
                raise PaperBrokerError(
                    f"paper account is busy: lock exists at {self.path}"
                ) from retry_error
        self.file_descriptor = file_descriptor
        os.write(file_descriptor, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
        self.path.unlink(missing_ok=True)

    def _remove_stale_lock(self) -> bool:
        try:
            age_seconds = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return True
        if age_seconds < self._STALE_AFTER_SECONDS:
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return True


def _commission(notional: Decimal, config: PaperTradingConfig) -> Decimal:
    if notional <= 0:
        return Decimal("0")
    variable = notional * config.commission_bps / Decimal("10000")
    return max(variable, config.minimum_commission)


def _order_id(batch_id: str, index: int, order: PaperOrder) -> str:
    payload = f"{batch_id}|{index}|{order.stock_id}|{order.side}|{order.quantity}|{order.reason}"
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def _batch_request_hash(
    trade_date: date,
    orders: Sequence[PaperOrder],
    quotes: Mapping[str, MarketQuote],
    request_context: Mapping[str, object] | None,
) -> str:
    payload = {
        "trade_date": trade_date.isoformat(),
        "request_context": dict(request_context or {}),
        "orders": [
            {
                "order": order.to_dict(),
                "quote": quotes[order.stock_id].to_dict()
                if order.stock_id in quotes
                else None,
            }
            for order in orders
        ],
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(content.encode("utf-8")).hexdigest()


def _dict_value(raw: dict[str, object], field: str) -> dict[str, object]:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise PaperBrokerError(f"paper account field must be an object: {field}")
    return cast(dict[str, object], value)


def _str_value(raw: Mapping[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise PaperBrokerError(f"paper account field must be a non-empty string: {field}")
    return value


def _int_value(raw: Mapping[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PaperBrokerError(f"paper account field must be an integer: {field}")
    return value


def _optional_date_value(raw: Mapping[str, object], field: str) -> date | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PaperBrokerError(f"paper account field must be an ISO date or null: {field}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise PaperBrokerError(f"paper account field must be an ISO date: {field}") from error
