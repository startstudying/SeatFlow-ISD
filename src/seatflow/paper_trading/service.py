from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from hashlib import sha256

from seatflow.paper_trading.broker import PaperBroker
from seatflow.paper_trading.models import (
    AccountSnapshot,
    MarketQuote,
    PaperOrder,
    PaperSignal,
    PaperTradingConfig,
    PaperTradingReport,
    QuoteBatch,
    SignalBatch,
)


class PaperTradingService:
    def __init__(self, broker: PaperBroker, config: PaperTradingConfig) -> None:
        self.broker = broker
        self.config = config

    def run(
        self,
        *,
        cycle_id: str,
        signal_batch: SignalBatch,
        quote_batch: QuoteBatch,
    ) -> PaperTradingReport:
        if not cycle_id.strip():
            raise ValueError("cycle_id must not be empty")
        if signal_batch.as_of_date >= quote_batch.trade_date:
            raise ValueError("paper execution requires signals from an earlier trading date")
        if signal_batch.generated_at.date() > quote_batch.trade_date:
            raise ValueError("signal batch was generated after the execution date")
        if signal_batch.generated_at >= quote_batch.captured_at:
            raise ValueError("signals must be generated before the quote snapshot")

        quotes = {quote.stock_id: quote for quote in quote_batch.quotes}
        cycle_request_hash = _cycle_request_hash(signal_batch, quote_batch, self.config)
        request_context = {"cycle_request_hash": cycle_request_hash}
        selected = self._select_signals(signal_batch.signals, quotes)
        account_before = self.broker.snapshot(quote_batch.trade_date, quotes)
        sell_orders = self._sell_orders(account_before, selected, quotes)
        sell_batch = self.broker.submit_market_orders(
            batch_id=f"{cycle_id}:sell",
            trade_date=quote_batch.trade_date,
            orders=sell_orders,
            quotes=quotes,
            request_context=request_context,
        )
        buy_orders = self._buy_orders(sell_batch.account, selected, quotes)
        buy_batch = self.broker.submit_market_orders(
            batch_id=f"{cycle_id}:buy",
            trade_date=quote_batch.trade_date,
            orders=buy_orders,
            quotes=quotes,
            request_context=request_context,
        )
        account_after = self.broker.snapshot(quote_batch.trade_date, quotes)
        return PaperTradingReport(
            cycle_id=cycle_id,
            cycle_request_hash=cycle_request_hash,
            trade_date=quote_batch.trade_date,
            selected_signals=selected,
            account_before=account_before,
            sell_batch=sell_batch,
            buy_batch=buy_batch,
            account_after=account_after,
        )

    def _select_signals(
        self,
        signals: Sequence[PaperSignal],
        quotes: Mapping[str, MarketQuote],
    ) -> tuple[PaperSignal, ...]:
        ranked = sorted(
            (
                signal
                for signal in signals
                if signal.score >= self.config.minimum_score and signal.stock_id in quotes
            ),
            key=lambda signal: (-signal.score, signal.stock_id),
        )
        return tuple(ranked[: self.config.top_n])

    @staticmethod
    def _sell_orders(
        account: AccountSnapshot,
        selected: Sequence[PaperSignal],
        quotes: Mapping[str, MarketQuote],
    ) -> tuple[PaperOrder, ...]:
        selected_ids = {signal.stock_id for signal in selected}
        return tuple(
            PaperOrder(position.stock_id, "sell", position.quantity, "signal_exit")
            for position in account.positions
            if position.stock_id not in selected_ids and position.stock_id in quotes
        )

    def _buy_orders(
        self,
        account: AccountSnapshot,
        selected: Sequence[PaperSignal],
        quotes: Mapping[str, MarketQuote],
    ) -> tuple[PaperOrder, ...]:
        if not selected:
            return ()
        position_map = account.position_map()
        allocation_pct = min(
            (Decimal("1") - self.config.cash_buffer_pct) / Decimal(len(selected)),
            self.config.max_position_pct,
        )
        target_value = account.total_equity * allocation_pct
        reserved_cash = account.total_equity * self.config.cash_buffer_pct
        spendable_cash = max(account.cash - reserved_cash, Decimal("0"))
        orders: list[PaperOrder] = []
        for signal in selected:
            quote = quotes[signal.stock_id]
            if not quote.tradable or quote.limit_up:
                continue
            current_value = (
                position_map[signal.stock_id].market_value
                if signal.stock_id in position_map
                else Decimal("0")
            )
            desired_notional = max(target_value - current_value, Decimal("0"))
            quantity = _round_lot(
                desired_notional / _buy_price(quote, self.config),
                self.config.minimum_lot_size,
            )
            quantity = self._cap_to_cash(quantity, quote, spendable_cash)
            if quantity <= 0:
                continue
            estimated_cost = _estimated_buy_cost(quantity, quote, self.config)
            spendable_cash -= estimated_cost
            orders.append(PaperOrder(signal.stock_id, "buy", quantity, "top_signal"))
        return tuple(orders)

    def _cap_to_cash(
        self, quantity: int, quote: MarketQuote, spendable_cash: Decimal
    ) -> int:
        lot_size = self.config.minimum_lot_size
        while quantity > 0:
            if _estimated_buy_cost(quantity, quote, self.config) <= spendable_cash:
                return quantity
            quantity -= lot_size
        return 0


def _buy_price(quote: MarketQuote, config: PaperTradingConfig) -> Decimal:
    return quote.price * (Decimal("1") + config.slippage_bps / Decimal("10000"))


def _estimated_buy_cost(
    quantity: int, quote: MarketQuote, config: PaperTradingConfig
) -> Decimal:
    notional = _buy_price(quote, config) * Decimal(quantity)
    commission = max(
        notional * config.commission_bps / Decimal("10000"),
        config.minimum_commission,
    )
    return notional + commission


def _round_lot(quantity: Decimal, lot_size: int) -> int:
    lots = (quantity / Decimal(lot_size)).to_integral_value(rounding=ROUND_FLOOR)
    return int(lots) * lot_size


def _cycle_request_hash(
    signal_batch: SignalBatch,
    quote_batch: QuoteBatch,
    config: PaperTradingConfig,
) -> str:
    payload = {
        "signal_batch": signal_batch.to_dict(),
        "quote_batch": quote_batch.to_dict(),
        "config": config.to_dict(),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(content.encode("utf-8")).hexdigest()
