from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from hashlib import sha256

from seatflow.paper_trading.broker import PaperBroker
from seatflow.paper_trading.models import (
    AccountSnapshot,
    DecisionTrace,
    MarketQuote,
    OrderBatchResult,
    PaperOrder,
    PaperSignal,
    PaperTradingConfig,
    PaperTradingReport,
    QuoteBatch,
    SignalBatch,
)
from seatflow.paper_trading.risk import (
    PortfolioRiskState,
    RiskCheckResult,
    RiskRuleConfig,
    check_buy_orders,
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
        risk_rule_config: RiskRuleConfig | None = None,
        stock_industries: Mapping[str, str] | None = None,
        risk_state: PortfolioRiskState | None = None,
        data_complete: bool = False,
        model_valid: bool = False,
    ) -> PaperTradingReport:
        if not cycle_id.strip():
            raise ValueError("cycle_id must not be empty")
        # Fail-closed: production execution requires both the risk rule
        # configuration and the risk state. Callers that explicitly want a
        # risk-less run (e.g. offline backtests) must pass a no-op config
        # AND set data_complete=True, model_valid=True so the audit trace
        # records the intent. Without both, refuse to execute. This check
        # runs first so a misconfigured caller cannot trigger downstream
        # input validation paths that might side-effect.
        if risk_rule_config is None or risk_state is None:
            raise ValueError(
                "PaperTradingService.run requires both risk_rule_config and "
                "risk_state for fail-closed execution. Pass an explicit "
                "RiskRuleConfig and PortfolioRiskState, or call from a "
                "backtest harness that supplies both."
            )
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

        # Validate the data and model lineage BEFORE submitting any orders.
        # If either is broken, the audit record is written and we abort;
        # the broker state is unchanged. Forced sell-only paths (e.g. delisting,
        # manual stop-loss) must go through a separate API and are not
        # implemented by this auto-rebalance entry point. We still compute
        # the candidate buy orders so the rejection record can show which
        # stock_ids were blocked by the lineage check, not just an empty list.
        account_before = self.broker.snapshot(quote_batch.trade_date, quotes)
        if not data_complete or not model_valid:
            candidate_buys = self._buy_orders(account_before, selected, quotes)
            risk_result = check_buy_orders(
                orders=candidate_buys,
                account=account_before,
                quotes=quotes,
                stock_industries=stock_industries or {},
                risk_state=risk_state,
                config=risk_rule_config,
                data_complete=data_complete,
                model_valid=model_valid,
            )
            report = PaperTradingReport(
                cycle_id=cycle_id,
                cycle_request_hash=cycle_request_hash,
                trade_date=quote_batch.trade_date,
                selected_signals=selected,
                account_before=account_before,
                sell_batch=_empty_batch(quote_batch.trade_date, f"{cycle_id}:sell"),
                buy_batch=_empty_batch(quote_batch.trade_date, f"{cycle_id}:buy"),
                account_after=account_before,
            )
            return _with_decision_trace(report, risk_result)

        # Compute desired sells before computing buys, but submit them
        # together via the broker. The broker already enforces T+1 and
        # limit-down on sells; the risk check below gates the buys.
        sell_orders = self._sell_orders(account_before, selected, quotes)
        sell_batch = self.broker.submit_market_orders(
            batch_id=f"{cycle_id}:sell",
            trade_date=quote_batch.trade_date,
            orders=sell_orders,
            quotes=quotes,
            request_context=request_context,
        )

        # Run the risk check on the post-sell account so position caps and
        # cash sufficiency reflect the freed-up capital. If risk rejects all
        # buys, the sell fills still stand (T+1 / limit-down were already
        # enforced at submission) and the audit record captures the rejection.
        buy_orders = self._buy_orders(sell_batch.account, selected, quotes)
        risk_result = check_buy_orders(
            orders=buy_orders,
            account=sell_batch.account,
            quotes=quotes,
            stock_industries=stock_industries or {},
            risk_state=risk_state,
            config=risk_rule_config,
            data_complete=True,
            model_valid=True,
        )
        gated_buy_orders = risk_result.accepted

        buy_batch = self.broker.submit_market_orders(
            batch_id=f"{cycle_id}:buy",
            trade_date=quote_batch.trade_date,
            orders=gated_buy_orders,
            quotes=quotes,
            request_context=request_context,
        )
        account_after = self.broker.snapshot(quote_batch.trade_date, quotes)
        report = PaperTradingReport(
            cycle_id=cycle_id,
            cycle_request_hash=cycle_request_hash,
            trade_date=quote_batch.trade_date,
            selected_signals=selected,
            account_before=account_before,
            sell_batch=sell_batch,
            buy_batch=buy_batch,
            account_after=account_after,
        )
        return _with_decision_trace(report, risk_result)

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


def _with_decision_trace(
    report: PaperTradingReport, risk_result: RiskCheckResult
) -> PaperTradingReport:
    """Attach a DecisionTrace audit record to the report.

    The trace is stored under the ``decision_trace`` attribute so the IO layer
    can persist it to ``sf_decision_trace`` without changing the report's
    public ``to_dict()`` shape (existing tests rely on that shape).
    """
    trace = DecisionTrace(
        cycle_id=report.cycle_id,
        trade_date=report.trade_date,
        cycle_request_hash=report.cycle_request_hash,
        risk_accepted=risk_result.accepted,
        risk_rejections=tuple(
            rejection.to_dict() for rejection in risk_result.rejections
        ),
        data_complete=risk_result.data_complete,
        model_valid=risk_result.model_valid,
        sell_fills=report.sell_batch.fills,
        buy_fills=report.buy_batch.fills,
    )
    object.__setattr__(report, "decision_trace", trace)
    return report


def _empty_batch(trade_date: date, batch_id: str) -> OrderBatchResult:
    """Build a zero-fill OrderBatchResult for cycles that never reach the broker.

    Used when fail-closed validation (data_complete / model_valid) blocks the
    cycle before any order is submitted: the audit record needs sell/buy
    placeholders so callers can rely on the report's shape, but the broker
    state is intentionally left untouched.
    """
    empty_account = AccountSnapshot(
        account_id="",
        cash=Decimal("0"),
        market_value=Decimal("0"),
        total_equity=Decimal("0"),
        positions=(),
    )
    return OrderBatchResult(
        batch_id=batch_id,
        trade_date=trade_date,
        request_hash="",
        replayed=False,
        fills=(),
        account=empty_account,
    )
