"""Week-3 risk rule enforcement for the paper-trading broker.

These rules are evaluated BEFORE orders are submitted to the broker. A rejected
rule produces a `RiskRejection` that the caller must record in the audit log
(``sf_decision_trace``) so the decision is traceable end-to-end.

The rules intentionally take industry membership, daily PnL, drawdown, and
``model_valid`` as inputs supplied by the Week-2 pipeline rather than reading
them from the broker state, so the risk layer stays pure and testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from seatflow.paper_trading.models import AccountSnapshot, PaperOrder

RiskRuleName = Literal[
    "single_stock_position_cap",
    "total_position_cap",
    "sector_concentration",
    "daily_loss_limit",
    "max_drawdown",
    "data_incomplete",
    "model_invalid",
]


@dataclass(frozen=True)
class RiskRuleConfig:
    """Configuration for the Week-3 risk rule engine.

    All thresholds are non-negative. ``max_sector_pct`` is the maximum fraction
    of total equity that may be allocated to a single industry. A value of
    ``Decimal("0")`` disables a numeric rule (it always passes).
    """

    max_single_stock_pct: Decimal = Decimal("0.20")
    max_total_position_pct: Decimal = Decimal("1.0")
    max_sector_pct: Decimal = Decimal("0.40")
    daily_loss_limit_pct: Decimal = Decimal("0.05")
    max_drawdown_pct: Decimal = Decimal("0.20")

    def __post_init__(self) -> None:
        for name, value in (
            ("max_single_stock_pct", self.max_single_stock_pct),
            ("max_total_position_pct", self.max_total_position_pct),
            ("max_sector_pct", self.max_sector_pct),
            ("daily_loss_limit_pct", self.daily_loss_limit_pct),
            ("max_drawdown_pct", self.max_drawdown_pct),
        ):
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not (
            Decimal("0") <= self.max_single_stock_pct <= Decimal("1")
        ):
            raise ValueError("max_single_stock_pct must be in [0, 1]")
        if not (
            Decimal("0") <= self.max_total_position_pct <= Decimal("1")
        ):
            raise ValueError("max_total_position_pct must be in [0, 1]")
        if not (Decimal("0") <= self.max_sector_pct <= Decimal("1")):
            raise ValueError("max_sector_pct must be in [0, 1]")


@dataclass(frozen=True)
class RiskRejection:
    """A single rejected order with the rule that blocked it."""

    order: PaperOrder
    rule: RiskRuleName
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order.to_dict(),
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RiskCheckResult:
    """The outcome of a risk check pass over a batch of buy orders."""

    accepted: tuple[PaperOrder, ...]
    rejections: tuple[RiskRejection, ...]
    data_complete: bool
    model_valid: bool

    @property
    def passed(self) -> bool:
        return self.data_complete and self.model_valid

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": [order.to_dict() for order in self.accepted],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "data_complete": self.data_complete,
            "model_valid": self.model_valid,
        }


@dataclass(frozen=True)
class PortfolioRiskState:
    """Daily portfolio risk state carried across the rebalance cycle.

    ``peak_equity`` is the highest total equity seen historically. The
    session-start equity is the portfolio's value at the open of the current
    trading day (i.e. the previous close, adjusted for overnight events).
    The current equity is read directly from the account snapshot during
    the risk check, so it is not part of this struct.
    """

    peak_equity: Decimal
    session_start_equity: Decimal
    trade_date: date

    def __post_init__(self) -> None:
        if not self.peak_equity.is_finite() or self.peak_equity <= 0:
            raise ValueError("peak_equity must be positive")
        if not self.session_start_equity.is_finite() or self.session_start_equity <= 0:
            raise ValueError("session_start_equity must be positive")
        if self.session_start_equity > self.peak_equity:
            raise ValueError("session_start_equity cannot exceed peak_equity")


def check_buy_orders(
    *,
    orders: tuple[PaperOrder, ...],
    account: AccountSnapshot,
    quotes: Mapping[str, object],
    stock_industries: Mapping[str, str],
    risk_state: PortfolioRiskState,
    config: RiskRuleConfig,
    data_complete: bool,
    model_valid: bool,
) -> RiskCheckResult:
    """Evaluate risk rules against a batch of buy orders.

    Sell orders are not rate-limited here; the broker already enforces T+1
    and limit-down. This function focuses on the buy-side risk envelope.

    The ``quotes`` mapping is keyed by stock_id and must expose at least a
    ``price`` attribute (``MarketQuote`` or any duck-typed equivalent).
    """
    if not data_complete:
        rejections = tuple(
            RiskRejection(order, "data_incomplete", "data pipeline incomplete")
            for order in orders
        )
        return RiskCheckResult(
            accepted=(),
            rejections=rejections,
            data_complete=False,
            model_valid=model_valid,
        )

    if not model_valid:
        rejections = tuple(
            RiskRejection(
                order,
                "model_invalid",
                "model_valid=false: human confirmation required, auto-trading blocked",
            )
            for order in orders
        )
        return RiskCheckResult(
            accepted=(),
            rejections=rejections,
            data_complete=True,
            model_valid=False,
        )

    rejections_list: list[RiskRejection] = []
    accepted: list[PaperOrder] = []

    total_equity = account.total_equity
    if total_equity <= 0:
        return RiskCheckResult(
            accepted=(),
            rejections=tuple(
                RiskRejection(order, "total_position_cap", "total equity non-positive")
                for order in orders
            ),
            data_complete=True,
            model_valid=True,
        )

    # Daily loss = today's drop from session start (previous close adjusted)
    # to the current equity. Drawdown = drop from the historical peak.
    # They are distinct metrics and must be measured separately; previously
    # both compared peak_equity - equity_today against peak_equity, which
    # silently double-counted drawdown as daily loss.
    daily_loss = risk_state.session_start_equity - total_equity
    if (
        config.daily_loss_limit_pct > 0
        and daily_loss / risk_state.session_start_equity >= config.daily_loss_limit_pct
    ):
        rejections_list.extend(
            RiskRejection(order, "daily_loss_limit", "daily loss limit reached")
            for order in orders
        )
        return RiskCheckResult(
            accepted=(),
            rejections=tuple(rejections_list),
            data_complete=True,
            model_valid=True,
        )
    drawdown = risk_state.peak_equity - total_equity
    if (
        config.max_drawdown_pct > 0
        and drawdown / risk_state.peak_equity >= config.max_drawdown_pct
    ):
        rejections_list.extend(
            RiskRejection(order, "max_drawdown", "max drawdown threshold breached")
            for order in orders
        )
        return RiskCheckResult(
            accepted=(),
            rejections=tuple(rejections_list),
            data_complete=True,
            model_valid=True,
        )

    # Projected position values after each buy, accumulated incrementally.
    projected_holdings: dict[str, Decimal] = {
        position.stock_id: position.market_value for position in account.positions
    }
    projected_cash = account.cash

    for order in orders:
        quote = quotes.get(order.stock_id)
        price = _quote_price(quote)
        if price is None:
            rejections_list.append(
                RiskRejection(order, "data_incomplete", f"quote missing for {order.stock_id}")
            )
            continue
        order_value = price * Decimal(order.quantity)

        # Single-stock cap: post-trade value cannot exceed max_single_stock_pct.
        new_holding = projected_holdings.get(order.stock_id, Decimal("0")) + order_value
        if (
            config.max_single_stock_pct > 0
            and new_holding / total_equity > config.max_single_stock_pct
        ):
            rejections_list.append(
                RiskRejection(
                    order,
                    "single_stock_position_cap",
                    f"post-trade {order.stock_id} weight exceeds "
                    f"{config.max_single_stock_pct}",
                )
            )
            continue

        # Sector concentration: aggregate industry weight post-trade.
        industry = stock_industries.get(order.stock_id)
        if industry is not None and config.max_sector_pct > 0:
            sector_value = Decimal("0")
            for stock_id, holding_value in projected_holdings.items():
                if stock_industries.get(stock_id) == industry:
                    sector_value += holding_value
            sector_value += order_value
            if sector_value / total_equity > config.max_sector_pct:
                rejections_list.append(
                    RiskRejection(
                        order,
                        "sector_concentration",
                        f"industry '{industry}' weight exceeds "
                        f"{config.max_sector_pct}",
                    )
                )
                continue

        # Total position cap: post-trade invested fraction cannot exceed limit.
        invested = sum(projected_holdings.values()) + order_value
        if (
            config.max_total_position_pct > 0
            and config.max_total_position_pct < Decimal("1")
            and invested / total_equity > config.max_total_position_pct
        ):
            rejections_list.append(
                RiskRejection(order, "total_position_cap", "total position cap reached")
            )
            continue

        # Cash sufficiency: this is a guard, the broker also checks it.
        if order_value > projected_cash:
            rejections_list.append(
                RiskRejection(order, "total_position_cap", "insufficient cash for order")
            )
            continue

        projected_holdings[order.stock_id] = new_holding
        projected_cash -= order_value
        accepted.append(order)

    return RiskCheckResult(
        accepted=tuple(accepted),
        rejections=tuple(rejections_list),
        data_complete=True,
        model_valid=True,
    )


def _quote_price(quote: object | None) -> Decimal | None:
    if quote is None:
        return None
    price = getattr(quote, "price", None)
    if isinstance(price, Decimal):
        return price
    return None
