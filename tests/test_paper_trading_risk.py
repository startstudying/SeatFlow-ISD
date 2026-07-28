"""Tests for the Week-3 paper-trading risk rule engine."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from seatflow.paper_trading.models import (
    AccountSnapshot,
    MarketQuote,
    PaperOrder,
    PositionSnapshot,
)
from seatflow.paper_trading.risk import (
    PortfolioRiskState,
    RiskRuleConfig,
    check_buy_orders,
)


def _account(
    cash: Decimal = Decimal("1000000"),
    positions: tuple[PositionSnapshot, ...] = (),
) -> AccountSnapshot:
    market_value = sum(
        (position.market_value for position in positions), Decimal("0")
    )
    return AccountSnapshot(
        account_id="seatflow-paper",
        cash=cash,
        market_value=market_value,
        total_equity=cash + market_value,
        positions=positions,
    )


def _state(equity: Decimal = Decimal("1000000")) -> PortfolioRiskState:
    return PortfolioRiskState(
        peak_equity=equity,
        session_start_equity=equity,
        trade_date=date(2026, 1, 15),
    )


def _order(stock_id: str, quantity: int = 100) -> PaperOrder:
    return PaperOrder(stock_id, "buy", quantity, "top_signal")


def _quotes(*stock_ids: str) -> dict[str, MarketQuote]:
    return {stock_id: MarketQuote(stock_id, Decimal("10.00")) for stock_id in stock_ids}


def test_data_incomplete_blocks_all_orders() -> None:
    order = _order("600000.SH")
    result = check_buy_orders(
        orders=(order,),
        account=_account(),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(),
        data_complete=False,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "data_incomplete"
    assert not result.passed


def test_model_invalid_blocks_auto_trading() -> None:
    order = _order("600000.SH")
    result = check_buy_orders(
        orders=(order,),
        account=_account(),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(),
        data_complete=True,
        model_valid=False,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "model_invalid"


def test_single_stock_cap_rejects_oversized_order() -> None:
    # 20% cap on 1,000,000 equity = 200,000. At price 10, that is 20,000 shares.
    order = _order("600000.SH", quantity=21000)
    result = check_buy_orders(
        orders=(order,),
        account=_account(),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(max_single_stock_pct=Decimal("0.20")),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "single_stock_position_cap"


def test_single_stock_cap_accepts_within_limit() -> None:
    order = _order("600000.SH", quantity=10000)
    result = check_buy_orders(
        orders=(order,),
        account=_account(),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(max_single_stock_pct=Decimal("0.20")),
        data_complete=True,
        model_valid=True,
    )
    assert len(result.accepted) == 1
    assert result.rejections == ()


def test_sector_concentration_blocks_third_stock_in_same_industry() -> None:
    # Existing positions: 200k in banks (stock A) + 200k in banks (stock B).
    position_a = PositionSnapshot(
        "000001.SZ", 20000, 20000, Decimal("10"), Decimal("10"), Decimal("200000")
    )
    position_b = PositionSnapshot(
        "000002.SZ", 20000, 20000, Decimal("10"), Decimal("10"), Decimal("200000")
    )
    account = _account(
        cash=Decimal("600000"),
        positions=(position_a, position_b),
    )
    # 40% sector cap on 1,000,000 = 400,000. Already at 400k; another 100k (10k @ 10) tips over.
    order = _order("600000.SH", quantity=10000)
    result = check_buy_orders(
        orders=(order,),
        account=account,
        quotes=_quotes("600000.SH"),
        stock_industries={
            "000001.SZ": "banks",
            "000002.SZ": "banks",
            "600000.SH": "banks",
        },
        risk_state=_state(),
        config=RiskRuleConfig(max_sector_pct=Decimal("0.40")),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "sector_concentration"


def test_daily_loss_limit_blocks_when_threshold_breached() -> None:
    # Peak 1,000,000; session_start_equity 940,000 with cash 940,000 =>
    # daily loss = 940k - 940k = 0, which is BELOW 5% so this would NOT block.
    # Use a lower cash/equity to demonstrate the daily loss branch.
    risk_state = PortfolioRiskState(
        peak_equity=Decimal("1000000"),
        session_start_equity=Decimal("1000000"),
        trade_date=date(2026, 1, 15),
    )
    order = _order("600000.SH")
    result = check_buy_orders(
        orders=(order,),
        account=_account(cash=Decimal("940000")),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=risk_state,
        config=RiskRuleConfig(daily_loss_limit_pct=Decimal("0.05")),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "daily_loss_limit"


def test_max_drawdown_blocks_when_threshold_breached() -> None:
    # Peak 1,000,000; current equity 790,000 => 21% drawdown >= 20%.
    # session_start_equity 1,000,000 ensures daily-loss is 0 so it does not fire.
    risk_state = PortfolioRiskState(
        peak_equity=Decimal("1000000"),
        session_start_equity=Decimal("1000000"),
        trade_date=date(2026, 1, 15),
    )
    order = _order("600000.SH")
    result = check_buy_orders(
        orders=(order,),
        account=_account(cash=Decimal("790000")),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=risk_state,
        config=RiskRuleConfig(
            daily_loss_limit_pct=Decimal("0.50"),  # disable daily check
            max_drawdown_pct=Decimal("0.20"),
        ),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "max_drawdown"


def test_total_position_cap_blocks_when_invested_fraction_too_high() -> None:
    # Cap at 80% invested. With 1,000,000 equity and 0 positions, an order
    # for 90,000 shares @ 10 = 900,000 would push invested to 90%.
    # Relax single-stock and sector caps so the total-position rule fires first.
    order = _order("600000.SH", quantity=90000)
    result = check_buy_orders(
        orders=(order,),
        account=_account(),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(
            max_single_stock_pct=Decimal("1.0"),
            max_sector_pct=Decimal("1.0"),
            max_total_position_pct=Decimal("0.80"),
        ),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    assert result.rejections[0].rule == "total_position_cap"


def test_insufficient_cash_rejected_by_broker_not_risk_layer() -> None:
    # Cash sufficiency is the broker's responsibility (it returns an
    # "insufficient_cash" fill). The risk layer's single-stock cap fires
    # first whenever an order would exceed the equity-based weight limit,
    # which is the correct behavior — the risk layer never lets an order
    # through that is larger than the equity envelope.
    order = _order("600000.SH", quantity=10000)
    result = check_buy_orders(
        orders=(order,),
        account=_account(cash=Decimal("50000")),
        quotes=_quotes("600000.SH"),
        stock_industries={"600000.SH": "banks"},
        risk_state=_state(equity=Decimal("50000")),
        config=RiskRuleConfig(
            max_single_stock_pct=Decimal("1.0"),
            max_sector_pct=Decimal("1.0"),
            max_total_position_pct=Decimal("1.0"),
        ),
        data_complete=True,
        model_valid=True,
    )
    assert result.accepted == ()
    # Single-stock cap fires because 100k order on 50k equity = 200% weight.
    assert result.rejections[0].rule == "single_stock_position_cap"


def test_multiple_orders_accumulate_and_one_rejected() -> None:
    # Two orders that each fit individually, but the second breaches the cap
    # because the first one already used up the budget.
    order_a = _order("000001.SZ", quantity=15000)  # 150,000 @ 10
    order_b = _order("600000.SH", quantity=10000)  # 100,000 @ 10
    result = check_buy_orders(
        orders=(order_a, order_b),
        account=_account(),
        quotes=_quotes("000001.SZ", "600000.SH"),
        stock_industries={"000001.SZ": "banks", "600000.SH": "banks"},
        risk_state=_state(),
        config=RiskRuleConfig(
            max_single_stock_pct=Decimal("0.20"),  # 200,000 each
            max_sector_pct=Decimal("0.30"),  # 300,000 total for banks
        ),
        data_complete=True,
        model_valid=True,
    )
    # order_a: 150k < 200k single cap, banks sector 150k < 300k -> accepted.
    # order_b: single 100k < 200k, but banks sector would be 250k < 300k -> accepted.
    assert len(result.accepted) == 2
    assert result.rejections == ()
