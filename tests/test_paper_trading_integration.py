"""Week-3 integration tests: service risk-gate wiring + decision_trace audit."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from seatflow.paper_trading.broker import JsonPaperBroker
from seatflow.paper_trading.io import write_decision_trace
from seatflow.paper_trading.models import (
    MarketQuote,
    PaperTradingConfig,
    QuoteBatch,
    SignalBatch,
)
from seatflow.paper_trading.risk import PortfolioRiskState, RiskRuleConfig
from seatflow.paper_trading.service import PaperTradingService

_TZ = UTC


def _config() -> PaperTradingConfig:
    return PaperTradingConfig(
        account_id="seatflow-paper",
        initial_cash=Decimal("1000000"),
        top_n=3,
        minimum_score=Decimal("0.5"),
        cash_buffer_pct=Decimal("0.05"),
        max_position_pct=Decimal("0.20"),
    )


def _signal_batch(signals: tuple[tuple[str, str], ...], as_of: date) -> SignalBatch:
    from seatflow.paper_trading.models import PaperSignal

    return SignalBatch(
        as_of_date=as_of,
        generated_at=datetime(as_of.year, as_of.month, as_of.day, 9, 30, tzinfo=_TZ),
        signals=tuple(PaperSignal(stock_id=s, score=Decimal(v)) for s, v in signals),
    )


def _quote_batch(quotes: tuple[tuple[str, str, bool], ...], trade_date: date) -> QuoteBatch:
    return QuoteBatch(
        trade_date=trade_date,
        captured_at=datetime(trade_date.year, trade_date.month, trade_date.day, 15, 0, tzinfo=_TZ),
        quotes=tuple(
            MarketQuote(stock_id=s, price=Decimal(p), tradable=t)
            for s, p, t in quotes
        ),
    )


def _service(state_dir: Path) -> PaperTradingService:
    broker = JsonPaperBroker(state_dir / "paper_account.json", _config())
    broker.initialize()
    return PaperTradingService(broker, _config())


def test_service_attaches_decision_trace_when_risk_config_supplied() -> None:
    with TemporaryDirectory() as tmp:
        service = _service(Path(tmp))
        as_of = date(2026, 1, 14)
        trade_date = date(2026, 1, 15)
        report = service.run(
            cycle_id="cycle-1",
            signal_batch=_signal_batch(
                (("600000.SH", "0.8"), ("000001.SZ", "0.7")), as_of
            ),
            quote_batch=_quote_batch(
                (("600000.SH", "10.00", True), ("000001.SZ", "10.00", True)),
                trade_date,
            ),
            risk_rule_config=RiskRuleConfig(),
            stock_industries={"600000.SH": "banks", "000001.SZ": "banks"},
            risk_state=PortfolioRiskState(
                peak_equity=Decimal("1000000"),
                session_start_equity=Decimal("1000000"),
                trade_date=trade_date,
            ),
            data_complete=True,
            model_valid=True,
        )
        trace = getattr(report, "decision_trace", None)
        assert trace is not None
        assert trace.cycle_id == "cycle-1"
        assert trace.trade_date == trade_date
        assert trace.data_complete is True
        assert trace.model_valid is True


def test_service_refuses_to_run_without_risk_state() -> None:
    """Fail-closed: omitting risk_rule_config or risk_state must raise, not
    silently produce a report with no decision_trace."""
    with TemporaryDirectory() as tmp:
        service = _service(Path(tmp))
        as_of = date(2026, 1, 14)
        trade_date = date(2026, 1, 15)
        with pytest.raises(ValueError, match="fail-closed"):
            service.run(
                cycle_id="cycle-2",
                signal_batch=_signal_batch((("600000.SH", "0.8"),), as_of),
                quote_batch=_quote_batch(
                    (("600000.SH", "10.00", True),), trade_date
                ),
            )


def test_model_invalid_blocks_all_buys_and_trace_records_it() -> None:
    with TemporaryDirectory() as tmp:
        service = _service(Path(tmp))
        as_of = date(2026, 1, 14)
        trade_date = date(2026, 1, 15)
        report = service.run(
            cycle_id="cycle-3",
            signal_batch=_signal_batch((("600000.SH", "0.8"),), as_of),
            quote_batch=_quote_batch((("600000.SH", "10.00", True),), trade_date),
            risk_rule_config=RiskRuleConfig(),
            stock_industries={"600000.SH": "banks"},
            risk_state=PortfolioRiskState(
                peak_equity=Decimal("1000000"),
                session_start_equity=Decimal("1000000"),
                trade_date=trade_date,
            ),
            data_complete=True,
            model_valid=False,
        )
        trace = report.decision_trace
        assert trace.model_valid is False
        assert trace.risk_accepted == ()
        assert len(trace.risk_rejections) >= 1
        assert trace.risk_rejections[0]["rule"] == "model_invalid"
        # Buy batch still ran but with zero accepted orders, so no fills.
        assert trace.buy_fills == ()


def test_data_incomplete_blocks_all_buys() -> None:
    with TemporaryDirectory() as tmp:
        service = _service(Path(tmp))
        as_of = date(2026, 1, 14)
        trade_date = date(2026, 1, 15)
        report = service.run(
            cycle_id="cycle-4",
            signal_batch=_signal_batch((("600000.SH", "0.8"),), as_of),
            quote_batch=_quote_batch((("600000.SH", "10.00", True),), trade_date),
            risk_rule_config=RiskRuleConfig(),
            stock_industries={"600000.SH": "banks"},
            risk_state=PortfolioRiskState(
                peak_equity=Decimal("1000000"),
                session_start_equity=Decimal("1000000"),
                trade_date=trade_date,
            ),
            data_complete=False,
            model_valid=True,
        )
        trace = report.decision_trace
        assert trace.data_complete is False
        assert trace.risk_accepted == ()
        assert trace.risk_rejections[0]["rule"] == "data_incomplete"


def test_write_decision_trace_round_trip() -> None:
    with TemporaryDirectory() as tmp:
        service = _service(Path(tmp))
        as_of = date(2026, 1, 14)
        trade_date = date(2026, 1, 15)
        report = service.run(
            cycle_id="cycle-5",
            signal_batch=_signal_batch((("600000.SH", "0.8"),), as_of),
            quote_batch=_quote_batch((("600000.SH", "10.00", True),), trade_date),
            risk_rule_config=RiskRuleConfig(),
            stock_industries={"600000.SH": "banks"},
            risk_state=PortfolioRiskState(
                peak_equity=Decimal("1000000"),
                session_start_equity=Decimal("1000000"),
                trade_date=trade_date,
            ),
            data_complete=True,
            model_valid=True,
        )
        out = Path(tmp) / "decision_trace.json"
        write_decision_trace(report.decision_trace, out)
        raw = json.loads(out.read_text(encoding="utf-8"))
        assert raw["cycle_id"] == "cycle-5"
        assert raw["trade_date"] == "2026-01-15"
        assert raw["data_complete"] is True
        assert raw["model_valid"] is True
        assert isinstance(raw["risk_rejections"], list)
        assert isinstance(raw["buy_fills"], list)
