from __future__ import annotations

import json
import os
import tempfile
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from seatflow.data.contracts import parse_decimal
from seatflow.paper_trading.models import (
    DecisionTrace,
    MarketQuote,
    PaperSignal,
    PaperTradingConfig,
    QuoteBatch,
    SignalBatch,
)


def load_paper_trading_config(path: str | Path) -> PaperTradingConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    section = raw.get("paper_trading")
    if not isinstance(section, dict):
        raise ValueError("paper trading config requires a [paper_trading] section")
    values = cast(dict[str, Any], section)
    return PaperTradingConfig(
        account_id=str(values.get("account_id", "seatflow-paper")),
        initial_cash=parse_decimal(values.get("initial_cash", "1000000"), "initial_cash"),
        top_n=int(values.get("top_n", 5)),
        minimum_score=parse_decimal(values.get("minimum_score", "0.5"), "minimum_score"),
        cash_buffer_pct=parse_decimal(
            values.get("cash_buffer_pct", "0.05"), "cash_buffer_pct"
        ),
        max_position_pct=parse_decimal(
            values.get("max_position_pct", "0.20"), "max_position_pct"
        ),
        commission_bps=parse_decimal(
            values.get("commission_bps", "3"), "commission_bps"
        ),
        minimum_commission=parse_decimal(
            values.get("minimum_commission", "5"), "minimum_commission"
        ),
        stamp_tax_bps=parse_decimal(
            values.get("stamp_tax_bps", "5"), "stamp_tax_bps"
        ),
        slippage_bps=parse_decimal(values.get("slippage_bps", "5"), "slippage_bps"),
        minimum_lot_size=int(values.get("minimum_lot_size", 100)),
    )


def load_signal_batch(path: str | Path) -> SignalBatch:
    raw = _load_json_object(path)
    raw_signals = raw.get("signals")
    if not isinstance(raw_signals, list):
        raise ValueError("signal file requires a signals array")
    signals: list[PaperSignal] = []
    for item in raw_signals:
        if not isinstance(item, dict):
            raise ValueError("each signal must be an object")
        signals.append(
            PaperSignal(
                stock_id=str(item.get("stock_id", "")),
                score=parse_decimal(item.get("score"), "score"),
            )
        )
    return SignalBatch(
        as_of_date=date.fromisoformat(_required_str(raw, "as_of_date")),
        generated_at=datetime.fromisoformat(_required_str(raw, "generated_at")),
        signals=tuple(signals),
    )


def load_quote_batch(path: str | Path) -> QuoteBatch:
    raw = _load_json_object(path)
    raw_quotes = raw.get("quotes")
    if not isinstance(raw_quotes, list):
        raise ValueError("quote file requires a quotes array")
    quotes: list[MarketQuote] = []
    for item in raw_quotes:
        if not isinstance(item, dict):
            raise ValueError("each quote must be an object")
        quotes.append(
            MarketQuote(
                stock_id=str(item.get("stock_id", "")),
                price=parse_decimal(item.get("price"), "price"),
                tradable=_bool_value(item, "tradable", True),
                limit_up=_bool_value(item, "limit_up", False),
                limit_down=_bool_value(item, "limit_down", False),
            )
        )
    return QuoteBatch(
        trade_date=date.fromisoformat(_required_str(raw, "trade_date")),
        captured_at=datetime.fromisoformat(_required_str(raw, "captured_at")),
        quotes=tuple(quotes),
    )


def _load_json_object(path: str | Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("input file must contain a JSON object")
    return cast(dict[str, object], raw)


def write_decision_trace(trace: DecisionTrace, path: str | Path) -> None:
    """Persist a Week-3 decision trace as a JSON audit record.

    Mirrors the columns of ``sf_decision_trace`` so the Java side can ingest
    the file verbatim: cycle_id, trade_date, cycle_request_hash, the risk
    verdict (accepted orders, rejections with rule + reason, data_complete,
    model_valid), and the resulting sell/buy fills. One file per cycle,
    written atomically: bytes are flushed to a temp file in the same
    directory, fsync-ed, then os.replace-d to the final path so a crash
    can never leave a partial audit record at ``path``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        trace.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        # Clean up the temp file on any failure (write, fsync, or replace)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _required_str(raw: dict[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool_value(raw: dict[str, object], field: str, default: bool) -> bool:
    value = raw.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value
