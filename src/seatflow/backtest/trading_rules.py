from __future__ import annotations

from decimal import Decimal

from seatflow.data.contracts import normalize_stock_id


def price_limit_pct(stock_id: str, fallback: Decimal) -> Decimal:
    """Return the A-share daily price-limit ratio for the stock board."""

    normalized = normalize_stock_id(stock_id)
    code, exchange = normalized.split(".")
    if exchange == "BJ":
        return Decimal("0.30")
    if exchange == "SH" and code.startswith(("688", "689")):
        return Decimal("0.20")
    if exchange == "SZ" and code.startswith(("300", "301")):
        return Decimal("0.20")
    if _is_main_board(code, exchange):
        return Decimal("0.10")
    return fallback


def _is_main_board(code: str, exchange: str) -> bool:
    if exchange == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if exchange == "SZ":
        return code.startswith(("000", "001", "002", "003"))
    return False
