"""Tests for PIT stock identity resolution (Task 7)."""

from datetime import datetime

import pytest

from seatflow.identity.stock_identity import StockIdentity, StockIdentityRegistry


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestStockIdentityResolution:
    def test_resolve_by_ts_code(self):
        reg = StockIdentityRegistry()
        ident = StockIdentity(
            stock_id="stk-001",
            ts_code="000001.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            valid_to=None,
            aliases=("平安银行",),
        )
        reg.register(ident)
        result = reg.resolve_stock("000001.SZ", _utc("2025-01-01T00:00:00+00:00"))
        assert result.stock_id == "stk-001"

    def test_resolve_by_alias(self):
        reg = StockIdentityRegistry()
        ident = StockIdentity(
            stock_id="stk-001",
            ts_code="000001.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            aliases=("平安银行", "深发展"),
        )
        reg.register(ident)
        result = reg.resolve_stock("深发展", _utc("2025-01-01T00:00:00+00:00"))
        assert result.stock_id == "stk-001"

    def test_renamed_stock_resolves_historical(self):
        """A stock that was renamed must resolve to the old name at the old time."""
        reg = StockIdentityRegistry()
        old = StockIdentity(
            stock_id="stk-001",
            ts_code="000001.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2010-01-01T00:00:00+00:00"),
            valid_to=_utc("2020-06-30T00:00:00+00:00"),
            aliases=("深发展A",),
        )
        new = StockIdentity(
            stock_id="stk-001",
            ts_code="000001.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2020-07-01T00:00:00+00:00"),
            valid_to=None,
            aliases=("平安银行",),
        )
        reg.register(old)
        reg.register(new)

        # At 2015, "深发展A" is valid
        result = reg.resolve_stock("深发展A", _utc("2015-01-01T00:00:00+00:00"))
        assert result.stock_id == "stk-001"

        # At 2025, "深发展A" is no longer valid
        with pytest.raises(KeyError):
            reg.resolve_stock("深发展A", _utc("2025-01-01T00:00:00+00:00"))

        # At 2025, "平安银行" is valid
        result = reg.resolve_stock("平安银行", _utc("2025-01-01T00:00:00+00:00"))
        assert result.stock_id == "stk-001"

    def test_current_alias_does_not_overwrite_past(self):
        reg = StockIdentityRegistry()
        old = StockIdentity(
            stock_id="stk-001",
            ts_code="600001.SH",
            exchange="SH",
            board="MAIN",
            valid_from=_utc("2010-01-01T00:00:00+00:00"),
            valid_to=_utc("2015-01-01T00:00:00+00:00"),
            aliases=("OldName",),
        )
        reg.register(old)
        # Resolving "OldName" at a time before the rename
        result = reg.resolve_stock("OldName", _utc("2012-01-01T00:00:00+00:00"))
        assert result.stock_id == "stk-001"

    def test_overlap_rejected_for_different_stocks(self):
        reg = StockIdentityRegistry()
        a = StockIdentity(
            stock_id="stk-001",
            ts_code="000001.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            valid_to=_utc("2025-01-01T00:00:00+00:00"),
            aliases=("SharedName",),
        )
        b = StockIdentity(
            stock_id="stk-002",
            ts_code="000002.SZ",
            exchange="SZ",
            board="MAIN",
            valid_from=_utc("2023-01-01T00:00:00+00:00"),
            valid_to=None,
            aliases=("SharedName",),
        )
        reg.register(a)
        with pytest.raises(ValueError, match="Overlapping"):
            reg.register(b)

    def test_unresolved_raises(self):
        reg = StockIdentityRegistry()
        with pytest.raises(KeyError):
            reg.resolve_stock("NONEXISTENT", _utc("2025-01-01T00:00:00+00:00"))
