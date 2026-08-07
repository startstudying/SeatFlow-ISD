"""Tests for the source-to-domain ISD adapter (Task 3)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from seatflow.dataset.isd_adapter import (
    ISDMarketBar,
    _normalize_reason,
    build_isd_inputs,
    build_isd_labels,
    build_lhb_stock_universe,
)


def _aware(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _aware_10am(year: int, month: int, day: int) -> datetime:
    """Asia/Shanghai 18:00 → 10:00 UTC."""
    return datetime(year, month, day, 10, 0, 0, tzinfo=timezone.utc)


def _write_snapshot(
    root: Path,
    *,
    dataset: str,
    trade_date: str,
    rows: list[dict],
    source_version: str = "top_list",
) -> None:
    dataset_dir = root / dataset
    snap_dir = dataset_dir / f"date={trade_date}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
    (snap_dir / "data.json").write_bytes(raw)
    manifest = {
        "dataset": dataset.split("/")[-1],
        "date": trade_date,
        "provider": "tushare",
        "row_count": len(rows),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_version": source_version,
    }
    (snap_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _minimal_summary(
    stock_id: str,
    reason: str,
    *,
    trade_date: str = "2023-01-03",
) -> dict:
    return {
        "stock_id": stock_id,
        "stock_name": "Test",
        "trade_date": trade_date,
        "reason": reason,
        "buy_amount": 1.0e8,
        "sell_amount": 5.0e7,
        "net_amount": 5.0e7,
        "available_time": f"{trade_date}T18:00:00+08:00",
        "source_name": "tushare",
        "source_version": "top_list",
    }


def _minimal_seat(
    stock_id: str,
    reason: str,
    *,
    trade_date: str = "2023-01-03",
    side: str = "buy",
    identity_scope: str = "named_seat",
    seat_id: str = "东方财富证券股份有限公司山南香曲东路证券营业部",
) -> dict:
    return {
        "stock_id": stock_id,
        "trade_date": trade_date,
        "event_reason": reason,
        "side": side,
        "amount": 1.0e6,
        "seat_id": seat_id,
        "identity_scope": identity_scope,
        "available_time": f"{trade_date}T18:00:00+08:00",
        "source_name": "tushare",
        "source_version": "top_inst",
    }


def _minimal_bar(stock_id: str, *, trade_date: str = "2023-01-03") -> dict:
    return {
        "stock_id": stock_id,
        "trade_date": trade_date,
        "open_price": "10.0",
        "close_price": "11.0",
        "high_price": "11.5",
        "low_price": "9.5",
        "pre_close": "10.0",
        "volume": "1000000.0",
        "amount": "10000.0",
        "amount_unit": "thousand_cny",
        "volume_unit": "hand",
        "available_time": f"{trade_date}T18:00:00+08:00",
        "source_name": "tushare",
        "source_version": "tushare-pro-rest",
    }


def _minimal_basic(stock_id: str, *, trade_date: str = "2023-01-03") -> dict:
    return {
        "stock_id": stock_id,
        "trade_date": trade_date,
        "turnover_rate": 1.0,
        "turnover_rate_f": 1.0,
        "volume_ratio": 1.0,
        "total_share": 1.0e6,
        "float_share": 1.0e6,
        "free_share": 1.0e6,
        "total_mv": 1.0e7,
        "circ_mv": 1.0e7,
        "available_time": f"{trade_date}T18:00:00+08:00",
        "source_name": "tushare",
        "source_version": "daily_basic",
    }


# -- Helpers for label tests --


def _bar(year: int, month: int, day: int, open_: float, close: float) -> ISDMarketBar:
    return ISDMarketBar(
        stock_id="000001.SZ",
        trade_date=_aware(year, month, day),
        open=open_,
        close=close,
        high=max(open_, close),
        low=min(open_, close),
        volume=1.0,
        available_time=_aware(year, month, day),
    )


# -- Reason normalization --


class TestReasonNormalization:
    def test_normalize_strips_whitespace(self):
        assert _normalize_reason("  R-A  ") == "R-A"

    def test_normalize_dedups_fragments(self):
        assert _normalize_reason("R-A,R-A,R-B") == "R-A,R-B"

    def test_normalize_preserves_order(self):
        assert _normalize_reason("R-B,R-A") == "R-B,R-A"


# -- Distinct reasons on one stock/day --


class TestDistinctReasons:
    def test_two_reasons_one_stock_day_yield_two_events(self, tmp_path: Path):
        summary = [
            _minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03"),
            _minimal_summary("000001.SZ", "R-B", trade_date="2023-01-03"),
        ]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        assert len(inputs.events) == 2
        reasons = sorted(ev.reason_codes[0] for ev in inputs.events)
        assert reasons == ["R-A", "R-B"]


# -- Seat participation binding by stock/day/reason --


class TestSeatBinding:
    def test_seats_bind_by_stock_day_reason(self, tmp_path: Path):
        summary = [
            _minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03"),
            _minimal_summary("000001.SZ", "R-B", trade_date="2023-01-03"),
        ]
        seat = [
            _minimal_seat("000001.SZ", "R-A"),
            _minimal_seat("000001.SZ", "R-B"),
        ]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_seat_event",
            trade_date="2023-01-03",
            rows=seat,
            source_version="top_inst",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        assert len(inputs.participations) == 2

    def test_seat_with_unknown_reason_is_unbound(self, tmp_path: Path):
        summary = [_minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03")]
        seat = [_minimal_seat("000001.SZ", "R-UNKNOWN")]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_seat_event",
            trade_date="2023-01-03",
            rows=seat,
            source_version="top_inst",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        assert len(inputs.participations) == 0
        assert inputs.stats.unbound_seat_count == 1


# -- Identity scope classification --


class TestIdentityScope:
    def test_named_seat_counted_separately_from_proxy(self, tmp_path: Path):
        summary = [_minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03")]
        seat = [
            _minimal_seat("000001.SZ", "R-A", identity_scope="named_seat"),
            _minimal_seat(
                "000001.SZ",
                "R-A",
                identity_scope="institutional_flow_proxy",
                seat_id="机构专用席位",
            ),
        ]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_seat_event",
            trade_date="2023-01-03",
            rows=seat,
            source_version="top_inst",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        assert inputs.stats.named_seat_count == 1
        assert inputs.stats.institutional_flow_proxy_count == 1

    def test_unknown_identity_scope_is_counted(self, tmp_path: Path):
        summary = [_minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03")]
        seat = [
            _minimal_seat("000001.SZ", "R-A", identity_scope="unknown_scope"),
        ]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_seat_event",
            trade_date="2023-01-03",
            rows=seat,
            source_version="top_inst",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        assert inputs.stats.unknown_scope_count == 1
        assert inputs.stats.institutional_flow_proxy_count == 0


# -- Timezone parsing --


class TestTimezoneParsing:
    def test_shanghai_availability_converts_to_aware_utc(self, tmp_path: Path):
        summary = [_minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03")]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=summary,
            source_version="top_list",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-03"
        )
        ev = inputs.events[0]
        assert ev.available_time.tzinfo is not None
        assert ev.available_time == _aware_10am(2023, 1, 3)
        assert ev.available_time >= ev.trade_date


# -- LHB stock universe --


class TestLHBStockUniverse:
    def test_universe_excludes_never_listed_stocks(self, tmp_path: Path):
        summary = [
            _minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03"),
            _minimal_summary("000002.SZ", "R-B", trade_date="2023-01-04"),
        ]
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=[summary[0]],
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-04",
            rows=[summary[1]],
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_bar",
            trade_date="2023-01-03",
            rows=[
                _minimal_bar("000001.SZ", trade_date="2023-01-03"),
                _minimal_bar("999999.SZ", trade_date="2023-01-03"),
            ],
            source_version="tushare-pro-rest",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_bar",
            trade_date="2023-01-04",
            rows=[_minimal_bar("000002.SZ", trade_date="2023-01-04")],
            source_version="tushare-pro-rest",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_basic",
            trade_date="2023-01-03",
            rows=[
                _minimal_basic("000001.SZ", trade_date="2023-01-03"),
                _minimal_basic("999999.SZ", trade_date="2023-01-03"),
            ],
            source_version="daily_basic",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_basic",
            trade_date="2023-01-04",
            rows=[_minimal_basic("000002.SZ", trade_date="2023-01-04")],
            source_version="daily_basic",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-04"
        )
        universe = build_lhb_stock_universe(inputs)
        assert universe.stock_ids == frozenset({"000001.SZ", "000002.SZ"})
        market_ids = {b.stock_id for b in inputs.market_data}
        basic_ids = {b.stock_id for b in inputs.daily_basic}
        assert "999999.SZ" not in market_ids
        assert "999999.SZ" not in basic_ids


# -- Label construction --


class TestLabelConstruction:
    def test_mature_label_has_all_five_returns(self):
        bars = [
            _bar(2023, 1, 4, 11.0, 12.0),
            _bar(2023, 1, 5, 12.0, 11.0),
            _bar(2023, 1, 6, 11.0, 13.0),
            _bar(2023, 1, 9, 13.0, 14.0),
            _bar(2023, 1, 10, 14.0, 15.0),
        ]
        labels = build_isd_labels(
            sample_id="smp-mature",
            stock_id="000001.SZ",
            entry_date=_aware(2023, 1, 4),
            horizon_end_date=_aware(2023, 1, 10),
            market_bars=bars,
            benchmark_bars=None,
            materialized_at=_aware(2023, 1, 11),
        )
        assert len(labels) == 1
        label = labels[0]
        assert label.maturity == "MATURE"
        assert label.exit_1d_return is not None
        assert label.exit_3d_return is not None
        assert label.exit_5d_return is not None
        assert label.cum_excess_1d is None
        assert label.cum_excess_3d is None
        assert label.cum_excess_5d is None
        assert label.excess_unavailable_reason is not None

    def test_unmature_tail_sample_has_null_returns(self):
        bars = [
            _bar(2023, 1, 4, 11.0, 12.0),
            _bar(2023, 1, 5, 12.0, 11.0),
        ]
        labels = build_isd_labels(
            sample_id="smp-tail",
            stock_id="000001.SZ",
            entry_date=_aware(2023, 1, 4),
            horizon_end_date=_aware(2023, 1, 10),
            market_bars=bars,
            benchmark_bars=None,
            materialized_at=_aware(2023, 1, 11),
        )
        assert len(labels) == 1
        label = labels[0]
        assert label.maturity == "UNMATURE"
        assert label.exit_1d_return is None
        assert label.exit_3d_return is None
        assert label.exit_5d_return is None
        assert label.excess_unavailable_reason is not None

    def test_missing_benchmark_keeps_excess_unavailable(self):
        bars = [
            _bar(2023, 1, 4, 11.0, 12.0),
            _bar(2023, 1, 5, 12.0, 11.0),
            _bar(2023, 1, 6, 11.0, 13.0),
            _bar(2023, 1, 9, 13.0, 14.0),
            _bar(2023, 1, 10, 14.0, 15.0),
        ]
        labels = build_isd_labels(
            sample_id="smp-no-bench",
            stock_id="000001.SZ",
            entry_date=_aware(2023, 1, 4),
            horizon_end_date=_aware(2023, 1, 10),
            market_bars=bars,
            benchmark_bars=None,
            materialized_at=_aware(2023, 1, 11),
        )
        assert len(labels) == 1
        assert labels[0].cum_excess_1d is None
        assert labels[0].cum_excess_3d is None
        assert labels[0].cum_excess_5d is None
        assert labels[0].exit_1d_return is not None


# -- Calendar construction --


class TestCalendarConstruction:
    def test_calendar_is_unique_and_sorted(self, tmp_path: Path):
        # LHB summary rows make 000001.SZ and 000002.SZ universe members.
        _write_snapshot(
            tmp_path,
            dataset="dragon_tiger_summary",
            trade_date="2023-01-03",
            rows=[
                _minimal_summary("000001.SZ", "R-A", trade_date="2023-01-03"),
                _minimal_summary("000002.SZ", "R-B", trade_date="2023-01-03"),
            ],
            source_version="top_list",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_bar",
            trade_date="2023-01-03",
            rows=[
                _minimal_bar("000002.SZ", trade_date="2023-01-03"),
                _minimal_bar("000001.SZ", trade_date="2023-01-03"),
            ],
            source_version="tushare-pro-rest",
        )
        _write_snapshot(
            tmp_path,
            dataset="daily_bar",
            trade_date="2023-01-04",
            rows=[_minimal_bar("000001.SZ", trade_date="2023-01-04")],
            source_version="tushare-pro-rest",
        )
        inputs = build_isd_inputs(
            tmp_path, start_date="2023-01-03", end_date="2023-01-04"
        )
        cal = inputs.trade_calendar
        assert list(cal) == sorted(cal)
        assert len(cal) == len(set(cal))
