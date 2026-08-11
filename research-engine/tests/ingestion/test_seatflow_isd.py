"""Tests for the seatflow_isd ingestion package (Stage A1)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from seatflow.ingestion.seatflow_isd.reader import (
    load_isd_dataset, verify_manifest, ManifestVerification,
)


def _aware(year, month, day, hour=10):
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


def _seed_minimal_dataset(root):
    """Write a complete but tiny SeatFlow-ISD V1 dataset."""
    version_dir = root / "v1.0-test"
    silver = version_dir / "silver"
    gold = version_dir / "gold"
    samples_dir = version_dir / "samples"
    labels_dir = version_dir / "labels"
    manifests_dir = version_dir / "manifests"
    quality_dir = version_dir / "quality_reports"
    schemas_dir = version_dir / "schemas"
    for d in [silver, gold, samples_dir, labels_dir, manifests_dir, quality_dir, schemas_dir]:
        d.mkdir(parents=True, exist_ok=True)

    events = [{
        "event_id": "lhb-001", "stock_id": "000001.SZ",
        "trade_date": "2023-01-03T00:00:00+00:00",
        "available_time": "2023-01-03T10:00:00+00:00",
        "reason_codes": ["R-A"], "total_buy": 1.0e8, "total_sell": 5.0e7,
        "net_amount": 5.0e7, "turnover_rate": 12.5, "event_confidence": 1.0,
        "source_lineage": "tushare", "source_version": "top_list",
    }]
    (silver / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8")

    participations = [{
        "event_id": "lhb-001", "stock_id": "000001.SZ",
        "trade_date": "2023-01-03T00:00:00+00:00",
        "seat_id": "营业部A", "side": "buy", "amount": 1.0e6,
        "rank_in_side": 1, "identity_scope": "named_seat",
        "available_time": "2023-01-03T10:00:00+00:00",
        "source_lineage": "tushare", "source_version": "top_inst",
    }]
    (silver / "participations.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in participations) + "\n",
        encoding="utf-8")

    market_bars = [{
        "stock_id": "000001.SZ",
        "trade_date": f"2023-01-{day:02d}T00:00:00+00:00",
        "open": 10.0+day, "close": 11.0+day, "high": 12.0+day, "low": 9.0+day,
        "volume": 1.0e6,
        "available_time": f"2023-01-{day:02d}T10:00:00+00:00",
        "source_lineage": "tushare", "source_version": "tushare-pro-rest",
    } for day in [3, 4, 5, 6, 9, 10, 11]]
    (silver / "market_bars.jsonl").write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in market_bars) + "\n",
        encoding="utf-8")

    daily_basic = [{
        "stock_id": "000001.SZ", "trade_date": "2023-01-03T00:00:00+00:00",
        "turnover_rate": 1.0, "total_share": 1.0e6, "float_share": 1.0e6,
        "free_share": 1.0e6, "total_mv": 1.0e7, "circ_mv": 1.0e7,
        "available_time": "2023-01-03T10:00:00+00:00",
        "source_lineage": "tushare", "source_version": "daily_basic",
    }]
    (silver / "daily_basic.jsonl").write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in daily_basic) + "\n",
        encoding="utf-8")

    (gold / "lhb_stock_universe.json").write_text(json.dumps({
        "stock_ids": ["000001.SZ"],
        "first_seen_by_stock": {"000001.SZ": "2023-01-03T00:00:00+00:00"},
        "size": 1,
    }, ensure_ascii=False), encoding="utf-8")

    cycles = [{
        "cycle_id": "cyc-001", "stock_id": "000001.SZ",
        "cycle_start": "2023-01-03T00:00:00+00:00",
        "last_event_date": "2023-01-03T00:00:00+00:00",
        "cycle_end": "2023-01-10T00:00:00+00:00",
        "event_count": 1, "status": "ACTIVE",
    }]
    (gold / "cycles.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cycles) + "\n",
        encoding="utf-8")

    samples = [{
        "sample_id": "smp-001", "cycle_id": "cyc-001",
        "stock_id": "000001.SZ",
        "prediction_date": "2023-01-04T00:00:00+00:00",
        "cutoff_time": "2023-01-04T00:00:00+00:00",
        "days_since_last_event": 0, "pool_type": "DIRECT",
        "tradeability": "TRADABLE",
        "available_time": "2023-01-04T00:00:00+00:00",
        "is_tradable": True, "is_backtest_eligible": True,
    }]
    (samples_dir / "direct_samples.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8")

    labels = [{
        "sample_id": "smp-001", "stock_id": "000001.SZ",
        "entry_date": "2023-01-04T00:00:00+00:00",
        "exit_1d_date": "2023-01-05T00:00:00+00:00",
        "exit_3d_date": "2023-01-09T00:00:00+00:00",
        "exit_5d_date": "2023-01-11T00:00:00+00:00",
        "horizon_end_date": "2023-01-11T00:00:00+00:00",
        "maturity": "MATURE", "exit_1d_return": 0.05,
        "exit_3d_return": 0.10, "exit_5d_return": 0.15,
        "cum_excess_1d": None, "cum_excess_3d": None, "cum_excess_5d": None,
        "excess_unavailable_reason": "verified benchmark source not provided",
        "entry_price": 11.0, "exit_1d_price": 12.0,
        "exit_3d_price": 14.0, "exit_5d_price": 16.0,
        "materialized_at": "2023-01-15T00:00:00+00:00",
    }]
    (labels_dir / "label_rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in labels) + "\n",
        encoding="utf-8")

    (schemas_dir / "dataset_schema.json").write_text(
        json.dumps({"version": "1.0"}), encoding="utf-8")
    (quality_dir / "quality_report.json").write_text(json.dumps({
        "dataset_valid": False,
        "row_counts": {"events": 1, "participations": 1, "cycles": 1,
                        "direct_samples": 1, "spillover_samples": 0,
                        "mature_labels": 1, "unmature_labels": 0,
                        "market_bars": 7, "daily_basic": 1,
                        "industry_membership": 0},
        "date_range": {"start_date": "2023-01-03", "end_date": "2023-01-11",
                       "trading_day_count": 7},
        "source_inventory_fingerprint": "test-fp",
    }), encoding="utf-8")

    file_entries = []
    for path in sorted(version_dir.rglob("*")):
        if path.is_file() and path.parent.name != "manifests":
            rel = path.relative_to(version_dir).as_posix()
            raw_bytes = path.read_bytes()
            file_entries.append({
                "relative_path": rel, "byte_count": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            })
    manifest = {
        "dataset_version": "v1.0-test",
        "materialized_at": "2023-01-15T00:00:00+00:00",
        "row_counts": {}, "source_inventory_fingerprint": "test-fp",
        "files": file_entries, "file_count": len(file_entries),
    }
    (manifests_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class TestReader:
    def test_verify_manifest_returns_verification(self, tmp_path):
        _seed_minimal_dataset(tmp_path)
        result = verify_manifest(tmp_path / "v1.0-test")
        assert isinstance(result, ManifestVerification)
        assert result.file_count > 0
        assert result.file_hashes_verified is True
        assert result.mismatches == ()

    def test_load_isd_dataset_returns_raw_payload(self, tmp_path):
        _seed_minimal_dataset(tmp_path)
        raw = load_isd_dataset(tmp_path, "v1.0-test")
        assert raw["manifest_verified"] is True
        assert len(raw["events"]) == 1
        assert len(raw["participations"]) == 1
        assert len(raw["market_bars"]) == 7
        assert len(raw["daily_basic"]) == 1
        assert len(raw["cycles"]) == 1
        assert len(raw["samples"]) == 1
        assert len(raw["labels"]) == 1

class TestLoader:
    def test_load_isd_v1_returns_report(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_isd_v1, ISDLoadReport
        # Reuse the seeder from above.
        _seed_minimal_dataset(tmp_path)
        report = load_isd_v1(tmp_path, "v1.0-test")
        assert isinstance(report, ISDLoadReport)
        assert report.result.manifest_hash_verified is True
        assert report.validation_passed is True
        assert report.validation_issue_count == 0
        assert report.result.universe.size == 1
        assert report.result.row_counts["events"] == 1

    def test_load_isd_v1_strict_raises_on_tamper(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_isd_v1
        _seed_minimal_dataset(tmp_path)
        # Tamper with a file.
        (tmp_path / "v1.0-test" / "silver" / "events.jsonl").write_text(
            "[]\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="hash mismatch"):
            load_isd_v1(tmp_path, "v1.0-test")

    def test_load_isd_v1_missing_version(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_isd_v1
        with pytest.raises(FileNotFoundError):
            load_isd_v1(tmp_path, "v0.0-missing")

    def test_load_isd_v1_strict_raises_on_validation_issue(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_isd_v1
        _seed_minimal_dataset(tmp_path)
        # Corrupt the sample to introduce a SPILLOVER sample (V1 must be DIRECT only).
        samples_path = tmp_path / "v1.0-test" / "samples" / "direct_samples.jsonl"
        sample = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
        sample["pool_type"] = "SPILLOVER"
        samples_path.write_text(
            json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Rebuild the manifest with the new file hashes so the loader
        # can surface a validator-level issue rather than a tamper
        # error.
        version_dir = tmp_path / "v1.0-test"
        file_entries = []
        for path in sorted(version_dir.rglob("*")):
            if path.is_file() and path.parent.name != "manifests":
                rel = path.relative_to(version_dir).as_posix()
                raw_bytes = path.read_bytes()
                file_entries.append({
                    "relative_path": rel, "byte_count": len(raw_bytes),
                    "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                })
        manifest_path = version_dir / "manifests" / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = file_entries
        manifest["file_count"] = len(file_entries)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = load_isd_v1(tmp_path, "v1.0-test", strict=False)
        assert report.validation_passed is False
        with pytest.raises(ValueError, match="validation failed"):
            load_isd_v1(tmp_path, "v1.0-test", strict=True)


class TestBenchmark:
    def _seed_csi500(self, root: Path, n_days: int = 5) -> None:
        dataset_dir = root / "CSI500"
        for i in range(n_days):
            snap_dir = dataset_dir / f"date=2023-01-{3+i:02d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "benchmark_id": "CSI500",
                "trade_date": f"2023-01-{3+i:02d}T00:00:00+00:00",
                "open": 100.0 + i,
                "close": 101.0 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "volume": 1.0e8,
                "available_time": f"2023-01-{3+i:02d}T10:00:00+00:00",
                "source_name": "tushare",
                "source_version": "index_daily",
            }
            raw = json.dumps([row], ensure_ascii=False).encode("utf-8")
            (snap_dir / "data.json").write_bytes(raw)
            manifest = {
                "dataset": "CSI500", "date": f"2023-01-{3+i:02d}",
                "provider": "tushare", "row_count": 1,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_version": "index_daily",
            }
            (snap_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

    def test_load_benchmark_returns_sorted_bars(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import (
            load_benchmark, to_isd_market_bars,
        )
        self._seed_csi500(tmp_path, n_days=5)
        bars, source = load_benchmark(tmp_path, "CSI500")
        assert len(bars) == 5
        assert all(b.benchmark_id == "CSI500" for b in bars)
        assert source.benchmark_id == "CSI500"
        assert source.date_count == 5
        # Sorted by trade_date ascending.
        dates = [b.trade_date for b in bars]
        assert dates == sorted(dates)
        # Converts to ISDMarketBar so it plugs into compute_market_excess_path.
        isd_bars = to_isd_market_bars(bars)
        assert all(b.stock_id == "CSI500" for b in isd_bars)

    def test_load_benchmark_respects_date_filter(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_benchmark
        self._seed_csi500(tmp_path, n_days=5)
        bars, _ = load_benchmark(
            tmp_path, "CSI500",
            start_date="2023-01-04",
            end_date="2023-01-06",
        )
        assert len(bars) == 3
        assert bars[0].trade_date.date().isoformat() == "2023-01-04"
        assert bars[-1].trade_date.date().isoformat() == "2023-01-06"

    def test_load_benchmark_missing_raises(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_benchmark
        with pytest.raises(FileNotFoundError, match="no verified snapshots"):
            load_benchmark(tmp_path, "CSI500")

    def test_benchmark_bar_rejects_non_positive(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import BenchmarkBar
        with pytest.raises(ValueError, match="strictly positive"):
            BenchmarkBar(
                benchmark_id="CSI500", stock_id="CSI500",
                trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                open=0.0, close=101.0, high=102.0, low=99.0,
                volume=1.0,
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
                source_lineage="tushare", source_version="index_daily",
            )

    def test_compute_excess_returns_with_real_benchmark(self, tmp_path):
        """End-to-end: stock prices vs CSI500 produces non-null excess."""
        from seatflow.labels.returns import compute_market_excess_path
        # Use a 5-day seed where the stock clearly diverges from CSI500
        # day-1 (stock +5%, benchmark +1%) so cum_excess_1d > 0.
        from seatflow.ingestion.seatflow_isd.benchmark import BenchmarkBar
        from datetime import datetime, timezone
        # Benchmark: closes 100, 101, 102, 103, 104 (linear +1/day).
        market_bars = [
            BenchmarkBar(
                benchmark_id="CSI500", stock_id="CSI500",
                trade_date=datetime(2023, 1, day, tzinfo=timezone.utc),
                open=100.0 + day - 1, close=100.0 + day,
                high=100.0 + day, low=100.0 + day - 1,
                volume=1.0,
                available_time=datetime(2023, 1, day, 10, tzinfo=timezone.utc),
                source_lineage="tushare", source_version="index_daily",
            )
            for day in [3, 4, 5, 6, 9]
        ]
        # Stock: closes 10, 11, 12, 13, 14 (also linear, but day-1 is +10%).
        stock_bars = [
            {"date": b.trade_date, "open": b.open / 10.0, "close": b.close / 10.0}
            for b in market_bars
        ]
        market_prices_df = [
            {"date": b.trade_date, "open": b.open, "close": b.close}
            for b in market_bars
        ]
        path = compute_market_excess_path(
            stock_id="X",
            market_id="CSI500",
            entry_date=market_bars[0].trade_date,
            prices_df=stock_bars,
            market_prices_df=market_prices_df,
        )
        # Day-1 stock: 1.1/1.0 - 1 = 0.10. Day-1 benchmark: 101/100 - 1 = 0.01.
        # Excess: 0.10 - 0.01 = 0.09.
        assert path.cum_excess_1d > 0
        assert path.cum_excess_5d > 0


class TestAdjustments:
    def _seed_adj_factor(self, root, n_days=3):
        d = root / "adj_factor"
        for i in range(n_days):
            snap = d / f"date=2023-01-{3+i:02d}"
            snap.mkdir(parents=True, exist_ok=True)
            row = {
                "stock_id": "000001.SZ",
                "trade_date": f"2023-01-{3+i:02d}T00:00:00+00:00",
                "factor": 1.0 + 0.1 * i,
                "available_time": f"2023-01-{3+i:02d}T10:00:00+00:00",
            }
            raw = json.dumps([row], ensure_ascii=False).encode("utf-8")
            (snap / "data.json").write_bytes(raw)
            (snap / "manifest.json").write_text(json.dumps({
                "dataset": "adj_factor", "date": f"2023-01-{3+i:02d}",
                "row_count": 1, "sha256": hashlib.sha256(raw).hexdigest(),
            }), encoding="utf-8")

    def _seed_company_actions(self, root, n_days=2):
        d = root / "company_actions"
        for i in range(n_days):
            snap = d / f"date=2023-01-{3+i:02d}"
            snap.mkdir(parents=True, exist_ok=True)
            row = {
                "stock_id": "000001.SZ",
                "ex_date": f"2023-01-{3+i:02d}T00:00:00+00:00",
                "trade_date": f"2023-01-{3+i:02d}T00:00:00+00:00",
                "action_type": "dividend",
                "amount": 0.5,
                "available_time": f"2023-01-{3+i:02d}T10:00:00+00:00",
            }
            raw = json.dumps([row], ensure_ascii=False).encode("utf-8")
            (snap / "data.json").write_bytes(raw)
            (snap / "manifest.json").write_text(json.dumps({
                "dataset": "company_actions", "date": f"2023-01-{3+i:02d}",
                "row_count": 1, "sha256": hashlib.sha256(raw).hexdigest(),
            }), encoding="utf-8")

    def test_load_adj_factor(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_adj_factor
        self._seed_adj_factor(tmp_path, n_days=3)
        rows, source = load_adj_factor(tmp_path)
        assert len(rows) == 3
        assert source.verified is True
        assert all(r.factor > 0 for r in rows)

    def test_load_company_actions(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_company_actions
        self._seed_company_actions(tmp_path, n_days=2)
        rows, source = load_company_actions(tmp_path)
        assert len(rows) == 2
        assert all(r.action_type == "dividend" for r in rows)
        assert source.verified is True

    def test_adjust_price_multiplies(self):
        from seatflow.ingestion.seatflow_isd import adjust_price
        assert adjust_price(10.0, 1.5) == 15.0
        assert adjust_price(10.0, 1.0) == 10.0

    def test_adj_factor_rejects_zero(self):
        from seatflow.ingestion.seatflow_isd import AdjFactorRow
        from datetime import datetime, timezone
        with pytest.raises(ValueError, match="must be > 0"):
            AdjFactorRow(
                stock_id="X", trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                factor=0.0,
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
            )


class TestTradeability:
    def _seed(self, root, n_days=3):
        d = root / "tradeability"
        for i in range(n_days):
            snap = d / f"date=2023-01-{3+i:02d}"
            snap.mkdir(parents=True, exist_ok=True)
            row = {
                "stock_id": "000001.SZ",
                "trade_date": f"2023-01-{3+i:02d}T00:00:00+00:00",
                "suspended": i == 0,
                "limit_up": i == 1,
                "limit_down": False,
                "one_price": i == 1,
                "available_time": f"2023-01-{3+i:02d}T10:00:00+00:00",
            }
            raw = json.dumps([row], ensure_ascii=False).encode("utf-8")
            (snap / "data.json").write_bytes(raw)
            (snap / "manifest.json").write_text(json.dumps({
                "dataset": "tradeability", "date": f"2023-01-{3+i:02d}",
                "row_count": 1, "sha256": hashlib.sha256(raw).hexdigest(),
            }), encoding="utf-8")

    def test_load_tradeability(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_tradeability
        self._seed(tmp_path, n_days=3)
        rows, source = load_tradeability(tmp_path)
        assert len(rows) == 3
        assert source.verified is True
        suspended_count = sum(1 for r in rows if r.suspended)
        assert suspended_count == 1
        limit_up_count = sum(1 for r in rows if r.limit_up)
        assert limit_up_count == 1

    def test_tradeability_rejects_both_limits(self):
        from seatflow.ingestion.seatflow_isd import TradeabilityRow
        from datetime import datetime, timezone
        with pytest.raises(ValueError, match="both"):
            TradeabilityRow(
                stock_id="X", trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                suspended=False, limit_up=True, limit_down=True, one_price=False,
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
            )


class TestIndustryAdapter:
    def _seed(self, root):
        """Seed the sw2021_l1_industry_membership layout."""
        d = root / "sw2021_l1_industry_membership"
        l1_a = d / "l1_code=801010.SI"
        l1_b = d / "l1_code=801030.SI"
        for l1_dir, l1_name, is_new, stocks in [
            (l1_a, "农林牧渔", "N", [
                ("000001.SZ", "平安银行", "2023-01-01", None),
                ("000002.SZ", "万科A", "2023-01-01", None),
            ]),
            (l1_b, "有色金属", "N", [
                ("600000.SH", "浦发银行", "2023-01-01", None),
            ]),
        ]:
            vdir = l1_dir / f"is_new={is_new}"
            vdir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "l1_code": l1_dir.name.split("=", 1)[1],
                    "l1_name": l1_name,
                    "ts_code": s[0], "name": s[1],
                    "in_date": s[2], "out_date": s[3],
                    "available_time": "2023-01-01T00:00:00+00:00",
                }
                for s in stocks
            ]
            raw = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            (vdir / "data.json").write_bytes(raw)
            (vdir / "manifest.json").write_text(json.dumps({
                "dataset": "sw2021_l1_industry_membership",
                "date": "2023-01-01",
                "row_count": len(rows),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }), encoding="utf-8")

    def test_load_industry_membership(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_industry_membership
        self._seed(tmp_path)
        rows, source = load_industry_membership(tmp_path)
        assert len(rows) == 3
        assert source.industry_count == 2
        assert source.verified is True
        # Sorted by (industry_code, stock_id).
        codes = [r.industry_code for r in rows]
        assert codes == sorted(codes)

    def test_industry_to_isd_row(self, tmp_path):
        from seatflow.ingestion.seatflow_isd import load_industry_membership
        self._seed(tmp_path)
        rows, _ = load_industry_membership(tmp_path)
        isd_row = rows[0].to_isd_row()
        assert isd_row.industry_code == rows[0].industry_code
        assert isd_row.source_version == "sw2021_l1_industry_membership"


class TestProxyAblation:
    def _make_participations(self, n_named=3, n_proxy=2, n_unknown=1):
        from seatflow.ingestion.seatflow_isd import ISDSeatParticipationV3
        from datetime import datetime, timezone
        parts = []
        for i in range(n_named):
            parts.append(ISDSeatParticipationV3(
                event_id=f"e{i}", stock_id=f"00000{i}.SZ",
                trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                seat_id=f"named-{i}", side="BUY", amount=1.0,
                rank_in_side=i + 1,
                identity_scope="named_seat",
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
                source_lineage="tushare", source_version="top_inst",
            ))
        for i in range(n_proxy):
            parts.append(ISDSeatParticipationV3(
                event_id=f"e{i}", stock_id=f"00001{i}.SZ",
                trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                seat_id=f"proxy-{i}", side="SELL", amount=0.5,
                rank_in_side=i + 1,
                identity_scope="institutional_flow_proxy",
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
                source_lineage="tushare", source_version="top_inst",
            ))
        for i in range(n_unknown):
            parts.append(ISDSeatParticipationV3(
                event_id=f"eu{i}", stock_id=f"99999{i}.SZ",
                trade_date=datetime(2023, 1, 3, tzinfo=timezone.utc),
                seat_id=f"unknown-{i}", side="BUY", amount=0.1,
                rank_in_side=i + 1,
                identity_scope="unknown",
                available_time=datetime(2023, 1, 3, 10, tzinfo=timezone.utc),
                source_lineage="tushare", source_version="top_inst",
            ))
        return parts

    def test_e0_keeps_only_named(self):
        from seatflow.ingestion.seatflow_isd import (
            scheme_e0_named_only,
        )
        parts = self._make_participations(n_named=3, n_proxy=2, n_unknown=1)
        s = scheme_e0_named_only(parts)
        assert s.name == "E0_named_only"
        assert s.keep_count == 3
        assert s.drop_count == 3  # 2 proxy + 1 unknown

    def test_e1_keeps_everything(self):
        from seatflow.ingestion.seatflow_isd import scheme_e1_named_plus_proxy
        parts = self._make_participations(n_named=3, n_proxy=2, n_unknown=1)
        s = scheme_e1_named_plus_proxy(parts)
        assert s.keep_count == 6
        assert s.drop_count == 0

    def test_e2_keeps_only_unknown(self):
        from seatflow.ingestion.seatflow_isd import scheme_e2_anonymous_only
        parts = self._make_participations(n_named=3, n_proxy=2, n_unknown=1)
        s = scheme_e2_anonymous_only(parts)
        assert s.keep_count == 1
        assert s.drop_count == 5

    def test_e3_shuffled_keeps_all(self):
        from seatflow.ingestion.seatflow_isd import scheme_e3_shuffled_seats
        parts = self._make_participations(n_named=3, n_proxy=2, n_unknown=1)
        s = scheme_e3_shuffled_seats(parts)
        assert s.keep_count == 6
        assert s.drop_count == 0
        assert "leakage baseline" in s.description

    def test_e4_keeps_only_proxy(self):
        from seatflow.ingestion.seatflow_isd import scheme_e4_proxy_only
        parts = self._make_participations(n_named=3, n_proxy=2, n_unknown=1)
        s = scheme_e4_proxy_only(parts)
        assert s.keep_count == 2
        assert s.drop_count == 4  # 3 named + 1 unknown

    def test_build_all_schemes_returns_five(self):
        from seatflow.ingestion.seatflow_isd import build_all_schemes
        parts = self._make_participations()
        schemes = build_all_schemes(parts)
        assert len(schemes) == 5
        assert [s.name for s in schemes] == [
            "E0_named_only", "E1_named_plus_proxy",
            "E2_anonymous_only", "E3_shuffled_seats",
            "E4_proxy_only",
        ]
        # E1 + E3 keep everything, E2 keeps 1 (unknown), E0 keeps 3, E4 keeps 2.
        keep_counts = sorted(s.keep_count for s in schemes)
        assert keep_counts == [1, 2, 3, 6, 6]

    def test_e0_matches_real_v1_counts(self):
        """Verify that E0 against the real v1.0-20260806 dataset drops
        exactly 56,493 rows (the documented institutional_flow_proxy
        count). This is the gate that proves the real data still has
        the documented shape."""
        from seatflow.ingestion.seatflow_isd import load_isd_v1
        from seatflow.ingestion.seatflow_isd.proxy_ablation import (
            scheme_e0_named_only, scheme_e1_named_plus_proxy,
        )
        report = load_isd_v1(
            r"D:\SeatFlow\datasets\seatflow_isd_v1",
            "v1.0-20260806",
        )
        e0 = scheme_e0_named_only(report.result.participations)
        e1 = scheme_e1_named_plus_proxy(report.result.participations)
        # Both schemes must observe the same total.
        total = e0.keep_count + e0.drop_count
        assert total == e1.keep_count == 696549
        # The documented proxy count must match.
        assert e0.drop_count == 56493
        assert e0.keep_count == 640056
