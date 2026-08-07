"""Tests for the SeatFlow-ISD build/audit CLI (Task 5)."""

from __future__ import annotations

import hashlib
import json

from seatflow.dataset.isd_cli import main


def _write_snapshot(root, *, dataset, trade_date, rows, source_version="top_list"):
    dataset_dir = root / dataset
    snap_dir = dataset_dir / ("date=" + trade_date)
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
    (snap_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _seed_minimal_source(root):
    summary = [{
        "stock_id": "000001.SZ", "stock_name": "Test",
        "trade_date": "2023-01-03", "reason": "R-A",
        "buy_amount": 1.0e8, "sell_amount": 5.0e7, "net_amount": 5.0e7,
        "available_time": "2023-01-03T18:00:00+08:00",
        "source_name": "tushare", "source_version": "top_list",
    }]
    seat = [{
        "stock_id": "000001.SZ", "trade_date": "2023-01-03",
        "event_reason": "R-A", "side": "buy", "amount": 1.0e6,
        "seat_id": "A", "identity_scope": "named_seat",
        "available_time": "2023-01-03T18:00:00+08:00",
        "source_name": "tushare", "source_version": "top_inst",
    }]
    bars = [
        {"stock_id": "000001.SZ", "trade_date": d,
         "open_price": str(o), "close_price": str(c),
         "high_price": str(max(o, c)), "low_price": str(min(o, c)),
         "pre_close": str(o), "volume": "1000000.0", "amount": "10000.0",
         "amount_unit": "thousand_cny", "volume_unit": "hand",
         "available_time": d + "T18:00:00+08:00",
         "source_name": "tushare", "source_version": "tushare-pro-rest"}
        for d, o, c in [
            ("2023-01-03", 10.0, 11.0), ("2023-01-04", 11.0, 12.0),
            ("2023-01-05", 12.0, 11.0), ("2023-01-06", 11.0, 13.0),
            ("2023-01-09", 13.0, 14.0), ("2023-01-10", 14.0, 15.0),
            ("2023-01-11", 15.0, 16.0),
        ]
    ]
    basic = [
        {"stock_id": "000001.SZ", "trade_date": d, "turnover_rate": 1.0,
         "turnover_rate_f": 1.0, "volume_ratio": 1.0, "total_share": 1.0e6,
         "float_share": 1.0e6, "free_share": 1.0e6, "total_mv": 1.0e7,
         "circ_mv": 1.0e7, "available_time": d + "T18:00:00+08:00",
         "source_name": "tushare", "source_version": "daily_basic"}
        for d in ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    ]
    _write_snapshot(root, dataset="dragon_tiger_summary", trade_date="2023-01-03",
                    rows=summary, source_version="top_list")
    _write_snapshot(root, dataset="dragon_tiger_seat_event", trade_date="2023-01-03",
                    rows=seat, source_version="top_inst")
    for bar in bars:
        _write_snapshot(root, dataset="daily_bar", trade_date=bar["trade_date"],
                        rows=[bar], source_version="tushare-pro-rest")
    for b in basic:
        _write_snapshot(root, dataset="daily_basic", trade_date=b["trade_date"],
                        rows=[b], source_version="daily_basic")


class TestBuildCommand:
    def test_build_outputs_json_and_exit_0(self, tmp_path, capsys):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        rc = main([
            "build",
            "--source-root", str(source),
            "--output-root", str(output),
            "--dataset-version", "v1.0-cli",
            "--materialized-at", "2023-01-15T00:00:00+00:00",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["dataset_version"] == "v1.0-cli"
        assert "manifest_hash" in payload
        assert "dataset_valid" in payload
        assert "quality_report" in payload

    def test_build_missing_source_exit_2(self, tmp_path, capsys):
        rc = main([
            "build",
            "--source-root", str(tmp_path / "nonexistent"),
            "--output-root", str(tmp_path / "out"),
            "--dataset-version", "v1.0-bad",
        ])
        assert rc == 2

    def test_build_require_formal_exit_3(self, tmp_path, capsys):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        rc = main([
            "build",
            "--source-root", str(source),
            "--output-root", str(output),
            "--dataset-version", "v1.0-formal",
            "--materialized-at", "2023-01-15T00:00:00+00:00",
            "--require-formal",
        ])
        assert rc == 3
        captured = capsys.readouterr()
        assert "formal validity blocked" in captured.err


class TestAuditCommand:
    def test_audit_exit_0(self, tmp_path, capsys):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        main([
            "build",
            "--source-root", str(source),
            "--output-root", str(output),
            "--dataset-version", "v1.0-audit",
            "--materialized-at", "2023-01-15T00:00:00+00:00",
        ])
        capsys.readouterr()
        ds_dir = output / "v1.0-audit"
        rc = main(["audit", "--dataset-root", str(ds_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["manifest_exists"] is True
        assert payload["manifest_hash_verified"] is True

    def test_audit_missing_root_exit_2(self, tmp_path):
        rc = main(["audit", "--dataset-root", str(tmp_path / "nonexistent")])
        assert rc == 2

    def test_audit_require_formal_exit_3(self, tmp_path, capsys):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        main([
            "build",
            "--source-root", str(source),
            "--output-root", str(output),
            "--dataset-version", "v1.0-reqformal",
            "--materialized-at", "2023-01-15T00:00:00+00:00",
        ])
        capsys.readouterr()
        ds_dir = output / "v1.0-reqformal"
        rc = main([
            "audit", "--dataset-root", str(ds_dir), "--require-formal",
        ])
        assert rc == 3
