"""Tests for the ISD dataset orchestrator (Task 4)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from seatflow.dataset.isd_builder import (
    ISDAuditResult,
    ISDBuildConfig,
    ISDBuildResult,
    audit_isd_dataset,
    build_isd_dataset,
)


def _aware(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


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
        "seat_id": "营业部A", "identity_scope": "named_seat",
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


class TestBuild:
    def test_writes_complete_directory_contract(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-test", materialized_at=_aware(2023, 1, 15))
        result = build_isd_dataset(config)
        assert isinstance(result, ISDBuildResult)
        ds_dir = result.output_dir
        assert (ds_dir / "bronze" / "source_inventory.json").exists()
        assert (ds_dir / "silver" / "events.jsonl").exists()
        assert (ds_dir / "silver" / "participations.jsonl").exists()
        assert (ds_dir / "silver" / "market_bars.jsonl").exists()
        assert (ds_dir / "silver" / "daily_basic.jsonl").exists()
        assert (ds_dir / "gold" / "lhb_stock_universe.json").exists()
        assert (ds_dir / "gold" / "cycles.jsonl").exists()
        assert (ds_dir / "samples" / "direct_samples.jsonl").exists()
        assert (ds_dir / "splits" / "walk_forward_plan.json").exists()
        assert (ds_dir / "schemas" / "dataset_schema.json").exists()
        assert (ds_dir / "manifests" / "dataset_manifest.json").exists()
        assert (ds_dir / "quality_reports" / "quality_report.json").exists()

    def test_no_formal_seal_when_blockers_present(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-fail", materialized_at=_aware(2023, 1, 15))
        result = build_isd_dataset(config)
        assert result.dataset_valid is False
        assert not (result.output_dir / "dataset_seal.json").exists()
        assert (result.output_dir / "manifests" / "dataset_manifest.json").exists()
        assert len(result.quality_report.formal_blockers) > 0

    def test_no_spillover_in_output(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-nosp", materialized_at=_aware(2023, 1, 15))
        result = build_isd_dataset(config)
        assert result.quality_report.spillover_count == 0
        assert result.quality_report.row_counts["spillover_samples"] == 0

    def test_atomic_write_leaves_existing_intact(self, tmp_path, monkeypatch):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-atomic", materialized_at=_aware(2023, 1, 15))
        result1 = build_isd_dataset(config)
        original_hash = result1.manifest_hash
        from seatflow.dataset import isd_builder
        def _fail(*a, **k):
            raise RuntimeError("simulated write failure")
        monkeypatch.setattr(isd_builder, "_write_dataset", _fail)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            build_isd_dataset(config)
        manifest_path = result1.output_dir / "manifests" / "dataset_manifest.json"
        current_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert current_hash == original_hash


class TestAudit:
    def test_audit_recomputes_hashes(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-audit", materialized_at=_aware(2023, 1, 15))
        result = build_isd_dataset(config)
        audit = audit_isd_dataset(result.output_dir)
        assert isinstance(audit, ISDAuditResult)
        assert audit.manifest_exists is True
        assert audit.manifest_hash_verified is True
        assert len(audit.hash_mismatches) == 0
        assert audit.file_count > 0

    def test_audit_detects_tampered_file(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        _seed_minimal_source(source)
        output = tmp_path / "out"
        config = ISDBuildConfig(
            source_root=source, output_root=output,
            dataset_version="v1.0-tamper", materialized_at=_aware(2023, 1, 15))
        result = build_isd_dataset(config)
        target = result.output_dir / "gold" / "lhb_stock_universe.json"
        target.write_text('{"tampered": true}', encoding="utf-8")
        audit = audit_isd_dataset(result.output_dir)
        assert audit.manifest_hash_verified is False
        assert any("lhb_stock_universe.json" in m for m in audit.hash_mismatches)
