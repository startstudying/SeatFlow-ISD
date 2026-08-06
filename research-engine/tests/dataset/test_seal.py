"""Dataset sealing requires complete, immutable materialization evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from seatflow.dataset.manifest import DatasetManifest, bind_materialization_evidence
from seatflow.dataset.materializer import (
    FeatureField,
    GraphNodeEvidence,
    MaterializedFeature,
    MaterializedLabel,
    materialize_dataset,
)
from seatflow.dataset.samples import PoolType, PredictionSample
from seatflow.dataset.seal import HardGate, SealReport, seal_dataset
from seatflow.labels.tradeability import TradeabilityState


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _materialized():
    day = _utc("2026-01-05T00:00:00+00:00")
    sample = PredictionSample(
        sample_id="s-001",
        cycle_id="cyc-001",
        stock_id="000001.SZ",
        prediction_date=day,
        cutoff_time=day,
        days_since_last_event=0,
        pool_type=PoolType.DIRECT,
        tradeability=TradeabilityState.TRADABLE,
        available_time=day,
        is_tradable=True,
        is_backtest_eligible=True,
    )
    return materialize_dataset(
        samples=(sample,),
        labels=(MaterializedLabel("s-001", _utc("2026-01-09T00:00:00+00:00")),),
        features=(
            MaterializedFeature("s-001", "stock_volatility_5d", "1.0", 0.2, day),
        ),
        feature_schema=(FeatureField("stock_volatility_5d", "1.0"),),
        graph_nodes=(GraphNodeEvidence("000001.SZ", "Stock"),),
        graph_edges=(),
        sample_graph_references={"s-001": ("000001.SZ",)},
        materialized_at=_utc("2026-01-12T00:00:00+00:00"),
    )


def _manifest(materialized) -> DatasetManifest:
    return bind_materialization_evidence(
        DatasetManifest(
            dataset_version="seatflow.v3.0",
            build_timestamp=_utc("2026-01-12T00:00:00+00:00"),
            source_manifest_hash="a" * 64,
            identity_policy_hash="b" * 64,
            cycle_policy_hash="c" * 64,
            feature_schema_hash=materialized.feature_schema_hash,
            label_policy_hash="e" * 64,
            split_definition_hash="f" * 64,
            graph_schema_hash=materialized.graph_schema_hash,
            row_counts={},
            node_counts={},
            edge_counts={},
            quality_gate_results={},
            artifact_uris=[],
        ),
        materialized,
    )


def test_seal_report_contains_complete_computed_gate_results() -> None:
    materialized = _materialized()
    report = seal_dataset(materialized, _manifest(materialized))

    assert isinstance(report, SealReport)
    assert report.passed is True
    assert report.violation_samples == []
    assert all(report.gate_results[gate.value] for gate in HardGate)


def test_missing_required_findings_fail_closed() -> None:
    materialized = _materialized()
    incomplete = replace(materialized, findings=())

    report = seal_dataset(incomplete, _manifest(materialized))

    assert report.passed is False
    assert report.violation_count > 0
