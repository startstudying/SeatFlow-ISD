"""Behavioral tests for computed Dataset v3 materialization gates."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import nan

import pytest

import seatflow.dataset.materializer as materializer_module
from seatflow.dataset.manifest import DatasetManifest, bind_materialization_evidence
from seatflow.dataset.materializer import (
    FeatureField,
    GraphEdgeEvidence,
    GraphNodeEvidence,
    MaterializedFeature,
    MaterializedLabel,
    compute_feature_schema_hash,
    materialize_dataset,
)
from seatflow.dataset.samples import PoolType, PredictionSample
from seatflow.dataset.seal import HardGate, seal_dataset
from seatflow.graph.schema import EDGE_SCHEMAS, NODE_SCHEMAS, EdgeType, NodeType
from seatflow.labels.tradeability import TradeabilityState


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sample(
    sample_id: str = "s-001",
    stock_id: str = "000001.SZ",
    *,
    pool_type: PoolType = PoolType.DIRECT,
    prediction_date: str = "2026-01-05T00:00:00+00:00",
) -> PredictionSample:
    prediction_time = _utc(prediction_date)
    return PredictionSample(
        sample_id=sample_id,
        cycle_id="cyc-001",
        stock_id=stock_id,
        prediction_date=prediction_time,
        cutoff_time=prediction_time,
        days_since_last_event=0,
        pool_type=pool_type,
        tradeability=TradeabilityState.TRADABLE,
        available_time=prediction_time,
        is_tradable=True,
        is_backtest_eligible=True,
    )


SCHEMA = (FeatureField(name="stock_volatility_5d", version="1.0"),)


def _materialize(
    *,
    samples: tuple[PredictionSample, ...] | None = None,
    labels: tuple[MaterializedLabel, ...] | None = None,
    features: tuple[MaterializedFeature, ...] | None = None,
    nodes: tuple[GraphNodeEvidence, ...] | None = None,
    edges: tuple[GraphEdgeEvidence, ...] = (),
    references: dict[str, tuple[str, ...]] | None = None,
    schema: tuple[FeatureField, ...] = SCHEMA,
    trade_calendar: tuple[datetime, ...] | None = None,
    materialized_at: datetime | None = None,
):
    sample = _sample() if samples is None else samples[0]
    actual_samples = (_sample(),) if samples is None else samples
    actual_labels = labels if labels is not None else (
        MaterializedLabel(
            sample_id=sample.sample_id,
            evaluation_time=_utc("2026-01-09T00:00:00+00:00"),
            horizon_end_time=_utc("2026-01-09T00:00:00+00:00"),
        ),
    )
    actual_features = features if features is not None else (
        MaterializedFeature(
            sample_id=sample.sample_id,
            name="stock_volatility_5d",
            version="1.0",
            value=0.2,
            available_time=sample.cutoff_time,
        ),
    )
    actual_nodes = nodes if nodes is not None else (
        GraphNodeEvidence(node_id=sample.stock_id, node_type="Stock"),
    )
    actual_references = (
        references if references is not None else {sample.sample_id: (sample.stock_id,)}
    )
    actual_calendar = trade_calendar or tuple(
        _utc(f"2026-01-{day:02d}T00:00:00+00:00") for day in range(5, 10)
    )
    return materialize_dataset(
        samples=actual_samples,
        labels=actual_labels,
        features=actual_features,
        feature_schema=schema,
        graph_nodes=actual_nodes,
        graph_edges=edges,
        sample_graph_references=actual_references,
        trade_calendar=actual_calendar,
        materialized_at=materialized_at
        or _utc("2026-01-12T00:00:00+00:00"),
    )


def _manifest(materialized) -> DatasetManifest:
    skeleton = DatasetManifest(
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
    )
    return bind_materialization_evidence(skeleton, materialized)


def _failed(materialized, gate: HardGate):
    matches = [finding for finding in materialized.findings if finding.gate is gate]
    assert matches, f"missing evidence for {gate.value}"
    assert any(not finding.passed for finding in matches)
    return matches


def test_valid_materialization_seals_from_computed_evidence() -> None:
    materialized = _materialize()
    report = seal_dataset(materialized, _manifest(materialized))

    assert report.passed is True
    assert report.violation_count == 0
    assert {sample.pool_type for sample in materialized.evaluation_samples} == {
        PoolType.DIRECT
    }


def test_legacy_boolean_gate_spoofing_is_not_a_public_seal_path() -> None:
    materialized = _materialize()

    with pytest.raises(TypeError):
        seal_dataset(  # type: ignore[call-arg]
            materialized.evaluation_samples,
            _manifest(materialized),
            {gate: True for gate in HardGate},
        )


def test_replaced_findings_cannot_spoof_computed_gate_evidence() -> None:
    sample = _sample()
    failing = _materialize(
        features=(
            MaterializedFeature(
                sample_id=sample.sample_id,
                name="stock_volatility_5d",
                version="1.0",
                value=0.2,
                available_time=_utc("2026-01-06T00:00:00+00:00"),
            ),
        )
    )
    forged = replace(
        failing,
        findings=tuple(replace(finding, passed=True) for finding in failing.findings),
    )

    report = seal_dataset(forged, _manifest(forged))

    assert report.passed is False
    assert report.gate_results[HardGate.PIT_AVAILABLE_TIME_VIOLATION.value] is False


def test_future_feature_available_time_fails_pit_gate() -> None:
    sample = _sample()
    materialized = _materialize(
        features=(
            MaterializedFeature(
                sample_id=sample.sample_id,
                name="stock_volatility_5d",
                version="1.0",
                value=0.2,
                available_time=_utc("2026-01-06T00:00:00+00:00"),
            ),
        )
    )

    findings = _failed(materialized, HardGate.PIT_AVAILABLE_TIME_VIOLATION)
    assert findings[0].row_reference == "s-001:stock_volatility_5d"
    assert seal_dataset(materialized, _manifest(materialized)).passed is False


def test_label_not_mature_at_materialization_time_fails_gate() -> None:
    materialized = _materialize(
        labels=(
            MaterializedLabel(
                sample_id="s-001",
                evaluation_time=_utc("2026-01-13T00:00:00+00:00"),
                horizon_end_time=_utc("2026-01-09T00:00:00+00:00"),
            ),
        )
    )

    _failed(materialized, HardGate.IMMATURE_LABEL)
    assert seal_dataset(materialized, _manifest(materialized)).passed is False


def test_same_day_label_cannot_claim_a_complete_five_day_horizon() -> None:
    materialized = _materialize(
        labels=(
            MaterializedLabel(
                sample_id="s-001",
                evaluation_time=_utc("2026-01-05T00:00:00+00:00"),
                horizon_end_time=_utc("2026-01-09T00:00:00+00:00"),
            ),
        )
    )

    _failed(materialized, HardGate.IMMATURE_LABEL)


def test_label_fails_before_derived_horizon_end_is_materialized() -> None:
    materialized = _materialize(
        labels=(
            MaterializedLabel(
                sample_id="s-001",
                evaluation_time=_utc("2026-01-08T00:00:00+00:00"),
                horizon_end_time=_utc("2026-01-09T00:00:00+00:00"),
            ),
        ),
        materialized_at=_utc("2026-01-08T00:00:00+00:00"),
    )

    _failed(materialized, HardGate.IMMATURE_LABEL)


def test_d0_plus_four_trading_day_label_matures_across_weekend() -> None:
    sample = _sample(prediction_date="2026-01-09T00:00:00+00:00")
    calendar = tuple(
        _utc(value)
        for value in (
            "2026-01-09T00:00:00+00:00",
            "2026-01-12T00:00:00+00:00",
            "2026-01-13T00:00:00+00:00",
            "2026-01-14T00:00:00+00:00",
            "2026-01-15T00:00:00+00:00",
        )
    )
    mature_time = _utc("2026-01-15T00:00:00+00:00")
    materialized = _materialize(
        samples=(sample,),
        labels=(MaterializedLabel("s-001", mature_time, mature_time),),
        trade_calendar=calendar,
        materialized_at=mature_time,
    )

    assert seal_dataset(materialized, _manifest(materialized)).passed is True


def test_tampered_label_horizon_end_cannot_self_certify_maturity() -> None:
    materialized = _materialize(
        labels=(
            MaterializedLabel(
                sample_id="s-001",
                evaluation_time=_utc("2026-01-09T00:00:00+00:00"),
                horizon_end_time=_utc("2026-01-08T00:00:00+00:00"),
            ),
        )
    )

    findings = _failed(materialized, HardGate.IMMATURE_LABEL)
    assert any(
        ("reason", "horizon_end_time does not match frozen trading calendar")
        in finding.evidence
        for finding in findings
    )
    assert seal_dataset(materialized, _manifest(materialized)).passed is False


def test_missing_label_horizon_evidence_fails_closed() -> None:
    materialized = _materialize(
        labels=(
            MaterializedLabel(
                sample_id="s-001",
                evaluation_time=_utc("2026-01-09T00:00:00+00:00"),
                horizon_end_time=None,
            ),
        )
    )

    _failed(materialized, HardGate.IMMATURE_LABEL)
    assert seal_dataset(materialized, _manifest(materialized)).passed is False


def test_duplicate_sample_primary_key_fails_gate() -> None:
    duplicate = _sample()
    materialized = _materialize(samples=(duplicate, duplicate))

    _failed(materialized, HardGate.PRIMARY_KEY_CONFLICT)


def test_dangling_sample_graph_reference_fails_gate() -> None:
    materialized = _materialize(references={"s-001": ("missing-stock",)})

    findings = _failed(materialized, HardGate.DANGLING_GRAPH_REFERENCE)
    assert any(("node_id", "missing-stock") in finding.evidence for finding in findings)


def test_graph_edge_with_missing_endpoint_fails_topology_gate() -> None:
    materialized = _materialize(
        edges=(
            GraphEdgeEvidence(
                edge_id="edge-1",
                edge_type="spillover_candidate",
                source_id="000001.SZ",
                target_id="missing-stock",
                available_time=_utc("2026-01-05T00:00:00+00:00"),
            ),
        )
    )

    _failed(materialized, HardGate.GRAPH_TOPOLOGY_VIOLATION)


def test_graph_edge_endpoint_types_must_match_frozen_topology() -> None:
    materialized = _materialize(
        nodes=(
            GraphNodeEvidence("000001.SZ", "Stock"),
            GraphNodeEvidence("000002.SZ", "Stock"),
        ),
        edges=(
            GraphEdgeEvidence(
                edge_id="edge-1",
                edge_type="event_impacts_stock",
                source_id="000001.SZ",
                target_id="000002.SZ",
                available_time=_utc("2026-01-05T00:00:00+00:00"),
            ),
        ),
    )

    _failed(materialized, HardGate.GRAPH_TOPOLOGY_VIOLATION)


def test_actual_feature_set_must_match_declared_schema_exactly() -> None:
    sample = _sample()
    materialized = _materialize(
        features=(
            MaterializedFeature(
                sample_id=sample.sample_id,
                name="unexpected_feature",
                version="1.0",
                value=0.2,
                available_time=sample.cutoff_time,
            ),
        )
    )

    _failed(materialized, HardGate.FEATURE_SCHEMA_MISMATCH)


def test_non_finite_semantic_feature_fails_gate() -> None:
    sample = _sample()
    materialized = _materialize(
        features=(
            MaterializedFeature(
                sample_id=sample.sample_id,
                name="stock_volatility_5d",
                version="1.0",
                value=nan,
                available_time=sample.cutoff_time,
            ),
        )
    )

    _failed(materialized, HardGate.NON_FINITE_SEMANTIC_FEATURE)


def test_identity_or_hash_field_cannot_be_a_continuous_feature() -> None:
    sample = _sample()
    identity_schema = (FeatureField(name="stock_id_hash", version="1.0"),)
    materialized = _materialize(
        schema=identity_schema,
        features=(
            MaterializedFeature(
                sample_id=sample.sample_id,
                name="stock_id_hash",
                version="1.0",
                value=123.0,
                available_time=sample.cutoff_time,
            ),
        ),
    )

    _failed(materialized, HardGate.IDENTITY_FEATURE_LEAKAGE)


def test_support_only_stock_is_excluded_from_main_evaluation_and_blocks_seal() -> None:
    direct = _sample(stock_id="support-stock")
    materialized = _materialize(
        samples=(direct,),
        nodes=(
            GraphNodeEvidence(
                node_id="support-stock",
                node_type="Stock",
                is_support_only=True,
            ),
        ),
    )

    assert materialized.evaluation_samples == ()
    _failed(materialized, HardGate.SUPPORT_ONLY_TARGET)
    assert seal_dataset(materialized, _manifest(materialized)).passed is False


def test_direct_and_legitimate_spillover_pools_remain_evaluation_targets() -> None:
    direct = _sample("s-direct")
    spill = _sample("s-spill", "000002.SZ", pool_type=PoolType.SPILLOVER)
    materialized = _materialize(
        samples=(direct, spill),
        labels=(
            MaterializedLabel(
                "s-direct",
                _utc("2026-01-09T00:00:00+00:00"),
                _utc("2026-01-09T00:00:00+00:00"),
            ),
            MaterializedLabel(
                "s-spill",
                _utc("2026-01-09T00:00:00+00:00"),
                _utc("2026-01-09T00:00:00+00:00"),
            ),
        ),
        features=(
            MaterializedFeature(
                "s-direct", "stock_volatility_5d", "1.0", 0.2, direct.cutoff_time
            ),
            MaterializedFeature(
                "s-spill", "stock_volatility_5d", "1.0", 0.3, spill.cutoff_time
            ),
        ),
        nodes=(
            GraphNodeEvidence("000001.SZ", "Stock"),
            GraphNodeEvidence("000002.SZ", "Stock"),
        ),
        references={
            "s-direct": ("000001.SZ",),
            "s-spill": ("000002.SZ",),
        },
    )

    assert [sample.pool_type for sample in materialized.evaluation_samples] == [
        PoolType.DIRECT,
        PoolType.SPILLOVER,
    ]


def test_manifest_fingerprint_tampering_blocks_seal() -> None:
    materialized = _materialize()
    valid_manifest = _manifest(materialized)
    tampered = replace(valid_manifest, materialization_fingerprint="0" * 64)

    report = seal_dataset(materialized, tampered)

    assert report.passed is False
    assert report.gate_results[HardGate.MANIFEST_EVIDENCE_MISMATCH.value] is False


def test_manifest_binding_rejects_a_mismatched_expected_schema() -> None:
    materialized = _materialize()
    mismatched = replace(_manifest(materialized), feature_schema_hash="0" * 64)

    with pytest.raises(ValueError, match="feature_schema_hash"):
        bind_materialization_evidence(mismatched, materialized)


def test_evidence_fingerprint_is_deterministic_across_input_order() -> None:
    first = _materialize()
    second = _materialize(
        schema=tuple(reversed(SCHEMA)),
        nodes=tuple(reversed(first.graph_nodes)),
        references={"s-001": tuple(reversed(first.sample_graph_references["s-001"]))},
    )

    assert first.evidence_fingerprint == second.evidence_fingerprint
    assert first.feature_schema_hash == compute_feature_schema_hash(SCHEMA)


@pytest.mark.parametrize(
    "contract_field",
    (
        "version",
        "node_properties",
        "node_required",
        "edge_source",
        "edge_target",
        "edge_properties",
        "edge_max_hops",
        "edge_top_k",
        "edge_directed",
    ),
)
def test_graph_schema_hash_binds_every_frozen_contract_field(
    monkeypatch: pytest.MonkeyPatch,
    contract_field: str,
) -> None:
    baseline = _materialize().graph_schema_hash
    if contract_field == "version":
        monkeypatch.setattr(
            materializer_module,
            "GRAPH_SCHEMA_CONTRACT_VERSION",
            "seatflow-graph-contract-test-version",
        )
    elif contract_field.startswith("node_"):
        schema = NODE_SCHEMAS[NodeType.STOCK]
        if contract_field == "node_properties":
            replacement = replace(schema, properties=schema.properties + ("probe",))
        else:
            replacement = replace(schema, required=schema.required + ("probe",))
        monkeypatch.setitem(NODE_SCHEMAS, NodeType.STOCK, replacement)
    else:
        schema = EDGE_SCHEMAS[EdgeType.SPILLOVER_CANDIDATE]
        replacements = {
            "edge_source": {"source_type": NodeType.SEAT},
            "edge_target": {"target_type": NodeType.EVENT},
            "edge_properties": {"properties": schema.properties + ("probe",)},
            "edge_max_hops": {"max_hops": schema.max_hops + 1},
            "edge_top_k": {"top_k": schema.top_k + 1},
            "edge_directed": {"directed": not schema.directed},
        }
        monkeypatch.setitem(
            EDGE_SCHEMAS,
            EdgeType.SPILLOVER_CANDIDATE,
            replace(schema, **replacements[contract_field]),
        )

    assert _materialize().graph_schema_hash != baseline
