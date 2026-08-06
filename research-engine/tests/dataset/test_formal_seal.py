"""Formal Dataset Seal must be derived from lineage-backed evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from seatflow.dataset.manifest import DatasetManifest, bind_materialization_evidence
from seatflow.dataset.materializer import (
    FeatureField,
    GraphNodeEvidence,
    MaterializedFeature,
    MaterializedLabel,
    materialize_dataset,
)
from seatflow.dataset.samples import PoolType, PredictionSample
from seatflow.dataset.seal import (
    FormalDatasetEvidence,
    bind_formal_dataset_evidence,
    compute_calendar_content_hash,
    seal_dataset,
    seal_formal_dataset,
)
from seatflow.labels.tradeability import TradeabilityState
from seatflow.lineage import (
    ArtifactLineage,
    DataClassification,
    LineageLayer,
    SampleLineage,
    SourceManifestRef,
    SourceRecordRef,
    TimeBounds,
    validate_lineage,
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _sha(seed: str) -> str:
    return seed * 64


def _calendar() -> tuple[datetime, ...]:
    return tuple(_utc(f"2026-01-{day:02d}T00:00:00+00:00") for day in range(5, 10))


def _sample(
    sample_id: str = "s-001",
    day: str = "2026-01-05T00:00:00+00:00",
) -> PredictionSample:
    timestamp = _utc(day)
    return PredictionSample(
        sample_id=sample_id,
        cycle_id="cyc-001",
        stock_id="000001.SZ",
        prediction_date=timestamp,
        cutoff_time=timestamp,
        days_since_last_event=0,
        pool_type=PoolType.DIRECT,
        tradeability=TradeabilityState.TRADABLE,
        available_time=timestamp,
        is_tradable=True,
        is_backtest_eligible=True,
    )


def _materialized(samples: tuple[PredictionSample, ...] | None = None):
    rows = samples or (_sample(),)
    return materialize_dataset(
        samples=rows,
        labels=tuple(
            MaterializedLabel(
                sample.sample_id,
                _utc("2026-01-09T00:00:00+00:00"),
                _utc("2026-01-09T00:00:00+00:00"),
            )
            for sample in rows
        ),
        features=tuple(
            MaterializedFeature(
                sample.sample_id,
                "stock_volatility_5d",
                "1.0",
                0.2,
                sample.cutoff_time,
            )
            for sample in rows
        ),
        feature_schema=(FeatureField("stock_volatility_5d", "1.0"),),
        graph_nodes=(GraphNodeEvidence("000001.SZ", "Stock"),),
        graph_edges=(),
        sample_graph_references={sample.sample_id: ("000001.SZ",) for sample in rows},
        trade_calendar=_calendar(),
        materialized_at=_utc("2026-01-12T00:00:00+00:00"),
    )


def _manifest(materialized) -> DatasetManifest:
    return bind_materialization_evidence(
        DatasetManifest(
            dataset_version="seatflow.v3.0",
            build_timestamp=_utc("2026-01-12T00:00:00+00:00"),
            source_manifest_hash="0" * 64,
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
            artifact_uris=["file:///caller/fake-tensor.parquet"],
        ),
        materialized,
    )


def _source_manifest(uri: str = "file:///source/real-manifest.json") -> SourceManifestRef:
    return SourceManifestRef(
        uri=uri,
        sha256=_sha("a"),
        schema_version="seatflow.source.v1",
    )


def _record(
    record_id: str,
    raw_payload_sha256: str,
    available_time: str = "2026-01-05T00:00:00+00:00",
) -> SourceRecordRef:
    return SourceRecordRef(
        source_namespace="akshare.trade_calendar"
        if record_id.startswith("calendar:")
        else "akshare.daily_bar",
        source_record_id=record_id,
        raw_payload_sha256=raw_payload_sha256,
        available_time=_utc(available_time),
    )


def _bounds() -> TimeBounds:
    return TimeBounds(
        event_time_start=_utc("2026-01-01T00:00:00+00:00"),
        event_time_end=_utc("2026-01-09T00:00:00+00:00"),
        available_time_start=_utc("2026-01-01T00:00:00+00:00"),
        available_time_end=_utc("2026-01-05T00:00:00+00:00"),
    )


def _artifact(
    artifact_id: str,
    layer: LineageLayer,
    parents: tuple[str, ...] = (),
    *,
    row_count: int = 1,
    classification: DataClassification = DataClassification.REAL_VERIFIED,
    source_records: tuple[SourceRecordRef, ...] = (),
    source_manifest: SourceManifestRef | None = None,
    uri: str | None = None,
) -> ArtifactLineage:
    seeds = {"bronze": "1", "silver": "2", "gold": "3", "tensor": "4"}
    return ArtifactLineage(
        artifact_id=artifact_id,
        layer=layer,
        uri=uri or f"file:///artifacts/{artifact_id}.parquet",
        sha256=seeds.get(artifact_id, "5") * 64,
        schema_version=f"seatflow.{artifact_id}.v1",
        schema_fingerprint=seeds.get(artifact_id, "6") * 64,
        row_count=row_count,
        time_bounds=_bounds(),
        parents=parents,
        source_manifest=source_manifest or _source_manifest(),
        data_classification=classification,
        source_records=source_records,
    )


def _formal_evidence(
    *,
    materialized=None,
    sample_lineages: tuple[SampleLineage, ...] | None = None,
    tensor_row_count: int = 1,
    classification: DataClassification = DataClassification.REAL_VERIFIED,
    bronze_records: tuple[SourceRecordRef, ...] | None = None,
    calendar_record_ids: tuple[str, ...] = ("calendar:2026-01",),
) -> FormalDatasetEvidence:
    materialized = materialized or _materialized()
    calendar_hash = compute_calendar_content_hash(materialized.trade_calendar)
    records = bronze_records or (
        _record("sample:s-001", _sha("8")),
        _record("calendar:2026-01", calendar_hash),
    )
    bronze = _artifact(
        "bronze",
        LineageLayer.BRONZE,
        row_count=len(records),
        classification=classification,
        source_records=records,
    )
    silver = _artifact("silver", LineageLayer.SILVER, ("bronze",), classification=classification)
    gold = _artifact("gold", LineageLayer.GOLD, ("silver",), classification=classification)
    tensor = _artifact(
        "tensor",
        LineageLayer.TENSOR,
        ("gold",),
        row_count=tensor_row_count,
        classification=classification,
    )
    lineages = sample_lineages or tuple(
        SampleLineage(
            sample_id=sample.sample_id,
            tensor_artifact_id="tensor",
            tensor_row_index=index,
            source_record_ids=("sample:s-001", "calendar:2026-01"),
            data_classification=classification,
        )
        for index, sample in enumerate(materialized.samples)
    )
    return FormalDatasetEvidence(
        lineage_graph=validate_lineage((bronze, silver, gold, tensor)),
        sample_lineages=lineages,
        calendar_source_record_ids=calendar_record_ids,
    )


def _formal_manifest(materialized, evidence: FormalDatasetEvidence) -> DatasetManifest:
    return bind_formal_dataset_evidence(_manifest(materialized), materialized, evidence)


def test_real_verified_lineage_bridge_remains_blocked_without_audit_attestation() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized)
    manifest = _formal_manifest(materialized, evidence)

    report = seal_formal_dataset(materialized, manifest, evidence)

    assert report.integrity_passed is True
    assert report.formal_passed is False
    assert report.seal_kind == "FORMAL"
    assert report.passed is False
    assert report.gate_results["FORMAL_AUDIT_ATTESTATION_REQUIRED"] is False
    assert manifest.formal is True
    assert manifest.source_manifest_hash == _sha("a")
    assert manifest.artifact_uris == ["file:///artifacts/tensor.parquet"]
    assert manifest.lineage_fingerprint == evidence.lineage_graph.fingerprint
    assert manifest.data_classification == DataClassification.REAL_VERIFIED.value


def test_engineering_seal_does_not_become_formal_readiness() -> None:
    materialized = _materialized()
    report = seal_dataset(materialized, _manifest(materialized))

    assert report.integrity_passed is True
    assert report.formal_passed is False
    assert report.seal_kind == "ENGINEERING"
    assert report.passed is False


def test_caller_source_hash_and_artifact_uri_spoofing_fails_formal_manifest() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized)
    manifest = replace(
        _formal_manifest(materialized, evidence),
        source_manifest_hash="f" * 64,
        artifact_uris=["file:///caller/fake-tensor.parquet"],
        lineage_fingerprint="e" * 64,
        data_classification=DataClassification.REAL_VERIFIED.value,
    )

    assert seal_formal_dataset(materialized, manifest, evidence).integrity_passed is False


def test_every_dataset_sample_requires_exactly_one_sample_lineage() -> None:
    materialized = _materialized(samples=(_sample("s-001"), _sample("s-002")))
    missing = _formal_evidence(
        materialized=materialized,
        tensor_row_count=2,
        sample_lineages=(
            SampleLineage(
                "s-001",
                "tensor",
                0,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
        ),
    )

    assert (
        seal_formal_dataset(
            materialized,
            _formal_manifest(materialized, missing),
            missing,
        ).integrity_passed
        is False
    )

    duplicate = _formal_evidence(
        materialized=materialized,
        tensor_row_count=2,
        sample_lineages=(
            SampleLineage(
                "s-001",
                "tensor",
                0,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
            SampleLineage(
                "s-001",
                "tensor",
                1,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
        ),
    )
    assert (
        seal_formal_dataset(
            materialized,
            _formal_manifest(materialized, duplicate),
            duplicate,
        ).integrity_passed
        is False
    )


@pytest.mark.parametrize("row_indices", ((1,), (0, 0)))
def test_tensor_rows_must_be_valid_unique_and_cover_dataset(
    row_indices: tuple[int, ...],
) -> None:
    materialized = _materialized()
    evidence = _formal_evidence(
        materialized=materialized,
        tensor_row_count=1,
        sample_lineages=tuple(
            SampleLineage(
                f"s-{index + 1:03d}",
                "tensor",
                row_index,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            )
            for index, row_index in enumerate(row_indices)
        ),
    )

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_tensor_rows_must_match_materialized_sample_order_exactly() -> None:
    materialized = _materialized(samples=(_sample("s-001"), _sample("s-002")))
    evidence = _formal_evidence(
        materialized=materialized,
        tensor_row_count=2,
        sample_lineages=(
            SampleLineage(
                "s-001",
                "tensor",
                1,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
            SampleLineage(
                "s-002",
                "tensor",
                0,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
        ),
    )

    report = seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence)

    assert report.integrity_passed is False
    assert any(
        ("tensor_row_sample_alignment", "SampleLineage row must match dataset row")
        in finding.evidence
        for finding in report.findings
    )


def test_sample_lineage_cannot_be_reordered_across_tensor_artifacts() -> None:
    materialized = _materialized(samples=(_sample("s-001"), _sample("s-002")))
    evidence = _formal_evidence(
        materialized=materialized,
        tensor_row_count=2,
        sample_lineages=(
            SampleLineage(
                "s-001",
                "tensor",
                0,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
            SampleLineage(
                "s-002",
                "tensor-extra",
                1,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_VERIFIED,
            ),
        ),
    )

    report = seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence)

    assert report.integrity_passed is False


def test_tensor_row_count_must_match_dataset_sample_count() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized, tensor_row_count=2)

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


@pytest.mark.parametrize(
    "classification",
    (DataClassification.SYNTHETIC_NON_FORMAL, DataClassification.REAL_UNVERIFIED),
)
def test_formal_seal_rejects_non_formal_classification(
    classification: DataClassification,
) -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized, classification=classification)

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_formal_seal_rejects_mixed_classification() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(
        materialized=materialized,
        classification=DataClassification.REAL_VERIFIED,
        sample_lineages=(
            SampleLineage(
                "s-001",
                "tensor",
                0,
                ("sample:s-001", "calendar:2026-01"),
                DataClassification.REAL_UNVERIFIED,
            ),
        ),
    )

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_calendar_source_record_must_be_in_tensor_ancestry() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(
        materialized=materialized,
        calendar_record_ids=("calendar:not-in-ancestry",),
    )

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_arbitrary_self_consistent_calendar_without_content_hash_fails() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(
        materialized=materialized,
        bronze_records=(
            _record("sample:s-001", _sha("8")),
            _record("calendar:2026-01", _sha("9")),
        ),
    )

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_calendar_and_source_manifest_tampering_changes_formal_manifest_hash() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized)
    manifest = _formal_manifest(materialized, evidence)
    tampered_calendar = replace(manifest, calendar_content_hash="0" * 64)
    tampered_manifest_uri = replace(manifest, source_manifest_uri="file:///source/tampered.json")

    assert seal_formal_dataset(materialized, tampered_calendar, evidence).integrity_passed is False
    assert seal_formal_dataset(materialized, tampered_manifest_uri, evidence).integrity_passed is False


def test_pit_source_after_sample_cutoff_fails_formal_seal() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(
        materialized=materialized,
        bronze_records=(
            _record("sample:s-001", _sha("8"), "2026-01-06T00:00:00+00:00"),
            _record(
                "calendar:2026-01",
                compute_calendar_content_hash(materialized.trade_calendar),
            ),
        ),
    )

    assert seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence).integrity_passed is False


def test_formal_fingerprint_is_deterministic_and_tamper_evident() -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized)
    first = _formal_manifest(materialized, evidence)
    second = _formal_manifest(materialized, evidence)

    assert first.formal_evidence_fingerprint == second.formal_evidence_fingerprint
    tampered = replace(first, formal_evidence_fingerprint="0" * 64)
    assert seal_formal_dataset(materialized, tampered, evidence).integrity_passed is False


def test_unexpected_trace_sample_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = _materialized()
    evidence = _formal_evidence(materialized=materialized)

    def raise_bug(_self: object, _sample_lineage: SampleLineage) -> object:
        raise RuntimeError("programmer error")

    monkeypatch.setattr(type(evidence.lineage_graph), "trace_sample", raise_bug)

    with pytest.raises(RuntimeError, match="programmer error"):
        seal_formal_dataset(materialized, _formal_manifest(materialized, evidence), evidence)
