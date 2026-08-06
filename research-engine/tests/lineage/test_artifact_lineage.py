"""Immutable Bronze-to-Tensor lineage validation tests."""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest


def _api():
    try:
        return importlib.import_module("seatflow.lineage")
    except ModuleNotFoundError as exc:
        pytest.fail(f"lineage package is missing: {exc}")


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _manifest(api):
    return api.SourceManifestRef(
        uri="file:///evidence/ingestion-manifest.json",
        sha256="a" * 64,
        schema_version="seatflow.ingestion.manifest.v1",
    )


def _bounds(api):
    return api.TimeBounds(
        event_time_start=_utc("2026-08-01T00:00:00+00:00"),
        event_time_end=_utc("2026-08-05T00:00:00+00:00"),
        available_time_start=_utc("2026-08-01T08:00:00+00:00"),
        available_time_end=_utc("2026-08-05T08:00:00+00:00"),
    )


def _source_record(api, artifact_id: str):
    digest = {
        "bronze": "c",
        "bronze-extra": "d",
    }.get(artifact_id, "e")
    return api.SourceRecordRef(
        source_namespace="synthetic-fixture",
        source_record_id=f"synthetic:{artifact_id}:{digest * 64}",
        raw_payload_sha256=digest * 64,
        available_time=_utc("2026-08-05T08:00:00+00:00"),
    )


def _artifact(api, artifact_id: str, layer, parents: tuple[str, ...] = ()):
    return api.ArtifactLineage(
        artifact_id=artifact_id,
        layer=layer,
        uri=f"file:///artifacts/{artifact_id}.parquet",
        sha256={
            "bronze": "1",
            "silver": "2",
            "gold": "3",
            "tensor": "4",
        }[artifact_id]
        * 64,
        schema_version=f"seatflow.{artifact_id}.v1",
        schema_fingerprint={
            "bronze": "5",
            "silver": "6",
            "gold": "7",
            "tensor": "8",
        }[artifact_id]
        * 64,
        row_count=4,
        time_bounds=_bounds(api),
        parents=parents,
        source_manifest=_manifest(api),
        data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
        source_records=(
            (_source_record(api, artifact_id),)
            if layer == api.LineageLayer.BRONZE
            else ()
        ),
    )


def _complete_chain(api):
    bronze = _artifact(api, "bronze", api.LineageLayer.BRONZE)
    silver = _artifact(api, "silver", api.LineageLayer.SILVER, ("bronze",))
    gold = _artifact(api, "gold", api.LineageLayer.GOLD, ("silver",))
    tensor = _artifact(api, "tensor", api.LineageLayer.TENSOR, ("gold",))
    return bronze, silver, gold, tensor


def test_complete_chain_traces_tensor_to_bronze_available_time() -> None:
    """Dropping any parent traversal must hide Bronze availability evidence."""
    api = _api()
    graph = api.validate_lineage(_complete_chain(api))

    trace = graph.trace_to_bronze("tensor")

    assert tuple(artifact.layer for artifact in trace) == (
        api.LineageLayer.TENSOR,
        api.LineageLayer.GOLD,
        api.LineageLayer.SILVER,
        api.LineageLayer.BRONZE,
    )
    assert trace[-1].time_bounds.available_time_end == _utc(
        "2026-08-05T08:00:00+00:00"
    )
    assert trace[-1].source_manifest.sha256 == "a" * 64
    assert all(
        artifact.data_classification == api.DataClassification.SYNTHETIC_NON_FORMAL
        for artifact in trace
    )


def test_graph_fingerprint_is_artifact_and_parent_order_invariant() -> None:
    """Persisting input iteration order must not change graph identity."""
    api = _api()
    bronze, silver, gold, tensor = _complete_chain(api)
    extra_bronze = api.ArtifactLineage(
        artifact_id="bronze-extra",
        layer=api.LineageLayer.BRONZE,
        uri="file:///artifacts/bronze-extra.parquet",
        sha256="9" * 64,
        schema_version="seatflow.bronze.v1",
        schema_fingerprint="b" * 64,
        row_count=1,
        time_bounds=_bounds(api),
        parents=(),
        source_manifest=_manifest(api),
        data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
        source_records=(_source_record(api, "bronze-extra"),),
    )
    silver_two_parent = api.ArtifactLineage(
        **{
            **silver.__dict__,
            "parents": ("bronze-extra", "bronze"),
        }
    )

    first = api.validate_lineage((bronze, extra_bronze, silver_two_parent, gold, tensor))
    reversed_silver = api.ArtifactLineage(
        **{
            **silver.__dict__,
            "parents": ("bronze", "bronze-extra"),
        }
    )
    second = api.validate_lineage((tensor, gold, reversed_silver, extra_bronze, bronze))

    assert first.fingerprint == second.fingerprint


def test_missing_parent_fails_closed() -> None:
    """Ignoring absent parents must allow an unverifiable Tensor artifact."""
    api = _api()
    bronze, silver, gold, _ = _complete_chain(api)
    tensor = _artifact(api, "tensor", api.LineageLayer.TENSOR, ("missing-gold",))

    with pytest.raises(ValueError, match="MISSING_PARENT"):
        api.validate_lineage((bronze, silver, gold, tensor))


def test_graph_without_tensor_target_is_incomplete() -> None:
    """A Bronze-only evidence fragment must never pass as a complete lineage graph."""
    api = _api()
    bronze = _artifact(api, "bronze", api.LineageLayer.BRONZE)

    with pytest.raises(ValueError, match="MISSING_TENSOR_TARGET"):
        api.validate_lineage((bronze,))


def test_layer_skip_fails_closed() -> None:
    """Allowing Bronze-to-Gold skips must make the chain incomplete."""
    api = _api()
    bronze = _artifact(api, "bronze", api.LineageLayer.BRONZE)
    gold = _artifact(api, "gold", api.LineageLayer.GOLD, ("bronze",))

    with pytest.raises(ValueError, match="LAYER_ORDER"):
        api.validate_lineage((bronze, gold))


def test_cycle_fails_before_layer_acceptance() -> None:
    """Removing cycle detection must admit a self-referential lineage graph."""
    api = _api()
    silver = _artifact(api, "silver", api.LineageLayer.SILVER, ("gold",))
    gold = _artifact(api, "gold", api.LineageLayer.GOLD, ("silver",))

    with pytest.raises(ValueError, match="CYCLE"):
        api.validate_lineage((silver, gold))


def test_invalid_hash_shape_is_rejected() -> None:
    """Accepting non-SHA-256 artifact hashes must fail this constructor contract."""
    api = _api()
    with pytest.raises(ValueError, match="sha256"):
        api.ArtifactLineage(
            artifact_id="bronze",
            layer=api.LineageLayer.BRONZE,
            uri="file:///artifacts/bronze.parquet",
            sha256="not-a-sha256",
            schema_version="seatflow.bronze.v1",
            schema_fingerprint="5" * 64,
            row_count=1,
            time_bounds=_bounds(api),
            parents=(),
            source_manifest=_manifest(api),
            data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
        )


def test_lineage_dtos_are_immutable() -> None:
    """Mutable artifact evidence must not be accepted as an immutable audit record."""
    api = _api()
    artifact = _artifact(api, "bronze", api.LineageLayer.BRONZE)

    with pytest.raises(FrozenInstanceError):
        artifact.row_count = 999


def test_tensor_sample_traces_to_exact_bronze_source_record() -> None:
    """Artifact-only ancestry must not satisfy exact sample-to-record traceability."""
    api = _api()
    source_record = api.SourceRecordRef(
        source_namespace="akshare",
        source_record_id="akshare:lhb_participation:" + "c" * 64,
        raw_payload_sha256="d" * 64,
        available_time=_utc("2026-08-05T08:00:00+00:00"),
    )
    bronze, silver, gold, tensor = _complete_chain(api)
    bronze_with_record = api.ArtifactLineage(
        **{
            **bronze.__dict__,
            "source_records": (source_record,),
        }
    )
    graph = api.validate_lineage((bronze_with_record, silver, gold, tensor))
    sample = api.SampleLineage(
        sample_id="sample-001",
        tensor_artifact_id="tensor",
        tensor_row_index=3,
        source_record_ids=(source_record.source_record_id,),
        data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
    )

    trace = graph.trace_sample(sample)

    assert trace.sample.tensor_row_index == 3
    assert trace.source_records == (source_record,)
    assert trace.source_records[0].available_time == _utc("2026-08-05T08:00:00+00:00")


def test_sample_reference_outside_tensor_ancestry_fails_closed() -> None:
    """Global source lookup must not let a sample claim an unrelated Bronze record."""
    api = _api()
    source_record = api.SourceRecordRef(
        source_namespace="akshare",
        source_record_id="akshare:daily_bars:" + "e" * 64,
        raw_payload_sha256="f" * 64,
        available_time=_utc("2026-08-05T08:00:00+00:00"),
    )
    bronze, silver, gold, tensor = _complete_chain(api)
    bronze_with_record = api.ArtifactLineage(
        **{
            **bronze.__dict__,
            "source_records": (source_record,),
        }
    )
    graph = api.validate_lineage((bronze_with_record, silver, gold, tensor))
    sample = api.SampleLineage(
        sample_id="sample-002",
        tensor_artifact_id="tensor",
        tensor_row_index=0,
        source_record_ids=("akshare:daily_bars:" + "0" * 64,),
        data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
    )

    with pytest.raises(ValueError, match="SOURCE_RECORD_NOT_IN_ANCESTRY"):
        graph.trace_sample(sample)


def test_sample_tensor_row_index_must_exist() -> None:
    """Accepting index equal to row_count must not create lineage for a missing row."""
    api = _api()
    bronze, silver, gold, tensor = _complete_chain(api)
    graph = api.validate_lineage((bronze, silver, gold, tensor))
    sample = api.SampleLineage(
        sample_id="sample-out-of-range",
        tensor_artifact_id="tensor",
        tensor_row_index=tensor.row_count,
        source_record_ids=(bronze.source_records[0].source_record_id,),
        data_classification=api.DataClassification.SYNTHETIC_NON_FORMAL,
    )

    with pytest.raises(ValueError, match="TENSOR_ROW_OUT_OF_RANGE"):
        graph.trace_sample(sample)
