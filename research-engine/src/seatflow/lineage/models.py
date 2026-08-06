"""Immutable artifact lineage DTOs and fail-closed graph validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable
from urllib.parse import urlparse

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LineageLayer(str, Enum):
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    TENSOR = "TENSOR"


class DataClassification(str, Enum):
    """Evidence classification; synthetic fixtures can never imply formal readiness."""

    SYNTHETIC_NON_FORMAL = "SYNTHETIC_NON_FORMAL"
    REAL_UNVERIFIED = "REAL_UNVERIFIED"


class LineageValidationError(ValueError):
    """Validation failure with a stable machine-readable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class SourceManifestRef:
    """Immutable reference to the ingestion manifest backing an artifact."""

    uri: str
    sha256: str
    schema_version: str

    def __post_init__(self) -> None:
        _validate_uri(self.uri, "source_manifest.uri")
        _validate_sha256(self.sha256, "source_manifest.sha256")
        _validate_non_empty(self.schema_version, "source_manifest.schema_version")


@dataclass(frozen=True)
class TimeBounds:
    """Event and availability time coverage carried through the lineage chain."""

    event_time_start: datetime
    event_time_end: datetime
    available_time_start: datetime
    available_time_end: datetime

    def __post_init__(self) -> None:
        for name, value in (
            ("event_time_start", self.event_time_start),
            ("event_time_end", self.event_time_end),
            ("available_time_start", self.available_time_start),
            ("available_time_end", self.available_time_end),
        ):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.event_time_start > self.event_time_end:
            raise ValueError("event_time_start must not exceed event_time_end")
        if self.available_time_start > self.available_time_end:
            raise ValueError("available_time_start must not exceed available_time_end")


@dataclass(frozen=True)
class SourceRecordRef:
    """Exact Bronze source record and its point-in-time availability evidence."""

    source_namespace: str
    source_record_id: str
    raw_payload_sha256: str
    available_time: datetime

    def __post_init__(self) -> None:
        _validate_non_empty(self.source_namespace, "source_namespace")
        _validate_non_empty(self.source_record_id, "source_record_id")
        _validate_sha256(self.raw_payload_sha256, "raw_payload_sha256")
        if self.available_time.tzinfo is None:
            raise ValueError("available_time must be timezone-aware")


@dataclass(frozen=True)
class ArtifactLineage:
    """Immutable evidence for one Bronze, Silver, Gold, or Tensor artifact."""

    artifact_id: str
    layer: LineageLayer
    uri: str
    sha256: str
    schema_version: str
    schema_fingerprint: str
    row_count: int
    time_bounds: TimeBounds
    parents: tuple[str, ...]
    source_manifest: SourceManifestRef
    data_classification: DataClassification
    source_records: tuple[SourceRecordRef, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty(self.artifact_id, "artifact_id")
        if not isinstance(self.layer, LineageLayer):
            raise ValueError("layer must be a LineageLayer")
        _validate_uri(self.uri, "uri")
        _validate_sha256(self.sha256, "sha256")
        _validate_non_empty(self.schema_version, "schema_version")
        _validate_sha256(self.schema_fingerprint, "schema_fingerprint")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int):
            raise ValueError("row_count must be an integer")
        if self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if not isinstance(self.parents, tuple):
            raise ValueError("parents must be an immutable tuple")
        if len(self.parents) != len(set(self.parents)):
            raise ValueError("parents must not contain duplicates")
        if any(not isinstance(parent, str) or not parent for parent in self.parents):
            raise ValueError("parent artifact IDs must be non-empty strings")
        if not isinstance(self.data_classification, DataClassification):
            raise ValueError("data_classification must be explicit")
        if not isinstance(self.source_records, tuple):
            raise ValueError("source_records must be an immutable tuple")
        record_ids = [record.source_record_id for record in self.source_records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("source_records must not contain duplicate source_record_id values")


@dataclass(frozen=True)
class SampleLineage:
    """Immutable mapping from a Tensor row to exact Bronze source records."""

    sample_id: str
    tensor_artifact_id: str
    tensor_row_index: int
    source_record_ids: tuple[str, ...]
    data_classification: DataClassification

    def __post_init__(self) -> None:
        _validate_non_empty(self.sample_id, "sample_id")
        _validate_non_empty(self.tensor_artifact_id, "tensor_artifact_id")
        if isinstance(self.tensor_row_index, bool) or not isinstance(self.tensor_row_index, int):
            raise ValueError("tensor_row_index must be an integer")
        if self.tensor_row_index < 0:
            raise ValueError("tensor_row_index must be non-negative")
        if not isinstance(self.source_record_ids, tuple) or not self.source_record_ids:
            raise ValueError("source_record_ids must be a non-empty immutable tuple")
        if len(self.source_record_ids) != len(set(self.source_record_ids)):
            raise ValueError("source_record_ids must not contain duplicates")
        if any(not isinstance(record_id, str) or not record_id for record_id in self.source_record_ids):
            raise ValueError("source_record_ids must contain non-empty strings")
        if not isinstance(self.data_classification, DataClassification):
            raise ValueError("data_classification must be explicit")


@dataclass(frozen=True)
class SampleLineageTrace:
    """Validated exact source evidence for one Tensor sample."""

    sample: SampleLineage
    artifact_trace: tuple[ArtifactLineage, ...]
    source_records: tuple[SourceRecordRef, ...]


@dataclass(frozen=True)
class ValidatedLineageGraph:
    """A validated, deterministically ordered artifact graph."""

    artifacts: tuple[ArtifactLineage, ...]
    fingerprint: str

    def trace_to_bronze(self, artifact_id: str) -> tuple[ArtifactLineage, ...]:
        """Return a deterministic depth-first trace from an artifact to Bronze roots."""
        index = {artifact.artifact_id: artifact for artifact in self.artifacts}
        if artifact_id not in index:
            raise KeyError(f"Unknown lineage artifact: {artifact_id!r}")
        trace: list[ArtifactLineage] = []
        seen: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in seen:
                return
            seen.add(current_id)
            current = index[current_id]
            trace.append(current)
            for parent_id in sorted(current.parents):
                visit(parent_id)

        visit(artifact_id)
        return tuple(trace)

    def trace_sample(self, sample: SampleLineage) -> SampleLineageTrace:
        """Resolve a Tensor sample only against source records in its ancestor chain."""
        index = {artifact.artifact_id: artifact for artifact in self.artifacts}
        tensor = index.get(sample.tensor_artifact_id)
        if tensor is None:
            raise LineageValidationError(
                "MISSING_TENSOR_ARTIFACT",
                f"Sample references unknown Tensor artifact {sample.tensor_artifact_id!r}",
            )
        if tensor.layer != LineageLayer.TENSOR:
            raise LineageValidationError(
                "NOT_A_TENSOR_ARTIFACT",
                f"Sample artifact {sample.tensor_artifact_id!r} is {tensor.layer.value}",
            )
        if sample.tensor_row_index >= tensor.row_count:
            raise LineageValidationError(
                "TENSOR_ROW_OUT_OF_RANGE",
                f"Tensor row {sample.tensor_row_index} is outside row_count={tensor.row_count}",
            )
        if sample.data_classification != tensor.data_classification:
            raise LineageValidationError(
                "CLASSIFICATION_MISMATCH",
                "Sample and Tensor artifact classifications differ",
            )
        artifact_trace = self.trace_to_bronze(sample.tensor_artifact_id)
        available_records = {
            record.source_record_id: record
            for artifact in artifact_trace
            if artifact.layer == LineageLayer.BRONZE
            for record in artifact.source_records
        }
        missing = tuple(
            record_id
            for record_id in sample.source_record_ids
            if record_id not in available_records
        )
        if missing:
            raise LineageValidationError(
                "SOURCE_RECORD_NOT_IN_ANCESTRY",
                f"Sample source records are absent from Tensor ancestry: {missing}",
            )
        records = tuple(available_records[record_id] for record_id in sample.source_record_ids)
        return SampleLineageTrace(
            sample=sample,
            artifact_trace=artifact_trace,
            source_records=records,
        )


def validate_lineage(artifacts: Iterable[ArtifactLineage]) -> ValidatedLineageGraph:
    """Validate completeness, acyclicity, layer order, and content identity."""
    artifact_tuple = tuple(artifacts)
    if not artifact_tuple:
        raise LineageValidationError("EMPTY_GRAPH", "At least one artifact is required")
    index: dict[str, ArtifactLineage] = {}
    for artifact in artifact_tuple:
        if artifact.artifact_id in index:
            raise LineageValidationError(
                "DUPLICATE_ARTIFACT_ID",
                f"Duplicate artifact ID {artifact.artifact_id!r}",
            )
        index[artifact.artifact_id] = artifact

    for artifact in artifact_tuple:
        for parent_id in artifact.parents:
            if parent_id not in index:
                raise LineageValidationError(
                    "MISSING_PARENT",
                    f"Artifact {artifact.artifact_id!r} references {parent_id!r}",
                )

    _validate_acyclic(index)
    layer_rank = {
        LineageLayer.BRONZE: 0,
        LineageLayer.SILVER: 1,
        LineageLayer.GOLD: 2,
        LineageLayer.TENSOR: 3,
    }
    for artifact in artifact_tuple:
        if artifact.layer == LineageLayer.BRONZE:
            if artifact.parents:
                raise LineageValidationError(
                    "LAYER_ORDER",
                    f"Bronze artifact {artifact.artifact_id!r} cannot have parents",
                )
            if not artifact.source_records:
                raise LineageValidationError(
                    "MISSING_SOURCE_RECORDS",
                    f"Bronze artifact {artifact.artifact_id!r} has no source record evidence",
                )
            continue
        if artifact.source_records:
            raise LineageValidationError(
                "SOURCE_RECORD_LAYER",
                f"Only Bronze artifacts may embed source record evidence: {artifact.artifact_id!r}",
            )
        if not artifact.parents:
            raise LineageValidationError(
                "INCOMPLETE_CHAIN",
                f"{artifact.layer.value} artifact {artifact.artifact_id!r} has no parent",
            )
        expected_parent_rank = layer_rank[artifact.layer] - 1
        for parent_id in artifact.parents:
            parent = index[parent_id]
            if layer_rank[parent.layer] != expected_parent_rank:
                raise LineageValidationError(
                    "LAYER_ORDER",
                    f"{artifact.layer.value} artifact {artifact.artifact_id!r} "
                    f"cannot directly depend on {parent.layer.value} {parent_id!r}",
                )

    _validate_source_record_uniqueness(artifact_tuple)
    tensors = tuple(
        artifact for artifact in artifact_tuple if artifact.layer == LineageLayer.TENSOR
    )
    if not tensors:
        raise LineageValidationError(
            "MISSING_TENSOR_TARGET",
            "A complete Bronze-to-Tensor graph requires at least one Tensor artifact",
        )
    _validate_all_artifacts_reach_tensor(index, tensors)

    ordered = tuple(sorted(artifact_tuple, key=lambda artifact: artifact.artifact_id))
    fingerprint = hashlib.sha256(_canonical_graph_json(ordered).encode("utf-8")).hexdigest()
    return ValidatedLineageGraph(artifacts=ordered, fingerprint=fingerprint)


def _validate_acyclic(index: dict[str, ArtifactLineage]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visiting:
            raise LineageValidationError(
                "CYCLE",
                f"Cycle detected at artifact {artifact_id!r}",
            )
        if artifact_id in visited:
            return
        visiting.add(artifact_id)
        for parent_id in index[artifact_id].parents:
            visit(parent_id)
        visiting.remove(artifact_id)
        visited.add(artifact_id)

    for artifact_id in sorted(index):
        visit(artifact_id)


def _validate_source_record_uniqueness(artifacts: tuple[ArtifactLineage, ...]) -> None:
    owners: dict[str, str] = {}
    for artifact in artifacts:
        for record in artifact.source_records:
            existing_owner = owners.get(record.source_record_id)
            if existing_owner is not None:
                raise LineageValidationError(
                    "DUPLICATE_SOURCE_RECORD_ID",
                    f"Source record {record.source_record_id!r} belongs to both "
                    f"{existing_owner!r} and {artifact.artifact_id!r}",
                )
            owners[record.source_record_id] = artifact.artifact_id


def _validate_all_artifacts_reach_tensor(
    index: dict[str, ArtifactLineage],
    tensors: tuple[ArtifactLineage, ...],
) -> None:
    reachable: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in reachable:
            return
        reachable.add(artifact_id)
        for parent_id in index[artifact_id].parents:
            visit(parent_id)

    for tensor in tensors:
        visit(tensor.artifact_id)
    disconnected = tuple(sorted(set(index) - reachable))
    if disconnected:
        raise LineageValidationError(
            "DISCONNECTED_ARTIFACT",
            f"Artifacts do not contribute to a Tensor target: {disconnected}",
        )


def _canonical_graph_json(artifacts: tuple[ArtifactLineage, ...]) -> str:
    payload = [_artifact_to_dict(artifact) for artifact in artifacts]
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _artifact_to_dict(artifact: ArtifactLineage) -> dict[str, Any]:
    bounds = artifact.time_bounds
    return {
        "artifact_id": artifact.artifact_id,
        "layer": artifact.layer.value,
        "uri": artifact.uri,
        "sha256": artifact.sha256,
        "schema_version": artifact.schema_version,
        "schema_fingerprint": artifact.schema_fingerprint,
        "row_count": artifact.row_count,
        "time_bounds": {
            "event_time_start": bounds.event_time_start.isoformat(),
            "event_time_end": bounds.event_time_end.isoformat(),
            "available_time_start": bounds.available_time_start.isoformat(),
            "available_time_end": bounds.available_time_end.isoformat(),
        },
        "parents": sorted(artifact.parents),
        "source_manifest": {
            "uri": artifact.source_manifest.uri,
            "sha256": artifact.source_manifest.sha256,
            "schema_version": artifact.source_manifest.schema_version,
        },
        "source_records": [
            {
                "source_namespace": record.source_namespace,
                "source_record_id": record.source_record_id,
                "raw_payload_sha256": record.raw_payload_sha256,
                "available_time": record.available_time.isoformat(),
            }
            for record in sorted(
                artifact.source_records,
                key=lambda record: record.source_record_id,
            )
        ],
        "data_classification": artifact.data_classification.value,
    }


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 64-character sha256")


def _validate_uri(value: str, field_name: str) -> None:
    _validate_non_empty(value, field_name)
    if not urlparse(value).scheme:
        raise ValueError(f"{field_name} must be an absolute URI")
