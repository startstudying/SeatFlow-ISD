"""Fail-closed Dataset v3 seal over computed materialization evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime

from seatflow.dataset.manifest import DatasetManifest, compute_manifest_hash
from seatflow.dataset.materializer import (
    MATERIALIZATION_GATES,
    GateFinding,
    HardGate,
    MaterializedDataset,
    materialize_dataset,
)
from seatflow.dataset.samples import PredictionSample
from seatflow.lineage.models import (
    ArtifactLineage,
    DataClassification,
    SampleLineage,
    SourceRecordRef,
    ValidatedLineageGraph,
)


@dataclass(frozen=True)
class SealReport:
    """Result of binding and validating a materialized dataset manifest."""

    integrity_passed: bool
    formal_passed: bool
    seal_kind: str
    gate_results: dict[str, bool]
    violation_count: int
    violation_samples: list[PredictionSample]
    findings: tuple[GateFinding, ...]
    sealed_manifest_hash: str

    @property
    def passed(self) -> bool:
        """Backward-compatible alias that is fail-closed for formal readiness."""

        return self.formal_passed


@dataclass(frozen=True)
class FormalDatasetEvidence:
    """Lineage-backed evidence required for formal Dataset sealing."""

    lineage_graph: ValidatedLineageGraph
    sample_lineages: tuple[SampleLineage, ...]
    calendar_source_record_ids: tuple[str, ...]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_calendar_content_hash(trade_calendar: tuple[datetime, ...]) -> str:
    """Hash frozen trading-calendar date content for formal evidence binding."""

    return _canonical_hash([timestamp.isoformat() for timestamp in trade_calendar])


def _tensor_artifacts(graph: ValidatedLineageGraph) -> tuple[ArtifactLineage, ...]:
    return tuple(
        artifact
        for artifact in graph.artifacts
        if artifact.layer.value == "TENSOR"
    )


def _source_manifests(graph: ValidatedLineageGraph) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            {
                (
                    artifact.source_manifest.uri,
                    artifact.source_manifest.sha256,
                    artifact.source_manifest.schema_version,
                )
                for artifact in graph.artifacts
            }
        )
    )


def _formal_classifications(
    graph: ValidatedLineageGraph,
    sample_lineages: tuple[SampleLineage, ...],
) -> set[DataClassification]:
    return {
        artifact.data_classification for artifact in graph.artifacts
    } | {sample.data_classification for sample in sample_lineages}


def _formal_fingerprint(
    materialized: MaterializedDataset,
    evidence: FormalDatasetEvidence,
    calendar_records: tuple[SourceRecordRef, ...],
) -> str:
    return _canonical_hash(
        {
            "lineage_graph_fingerprint": evidence.lineage_graph.fingerprint,
            "sample_lineages": [
                {
                    "sample_id": sample.sample_id,
                    "tensor_artifact_id": sample.tensor_artifact_id,
                    "tensor_row_index": sample.tensor_row_index,
                    "source_record_ids": list(sample.source_record_ids),
                    "data_classification": sample.data_classification.value,
                }
                for sample in sorted(
                    evidence.sample_lineages,
                    key=lambda row: (row.sample_id, row.tensor_artifact_id, row.tensor_row_index),
                )
            ],
            "calendar_content_hash": compute_calendar_content_hash(
                materialized.trade_calendar
            ),
            "calendar_records": [
                {
                    "source_namespace": record.source_namespace,
                    "source_record_id": record.source_record_id,
                    "raw_payload_sha256": record.raw_payload_sha256,
                    "available_time": record.available_time.isoformat(),
                }
                for record in sorted(calendar_records, key=lambda row: row.source_record_id)
            ],
            "tensor_artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "uri": artifact.uri,
                    "sha256": artifact.sha256,
                    "row_count": artifact.row_count,
                    "data_classification": artifact.data_classification.value,
                    "source_manifest": {
                        "uri": artifact.source_manifest.uri,
                        "sha256": artifact.source_manifest.sha256,
                        "schema_version": artifact.source_manifest.schema_version,
                    },
                }
                for artifact in _tensor_artifacts(evidence.lineage_graph)
            ],
        }
    )


def _calendar_records_for_evidence(
    evidence: FormalDatasetEvidence,
) -> tuple[SourceRecordRef, ...]:
    records: dict[str, SourceRecordRef] = {}
    for sample_lineage in evidence.sample_lineages:
        try:
            trace = evidence.lineage_graph.trace_sample(sample_lineage)
        except (KeyError, ValueError):
            continue
        for record in trace.source_records:
            if record.source_record_id in evidence.calendar_source_record_ids:
                records[record.source_record_id] = record
    return tuple(records[record_id] for record_id in sorted(records))


def bind_formal_dataset_evidence(
    manifest: DatasetManifest,
    materialized: MaterializedDataset,
    evidence: FormalDatasetEvidence,
) -> DatasetManifest:
    """Derive formal manifest fields from lineage evidence, never caller strings."""

    source_manifests = _source_manifests(evidence.lineage_graph)
    source_uri = source_hash = source_schema = ""
    if len(source_manifests) == 1:
        source_uri, source_hash, source_schema = source_manifests[0]
    tensor_artifacts = _tensor_artifacts(evidence.lineage_graph)
    classifications = _formal_classifications(evidence.lineage_graph, evidence.sample_lineages)
    classification = (
        next(iter(classifications)).value if len(classifications) == 1 else "MIXED"
    )
    calendar_records = _calendar_records_for_evidence(evidence)
    return replace(
        manifest,
        formal=True,
        source_manifest_hash=source_hash,
        source_manifest_uri=source_uri,
        source_manifest_schema_version=source_schema,
        artifact_uris=[artifact.uri for artifact in tensor_artifacts],
        lineage_fingerprint=evidence.lineage_graph.fingerprint,
        data_classification=classification,
        calendar_content_hash=compute_calendar_content_hash(materialized.trade_calendar),
        calendar_source_record_ids=sorted(evidence.calendar_source_record_ids),
        formal_evidence_fingerprint=_formal_fingerprint(
            materialized,
            evidence,
            calendar_records,
        ),
    )


def _computed_gate_results(
    findings: tuple[GateFinding, ...],
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for gate in MATERIALIZATION_GATES:
        gate_findings = [finding for finding in findings if finding.gate is gate]
        results[gate.value] = bool(gate_findings) and all(
            finding.passed for finding in gate_findings
        )
    return results


def _manifest_mismatches(
    materialized: MaterializedDataset,
    manifest: DatasetManifest,
    computed_results: dict[str, bool],
) -> tuple[tuple[str, str], ...]:
    mismatches: list[tuple[str, str]] = []
    expected: dict[str, object] = {
        "build_timestamp": materialized.materialized_at.isoformat(),
        "feature_schema_hash": materialized.feature_schema_hash,
        "graph_schema_hash": materialized.graph_schema_hash,
        "materialization_fingerprint": materialized.evidence_fingerprint,
        "gate_evidence_fingerprint": materialized.gate_evidence_fingerprint,
        "row_counts": dict(materialized.row_counts),
        "node_counts": dict(materialized.node_counts),
        "edge_counts": dict(materialized.edge_counts),
        "quality_gate_results": dict(sorted(computed_results.items())),
    }
    actual: dict[str, object] = {
        "build_timestamp": manifest.build_timestamp.isoformat(),
        "feature_schema_hash": manifest.feature_schema_hash,
        "graph_schema_hash": manifest.graph_schema_hash,
        "materialization_fingerprint": manifest.materialization_fingerprint,
        "gate_evidence_fingerprint": manifest.gate_evidence_fingerprint,
        "row_counts": manifest.row_counts,
        "node_counts": manifest.node_counts,
        "edge_counts": manifest.edge_counts,
        "quality_gate_results": manifest.quality_gate_results,
    }
    for field in sorted(expected):
        if actual[field] != expected[field]:
            mismatches.append(
                (field, f"expected={expected[field]!r};actual={actual[field]!r}")
            )
    return tuple(mismatches)


def _field_mismatches(
    expected: DatasetManifest,
    actual: DatasetManifest,
    fields: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    mismatches: list[tuple[str, str]] = []
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            mismatches.append((field, "formal derived evidence mismatch"))
    return tuple(mismatches)


def _formal_evidence_mismatches(
    materialized: MaterializedDataset,
    evidence: FormalDatasetEvidence,
) -> tuple[tuple[str, str], ...]:
    mismatches: list[tuple[str, str]] = []
    dataset_sample_ids = [sample.sample_id for sample in materialized.samples]
    lineage_sample_ids = [sample.sample_id for sample in evidence.sample_lineages]
    if sorted(lineage_sample_ids) != sorted(dataset_sample_ids) or len(
        set(lineage_sample_ids)
    ) != len(lineage_sample_ids):
        mismatches.append(("sample_lineage_coverage", "expected exactly one lineage per sample"))

    traces = []
    for sample_lineage in evidence.sample_lineages:
        try:
            traces.append(evidence.lineage_graph.trace_sample(sample_lineage))
        except (KeyError, ValueError) as exc:
            mismatches.append(("sample_lineage_trace", str(exc)))

    tensor_artifacts = _tensor_artifacts(evidence.lineage_graph)
    if len(tensor_artifacts) != 1:
        mismatches.append(("tensor_artifact_count", "expected exactly one Tensor artifact"))
    elif tensor_artifacts[0].row_count != len(materialized.samples):
        mismatches.append(("tensor_row_count", "Tensor row_count must equal dataset samples"))
    if len(_source_manifests(evidence.lineage_graph)) != 1:
        mismatches.append(("source_manifest", "formal seal requires one source manifest"))

    row_indices = [sample.tensor_row_index for sample in evidence.sample_lineages]
    if sorted(row_indices) != list(range(len(materialized.samples))):
        mismatches.append(("tensor_rows", "Tensor rows must be unique and cover dataset rows"))
    expected_row_by_sample_id = {
        sample.sample_id: index for index, sample in enumerate(materialized.samples)
    }
    for sample_lineage in evidence.sample_lineages:
        if (
            sample_lineage.sample_id in expected_row_by_sample_id
            and sample_lineage.tensor_row_index
            != expected_row_by_sample_id[sample_lineage.sample_id]
        ):
            mismatches.append(
                (
                    "tensor_row_sample_alignment",
                    "SampleLineage row must match dataset row",
                )
            )

    classifications = _formal_classifications(evidence.lineage_graph, evidence.sample_lineages)
    if classifications != {DataClassification.REAL_VERIFIED}:
        mismatches.append(("data_classification", "formal seal requires uniform REAL_VERIFIED"))

    samples_by_id = {sample.sample_id: sample for sample in materialized.samples}
    for trace in traces:
        sample = samples_by_id.get(trace.sample.sample_id)
        if sample is None:
            continue
        for record in trace.source_records:
            if record.available_time > sample.cutoff_time:
                mismatches.append(
                    (
                        "source_record_available_time",
                        f"{record.source_record_id} after sample cutoff",
                    )
                )

    calendar_records = _calendar_records_for_evidence(evidence)
    if len(calendar_records) != len(set(evidence.calendar_source_record_ids)):
        mismatches.append(("calendar_source_records", "calendar records must be in Tensor ancestry"))
    calendar_hash = compute_calendar_content_hash(materialized.trade_calendar)
    if not any(record.raw_payload_sha256 == calendar_hash for record in calendar_records):
        mismatches.append(("calendar_content_hash", "calendar source hash must bind dates"))
    for trace in traces:
        present = {
            record.source_record_id
            for record in trace.source_records
            if record.source_record_id in evidence.calendar_source_record_ids
        }
        if set(evidence.calendar_source_record_ids) - present:
            mismatches.append(
                ("calendar_sample_ancestry", "calendar record missing from sample ancestry")
            )
            break
    return tuple(mismatches)


def _materialization_mismatches(
    provided: MaterializedDataset,
    recomputed: MaterializedDataset,
) -> tuple[tuple[str, str], ...]:
    """Detect attempts to replace derived evidence on a frozen dataset value."""

    mismatches: list[tuple[str, str]] = []
    derived_fields: tuple[tuple[str, object, object], ...] = (
        ("evaluation_samples", provided.evaluation_samples, recomputed.evaluation_samples),
        ("findings", provided.findings, recomputed.findings),
        ("trade_calendar", provided.trade_calendar, recomputed.trade_calendar),
        (
            "label_horizon_trading_days",
            provided.label_horizon_trading_days,
            recomputed.label_horizon_trading_days,
        ),
        (
            "feature_schema_hash",
            provided.feature_schema_hash,
            recomputed.feature_schema_hash,
        ),
        ("graph_schema_hash", provided.graph_schema_hash, recomputed.graph_schema_hash),
        (
            "gate_evidence_fingerprint",
            provided.gate_evidence_fingerprint,
            recomputed.gate_evidence_fingerprint,
        ),
        (
            "evidence_fingerprint",
            provided.evidence_fingerprint,
            recomputed.evidence_fingerprint,
        ),
        ("row_counts", dict(provided.row_counts), dict(recomputed.row_counts)),
        ("node_counts", dict(provided.node_counts), dict(recomputed.node_counts)),
        ("edge_counts", dict(provided.edge_counts), dict(recomputed.edge_counts)),
    )
    for field, actual, expected in derived_fields:
        if actual != expected:
            mismatches.append((f"materialized.{field}", "derived evidence mismatch"))
    return tuple(mismatches)


def seal_dataset(
    materialized: MaterializedDataset,
    manifest: DatasetManifest,
) -> SealReport:
    """Seal only complete evidence whose manifest exactly matches its rows."""

    recomputed = materialize_dataset(
        samples=materialized.samples,
        labels=materialized.labels,
        features=materialized.features,
        feature_schema=materialized.feature_schema,
        graph_nodes=materialized.graph_nodes,
        graph_edges=materialized.graph_edges,
        sample_graph_references=materialized.sample_graph_references,
        trade_calendar=materialized.trade_calendar,
        materialized_at=materialized.materialized_at,
    )
    computed_results = _computed_gate_results(recomputed.findings)
    mismatch_evidence = _materialization_mismatches(materialized, recomputed) + (
        _manifest_mismatches(recomputed, manifest, computed_results)
    )
    manifest_finding = GateFinding(
        gate=HardGate.MANIFEST_EVIDENCE_MISMATCH,
        row_reference=None,
        evidence=mismatch_evidence,
        passed=not mismatch_evidence,
    )
    findings = recomputed.findings + (manifest_finding,)
    gate_results = dict(computed_results)
    gate_results[HardGate.MANIFEST_EVIDENCE_MISMATCH.value] = manifest_finding.passed

    failed_gates = {gate for gate, passed in gate_results.items() if not passed}
    failed_sample_ids = {
        finding.row_reference.split(":", 1)[0]
        for finding in findings
        if not finding.passed and finding.row_reference is not None
    }
    if failed_gates and not failed_sample_ids:
        violation_samples = list(recomputed.samples)
    else:
        violation_samples = [
            sample for sample in recomputed.samples if sample.sample_id in failed_sample_ids
        ]

    return SealReport(
        integrity_passed=not failed_gates,
        formal_passed=False,
        seal_kind="ENGINEERING",
        gate_results=gate_results,
        violation_count=len(failed_gates),
        violation_samples=violation_samples,
        findings=findings,
        sealed_manifest_hash=compute_manifest_hash(manifest),
    )


def seal_formal_dataset(
    materialized: MaterializedDataset,
    manifest: DatasetManifest,
    evidence: FormalDatasetEvidence,
) -> SealReport:
    """Seal a Dataset only when formal lineage/calendar evidence is complete."""

    engineering_report = seal_dataset(materialized, manifest)
    expected_manifest = bind_formal_dataset_evidence(manifest, materialized, evidence)
    formal_fields = (
        "formal",
        "source_manifest_hash",
        "source_manifest_uri",
        "source_manifest_schema_version",
        "artifact_uris",
        "lineage_fingerprint",
        "data_classification",
        "calendar_content_hash",
        "calendar_source_record_ids",
        "formal_evidence_fingerprint",
    )
    mismatches = _formal_evidence_mismatches(materialized, evidence) + _field_mismatches(
        expected_manifest,
        manifest,
        formal_fields,
    )
    formal_finding = GateFinding(
        gate=HardGate.MANIFEST_EVIDENCE_MISMATCH,
        row_reference=None,
        evidence=mismatches,
        passed=not mismatches,
    )
    audit_finding = GateFinding(
        gate=HardGate.FORMAL_AUDIT_ATTESTATION_REQUIRED,
        row_reference=None,
        evidence=(
            (
                "reason",
                "Task5C immutable full-audit attestation has not been bound",
            ),
        ),
        passed=False,
    )
    findings = engineering_report.findings + (formal_finding, audit_finding)
    gate_results = dict(engineering_report.gate_results)
    gate_results[HardGate.MANIFEST_EVIDENCE_MISMATCH.value] = (
        engineering_report.gate_results[HardGate.MANIFEST_EVIDENCE_MISMATCH.value]
        and formal_finding.passed
    )
    gate_results[HardGate.FORMAL_AUDIT_ATTESTATION_REQUIRED.value] = False
    integrity_passed = (
        engineering_report.integrity_passed and formal_finding.passed
    )
    failed_gates = {gate for gate, passed in gate_results.items() if not passed}
    return SealReport(
        integrity_passed=integrity_passed,
        formal_passed=False,
        seal_kind="FORMAL",
        gate_results=gate_results,
        violation_count=len(failed_gates),
        violation_samples=[] if integrity_passed else list(materialized.samples),
        findings=findings,
        sealed_manifest_hash=compute_manifest_hash(manifest),
    )
