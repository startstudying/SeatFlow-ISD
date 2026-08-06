"""Fail-closed Dataset v3 seal over computed materialization evidence."""

from __future__ import annotations

from dataclasses import dataclass

from seatflow.dataset.manifest import DatasetManifest, compute_manifest_hash
from seatflow.dataset.materializer import (
    MATERIALIZATION_GATES,
    GateFinding,
    HardGate,
    MaterializedDataset,
    materialize_dataset,
)
from seatflow.dataset.samples import PredictionSample


@dataclass(frozen=True)
class SealReport:
    """Result of binding and validating a materialized dataset manifest."""

    passed: bool
    gate_results: dict[str, bool]
    violation_count: int
    violation_samples: list[PredictionSample]
    findings: tuple[GateFinding, ...]
    sealed_manifest_hash: str


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
        passed=not failed_gates,
        gate_results=gate_results,
        violation_count=len(failed_gates),
        violation_samples=violation_samples,
        findings=findings,
        sealed_manifest_hash=compute_manifest_hash(manifest),
    )
