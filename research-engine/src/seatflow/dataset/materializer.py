"""Immutable Dataset v3 materialization evidence and computed hard gates.

The materializer is the only producer of seal evidence.  Callers provide
rows, not pass/fail booleans; every hard-gate result is derived from those
rows and included in deterministic fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from seatflow.dataset.samples import PredictionSample
from seatflow.features.registry import FORBIDDEN_ID_FEATURES
from seatflow.graph.schema import (
    EDGE_SCHEMAS,
    NODE_SCHEMAS,
    EdgeType,
    NodeType,
    get_edge_schema,
)


GRAPH_SCHEMA_CONTRACT_VERSION = "seatflow-frozen-graph-v1"
LABEL_HORIZON_TRADING_DAYS = 5


class HardGate(str, Enum):
    """A fail-closed Dataset v3 integrity gate."""

    PIT_AVAILABLE_TIME_VIOLATION = "PIT_AVAILABLE_TIME_VIOLATION"
    IMMATURE_LABEL = "IMMATURE_LABEL"
    PRIMARY_KEY_CONFLICT = "PRIMARY_KEY_CONFLICT"
    DANGLING_GRAPH_REFERENCE = "DANGLING_GRAPH_REFERENCE"
    GRAPH_TOPOLOGY_VIOLATION = "GRAPH_TOPOLOGY_VIOLATION"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    NON_FINITE_SEMANTIC_FEATURE = "NON_FINITE_SEMANTIC_FEATURE"
    IDENTITY_FEATURE_LEAKAGE = "IDENTITY_FEATURE_LEAKAGE"
    SUPPORT_ONLY_TARGET = "SUPPORT_ONLY_TARGET"
    MANIFEST_EVIDENCE_MISMATCH = "MANIFEST_EVIDENCE_MISMATCH"


MATERIALIZATION_GATES = tuple(
    gate for gate in HardGate if gate is not HardGate.MANIFEST_EVIDENCE_MISMATCH
)


@dataclass(frozen=True, order=True)
class FeatureField:
    """One versioned continuous semantic feature in the exact schema."""

    name: str
    version: str


@dataclass(frozen=True)
class MaterializedFeature:
    """A point-in-time feature value attached to one prediction sample."""

    sample_id: str
    name: str
    version: str
    value: float
    available_time: datetime


@dataclass(frozen=True)
class MaterializedLabel:
    """Label maturity evidence for one prediction sample."""

    sample_id: str
    evaluation_time: datetime
    horizon_end_time: datetime | None = None


@dataclass(frozen=True)
class GraphNodeEvidence:
    """A graph node and whether it exists only to support propagation."""

    node_id: str
    node_type: str
    is_support_only: bool = False


@dataclass(frozen=True)
class GraphEdgeEvidence:
    """Topology and PIT evidence for one graph edge."""

    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    available_time: datetime
    sample_id: str | None = None


@dataclass(frozen=True)
class GateFinding:
    """One immutable, row-addressable result emitted by a computed gate."""

    gate: HardGate
    row_reference: str | None
    evidence: tuple[tuple[str, str], ...]
    passed: bool


@dataclass(frozen=True)
class MaterializedDataset:
    """Immutable rows and evidence eligible for fail-closed sealing."""

    samples: tuple[PredictionSample, ...]
    evaluation_samples: tuple[PredictionSample, ...]
    labels: tuple[MaterializedLabel, ...]
    features: tuple[MaterializedFeature, ...]
    feature_schema: tuple[FeatureField, ...]
    graph_nodes: tuple[GraphNodeEvidence, ...]
    graph_edges: tuple[GraphEdgeEvidence, ...]
    sample_graph_references: Mapping[str, tuple[str, ...]]
    trade_calendar: tuple[datetime, ...]
    label_horizon_trading_days: int
    materialized_at: datetime
    findings: tuple[GateFinding, ...]
    feature_schema_hash: str
    graph_schema_hash: str
    gate_evidence_fingerprint: str
    evidence_fingerprint: str
    row_counts: Mapping[str, int]
    node_counts: Mapping[str, int]
    edge_counts: Mapping[str, int]


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_feature_schema_hash(schema: Sequence[FeatureField]) -> str:
    """Hash the exact, order-independent feature name/version contract."""

    return _canonical_hash(
        [{"name": field.name, "version": field.version} for field in sorted(schema)]
    )


def _compute_graph_schema_hash() -> str:
    """Hash the complete frozen graph contract, not observed row types."""

    return _canonical_hash(
        {
            "schema_version": GRAPH_SCHEMA_CONTRACT_VERSION,
            "nodes": [
                {
                    "node_type": node_type.value,
                    "properties": list(schema.properties),
                    "required": list(schema.required),
                }
                for node_type, schema in sorted(
                    NODE_SCHEMAS.items(), key=lambda item: item[0].value
                )
            ],
            "edges": [
                {
                    "edge_type": edge_type.value,
                    "source_type": schema.source_type.value,
                    "target_type": schema.target_type.value,
                    "properties": list(schema.properties),
                    "max_hops": schema.max_hops,
                    "top_k": schema.top_k,
                    "directed": schema.directed,
                }
                for edge_type, schema in sorted(
                    EDGE_SCHEMAS.items(), key=lambda item: item[0].value
                )
            ],
        }
    )


def _evidence(**values: object) -> tuple[tuple[str, str], ...]:
    def render(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    return tuple(sorted((key, render(value)) for key, value in values.items()))


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _is_forbidden_identity_feature(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in FORBIDDEN_ID_FEATURES
        or normalized.endswith("_id")
        or "hash" in normalized
    )


def _sample_payload(sample: PredictionSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "cycle_id": sample.cycle_id,
        "stock_id": sample.stock_id,
        "prediction_date": sample.prediction_date.isoformat(),
        "cutoff_time": sample.cutoff_time.isoformat(),
        "days_since_last_event": sample.days_since_last_event,
        "pool_type": sample.pool_type.value,
        "tradeability": sample.tradeability.value,
        "available_time": sample.available_time.isoformat(),
        "is_tradable": sample.is_tradable,
        "is_backtest_eligible": sample.is_backtest_eligible,
    }


def _finding_payload(finding: GateFinding) -> dict[str, object]:
    return {
        "gate": finding.gate.value,
        "row_reference": finding.row_reference,
        "evidence": list(finding.evidence),
        "passed": finding.passed,
    }


def materialize_dataset(
    *,
    samples: Sequence[PredictionSample],
    labels: Sequence[MaterializedLabel],
    features: Sequence[MaterializedFeature],
    feature_schema: Sequence[FeatureField],
    graph_nodes: Sequence[GraphNodeEvidence],
    graph_edges: Sequence[GraphEdgeEvidence],
    sample_graph_references: Mapping[str, Sequence[str]],
    trade_calendar: Sequence[datetime] = (),
    materialized_at: datetime,
) -> MaterializedDataset:
    """Compute immutable Dataset v3 gate evidence from materialized rows."""

    sample_rows = tuple(samples)
    label_rows = tuple(labels)
    feature_rows = tuple(features)
    schema_rows = tuple(sorted(feature_schema))
    node_rows = tuple(graph_nodes)
    edge_rows = tuple(graph_edges)
    calendar_rows = tuple(
        sorted(set(trade_calendar), key=lambda timestamp: timestamp.isoformat())
    )
    references = MappingProxyType(
        {
            sample_id: tuple(sorted(node_ids))
            for sample_id, node_ids in sorted(sample_graph_references.items())
        }
    )
    findings_by_gate: dict[HardGate, list[GateFinding]] = {
        gate: [] for gate in MATERIALIZATION_GATES
    }

    def fail(
        gate: HardGate,
        row_reference: str | None,
        **evidence: object,
    ) -> None:
        findings_by_gate[gate].append(
            GateFinding(gate, row_reference, _evidence(**evidence), False)
        )

    if not _aware(materialized_at):
        fail(
            HardGate.IMMATURE_LABEL,
            None,
            reason="materialized_at must be timezone-aware",
        )
    if not calendar_rows:
        fail(
            HardGate.IMMATURE_LABEL,
            None,
            reason="missing frozen exchange trading calendar",
        )
    for trading_day in calendar_rows:
        if not _aware(trading_day):
            fail(
                HardGate.IMMATURE_LABEL,
                None,
                reason="trading calendar timestamps must be timezone-aware",
                trading_day=trading_day,
            )
    if not sample_rows:
        fail(
            HardGate.PRIMARY_KEY_CONFLICT,
            None,
            reason="dataset contains no prediction samples",
        )

    samples_by_id: dict[str, PredictionSample] = {}
    composite_keys: set[tuple[str, str, str, str]] = set()
    for sample in sample_rows:
        if sample.sample_id in samples_by_id:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                sample.sample_id,
                key="sample_id",
            )
        else:
            samples_by_id[sample.sample_id] = sample
        composite = (
            sample.cycle_id,
            sample.stock_id,
            sample.prediction_date.isoformat(),
            sample.pool_type.value,
        )
        if composite in composite_keys:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                sample.sample_id,
                key="cycle_id,stock_id,prediction_date,pool_type",
            )
        composite_keys.add(composite)
        if not _aware(sample.available_time) or not _aware(sample.cutoff_time):
            fail(
                HardGate.PIT_AVAILABLE_TIME_VIOLATION,
                sample.sample_id,
                reason="sample timestamps must be timezone-aware",
            )
        elif sample.available_time > sample.cutoff_time:
            fail(
                HardGate.PIT_AVAILABLE_TIME_VIOLATION,
                sample.sample_id,
                available_time=sample.available_time,
                cutoff_time=sample.cutoff_time,
            )

    label_ids: set[str] = set()
    labels_by_sample: dict[str, MaterializedLabel] = {}
    aware_calendar = tuple(day for day in calendar_rows if _aware(day))
    horizon_end_by_sample: dict[str, datetime] = {}
    for sample in sample_rows:
        if not _aware(sample.prediction_date):
            fail(
                HardGate.IMMATURE_LABEL,
                sample.sample_id,
                reason="prediction_date must be timezone-aware",
            )
            continue
        if sample.prediction_date not in aware_calendar:
            fail(
                HardGate.IMMATURE_LABEL,
                sample.sample_id,
                reason="prediction_date missing from frozen trading calendar",
                prediction_date=sample.prediction_date,
            )
            continue
        horizon_days = [day for day in aware_calendar if day >= sample.prediction_date]
        if len(horizon_days) < LABEL_HORIZON_TRADING_DAYS:
            fail(
                HardGate.IMMATURE_LABEL,
                sample.sample_id,
                reason="frozen trading calendar does not cover full label horizon",
                available_days=len(horizon_days),
                required_days=LABEL_HORIZON_TRADING_DAYS,
            )
            continue
        horizon_end_by_sample[sample.sample_id] = horizon_days[
            LABEL_HORIZON_TRADING_DAYS - 1
        ]
    for label in label_rows:
        if label.sample_id in label_ids:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                label.sample_id,
                key="label.sample_id",
            )
        label_ids.add(label.sample_id)
        labels_by_sample[label.sample_id] = label
        label_sample = samples_by_id.get(label.sample_id)
        if label_sample is None:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                label.sample_id,
                reason="label references unknown sample",
            )
            continue
        if not _aware(label.evaluation_time):
            fail(
                HardGate.IMMATURE_LABEL,
                label.sample_id,
                reason="evaluation_time must be timezone-aware",
            )
            continue
        expected_horizon_end = horizon_end_by_sample.get(label.sample_id)
        if label.horizon_end_time is None:
            fail(
                HardGate.IMMATURE_LABEL,
                label.sample_id,
                reason="missing horizon_end_time maturity evidence",
            )
            continue
        if not _aware(label.horizon_end_time):
            fail(
                HardGate.IMMATURE_LABEL,
                label.sample_id,
                reason="horizon_end_time must be timezone-aware",
            )
            continue
        if expected_horizon_end is None:
            continue
        if label.horizon_end_time != expected_horizon_end:
            fail(
                HardGate.IMMATURE_LABEL,
                label.sample_id,
                reason="horizon_end_time does not match frozen trading calendar",
                actual_horizon_end=label.horizon_end_time,
                expected_horizon_end=expected_horizon_end,
            )
            continue
        if not _aware(materialized_at) or not (
            expected_horizon_end <= label.evaluation_time <= materialized_at
        ):
            fail(
                HardGate.IMMATURE_LABEL,
                label.sample_id,
                horizon_end_time=expected_horizon_end,
                evaluation_time=label.evaluation_time,
                materialized_at=materialized_at,
                reason="complete label horizon is not mature at materialization time",
            )
    for sample in sample_rows:
        if sample.sample_id not in labels_by_sample:
            fail(
                HardGate.IMMATURE_LABEL,
                sample.sample_id,
                reason="missing label maturity evidence",
            )

    expected_schema = {(field.name, field.version) for field in schema_rows}
    if len(expected_schema) != len(schema_rows):
        fail(
            HardGate.FEATURE_SCHEMA_MISMATCH,
            None,
            reason="duplicate feature schema field",
        )
    for field in schema_rows:
        if _is_forbidden_identity_feature(field.name):
            fail(
                HardGate.IDENTITY_FEATURE_LEAKAGE,
                f"schema:{field.name}",
                feature_name=field.name,
            )

    feature_keys: set[tuple[str, str, str]] = set()
    actual_schema_by_sample: dict[str, set[tuple[str, str]]] = {
        sample.sample_id: set() for sample in sample_rows
    }
    for feature in feature_rows:
        key = (feature.sample_id, feature.name, feature.version)
        if key in feature_keys:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                f"{feature.sample_id}:{feature.name}",
                key="sample_id,feature_name,version",
            )
        feature_keys.add(key)
        feature_sample = samples_by_id.get(feature.sample_id)
        if feature_sample is None:
            fail(
                HardGate.FEATURE_SCHEMA_MISMATCH,
                f"{feature.sample_id}:{feature.name}",
                reason="feature references unknown sample",
            )
            continue
        actual_schema_by_sample[feature.sample_id].add((feature.name, feature.version))
        row_reference = f"{feature.sample_id}:{feature.name}"
        if _is_forbidden_identity_feature(feature.name):
            fail(
                HardGate.IDENTITY_FEATURE_LEAKAGE,
                row_reference,
                feature_name=feature.name,
            )
        try:
            is_finite = not isinstance(feature.value, bool) and math.isfinite(
                float(feature.value)
            )
        except (TypeError, ValueError, OverflowError):
            is_finite = False
        if not is_finite:
            fail(
                HardGate.NON_FINITE_SEMANTIC_FEATURE,
                row_reference,
                value=feature.value,
            )
        if (
            not _aware(feature.available_time)
            or feature.available_time > feature_sample.cutoff_time
        ):
            fail(
                HardGate.PIT_AVAILABLE_TIME_VIOLATION,
                row_reference,
                available_time=feature.available_time,
                cutoff_time=feature_sample.cutoff_time,
            )
    for sample in sample_rows:
        actual = actual_schema_by_sample[sample.sample_id]
        if actual != expected_schema:
            fail(
                HardGate.FEATURE_SCHEMA_MISMATCH,
                sample.sample_id,
                actual=sorted(actual),
                expected=sorted(expected_schema),
            )

    node_ids: set[str] = set()
    node_types: dict[str, NodeType] = {}
    support_only_ids: set[str] = set()
    for node in node_rows:
        if node.node_id in node_ids:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                node.node_id,
                key="graph_node.node_id",
            )
        node_ids.add(node.node_id)
        try:
            node_types[node.node_id] = NodeType(node.node_type)
        except ValueError:
            fail(
                HardGate.GRAPH_TOPOLOGY_VIOLATION,
                node.node_id,
                node_type=node.node_type,
                reason="unknown node type",
            )
        if node.is_support_only:
            support_only_ids.add(node.node_id)

    edge_ids: set[str] = set()
    edge_counts: dict[str, int] = {}
    for edge in edge_rows:
        if edge.edge_id in edge_ids:
            fail(
                HardGate.PRIMARY_KEY_CONFLICT,
                edge.edge_id,
                key="graph_edge.edge_id",
            )
        edge_ids.add(edge.edge_id)
        edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1
        missing = sorted({edge.source_id, edge.target_id} - node_ids)
        if missing:
            fail(
                HardGate.GRAPH_TOPOLOGY_VIOLATION,
                edge.edge_id,
                missing_endpoints=missing,
            )
        try:
            edge_schema = get_edge_schema(EdgeType(edge.edge_type))
        except ValueError:
            fail(
                HardGate.GRAPH_TOPOLOGY_VIOLATION,
                edge.edge_id,
                edge_type=edge.edge_type,
                reason="unknown edge type",
            )
        else:
            source_type = node_types.get(edge.source_id)
            target_type = node_types.get(edge.target_id)
            if source_type is not None and target_type is not None and (
                source_type is not edge_schema.source_type
                or target_type is not edge_schema.target_type
            ):
                fail(
                    HardGate.GRAPH_TOPOLOGY_VIOLATION,
                    edge.edge_id,
                    actual=f"{source_type.value}->{target_type.value}",
                    expected=(
                        f"{edge_schema.source_type.value}->{edge_schema.target_type.value}"
                    ),
                )
        edge_cutoff = materialized_at
        if edge.sample_id is not None:
            edge_sample = samples_by_id.get(edge.sample_id)
            if edge_sample is None:
                fail(
                    HardGate.GRAPH_TOPOLOGY_VIOLATION,
                    edge.edge_id,
                    reason="edge references unknown sample",
                )
            else:
                edge_cutoff = edge_sample.cutoff_time
        if (
            not _aware(edge.available_time)
            or not _aware(edge_cutoff)
            or edge.available_time > edge_cutoff
        ):
            fail(
                HardGate.PIT_AVAILABLE_TIME_VIOLATION,
                edge.edge_id,
                available_time=edge.available_time,
                cutoff_time=edge_cutoff,
            )

    evaluation_samples: list[PredictionSample] = []
    for sample in sample_rows:
        if sample.stock_id in support_only_ids:
            fail(
                HardGate.SUPPORT_ONLY_TARGET,
                sample.sample_id,
                stock_id=sample.stock_id,
            )
        else:
            evaluation_samples.append(sample)
        referenced_ids = references.get(sample.sample_id)
        if referenced_ids is None:
            fail(
                HardGate.DANGLING_GRAPH_REFERENCE,
                sample.sample_id,
                reason="missing sample graph-reference evidence",
            )
            continue
        if sample.stock_id not in referenced_ids:
            fail(
                HardGate.DANGLING_GRAPH_REFERENCE,
                sample.sample_id,
                reason="sample stock node is not referenced",
                node_id=sample.stock_id,
            )
        for node_id in referenced_ids:
            if node_id not in node_ids:
                fail(
                    HardGate.DANGLING_GRAPH_REFERENCE,
                    sample.sample_id,
                    node_id=node_id,
                )
    for sample_id in references:
        if sample_id not in samples_by_id:
            fail(
                HardGate.DANGLING_GRAPH_REFERENCE,
                sample_id,
                reason="graph references unknown sample",
            )

    findings: list[GateFinding] = []
    for gate in MATERIALIZATION_GATES:
        gate_findings = findings_by_gate[gate]
        if gate_findings:
            findings.extend(gate_findings)
        else:
            findings.append(
                GateFinding(
                    gate=gate,
                    row_reference=None,
                    evidence=_evidence(checked_rows=len(sample_rows)),
                    passed=True,
                )
            )
    findings.sort(
        key=lambda finding: (
            finding.gate.value,
            finding.row_reference or "",
            finding.evidence,
            finding.passed,
        )
    )

    feature_schema_hash = compute_feature_schema_hash(schema_rows)
    graph_schema_hash = _compute_graph_schema_hash()
    gate_evidence_fingerprint = _canonical_hash(
        [_finding_payload(finding) for finding in findings]
    )
    evidence_fingerprint = _canonical_hash(
        {
            "samples": sorted(
                (_sample_payload(sample) for sample in sample_rows),
                key=lambda row: str(row["sample_id"]),
            ),
            "labels": sorted(
                (
                    {
                        "sample_id": label.sample_id,
                        "evaluation_time": label.evaluation_time.isoformat(),
                        "horizon_end_time": (
                            label.horizon_end_time.isoformat()
                            if label.horizon_end_time is not None
                            else None
                        ),
                    }
                    for label in label_rows
                ),
                key=lambda row: (row["sample_id"], row["evaluation_time"]),
            ),
            "features": sorted(
                (
                    {
                        "sample_id": feature.sample_id,
                        "name": feature.name,
                        "version": feature.version,
                        "value": str(feature.value),
                        "available_time": feature.available_time.isoformat(),
                    }
                    for feature in feature_rows
                ),
                key=lambda row: (row["sample_id"], row["name"], row["version"]),
            ),
            "feature_schema_hash": feature_schema_hash,
            "graph_nodes": sorted(
                (
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "is_support_only": node.is_support_only,
                    }
                    for node in node_rows
                ),
                key=lambda row: str(row["node_id"]),
            ),
            "graph_edges": sorted(
                (
                    {
                        "edge_id": edge.edge_id,
                        "edge_type": edge.edge_type,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "available_time": edge.available_time.isoformat(),
                        "sample_id": edge.sample_id,
                    }
                    for edge in edge_rows
                ),
                key=lambda row: str(row["edge_id"]),
            ),
            "graph_schema_hash": graph_schema_hash,
            "references": {
                sample_id: list(node_ids) for sample_id, node_ids in references.items()
            },
            "trade_calendar": [day.isoformat() for day in calendar_rows],
            "label_horizon_trading_days": LABEL_HORIZON_TRADING_DAYS,
            "materialized_at": materialized_at.isoformat(),
            "gate_evidence_fingerprint": gate_evidence_fingerprint,
        }
    )
    node_counts: dict[str, int] = {}
    for node in node_rows:
        node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1

    return MaterializedDataset(
        samples=sample_rows,
        evaluation_samples=tuple(evaluation_samples),
        labels=label_rows,
        features=feature_rows,
        feature_schema=schema_rows,
        graph_nodes=node_rows,
        graph_edges=edge_rows,
        sample_graph_references=references,
        trade_calendar=calendar_rows,
        label_horizon_trading_days=LABEL_HORIZON_TRADING_DAYS,
        materialized_at=materialized_at,
        findings=tuple(findings),
        feature_schema_hash=feature_schema_hash,
        graph_schema_hash=graph_schema_hash,
        gate_evidence_fingerprint=gate_evidence_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        row_counts=MappingProxyType(
            {
                "samples": len(sample_rows),
                "labels": len(label_rows),
                "features": len(feature_rows),
            }
        ),
        node_counts=MappingProxyType(dict(sorted(node_counts.items()))),
        edge_counts=MappingProxyType(dict(sorted(edge_counts.items()))),
    )
