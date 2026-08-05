"""Dataset manifest for provenance and integrity (Task 13).

The :class:`DatasetManifest` captures the full provenance of a built
dataset: every policy hash that could affect the rows (identity, cycle,
feature schema, label, split, graph) plus row/node/edge counts and
quality-gate results. ``compute_manifest_hash`` produces a deterministic
SHA-256 over the canonical JSON so two builds with identical inputs yield
identical hashes, and any change to a policy hash changes the digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class DatasetManifest:
    """Full provenance record for a built dataset.

    Attributes
    ----------
    dataset_version:
        Semantic version string (e.g. ``"seatflow.v3.0"``).
    build_timestamp:
        When the dataset was built (tz-aware UTC).
    source_manifest_hash:
        SHA-256 of the ingestion manifest (source data provenance).
    identity_policy_hash:
        SHA-256 of the identity-resolution configuration.
    cycle_policy_hash:
        SHA-256 of the cycle-construction configuration.
    feature_schema_hash:
        SHA-256 of the feature schema.
    label_policy_hash:
        SHA-256 of the label-computation configuration.
    split_definition_hash:
        SHA-256 of the split definition (fold boundaries).
    graph_schema_hash:
        SHA-256 of the graph schema (node/edge types).
    row_counts:
        Mapping of table name → row count.
    node_counts:
        Mapping of node type → node count.
    edge_counts:
        Mapping of edge type → edge count.
    quality_gate_results:
        Mapping of gate name → pass/fail.
    artifact_uris:
        List of URIs for the dataset's output artifacts.
    """

    dataset_version: str
    build_timestamp: datetime
    source_manifest_hash: str
    identity_policy_hash: str
    cycle_policy_hash: str
    feature_schema_hash: str
    label_policy_hash: str
    split_definition_hash: str
    graph_schema_hash: str
    row_counts: dict[str, int]
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    quality_gate_results: dict[str, bool]
    artifact_uris: list[str]


def compute_manifest_hash(manifest: DatasetManifest) -> str:
    """Compute a deterministic SHA-256 hash over a manifest's canonical JSON.

    The manifest is serialized to a JSON string with sorted keys,
    no ASCII escaping, and minimal separators. Datetime values are
    converted to ISO-8601 strings. This ensures that identical manifests
    always produce identical hashes.
    """
    raw = asdict(manifest)
    # Convert datetime objects to ISO strings for JSON serialization.
    raw["build_timestamp"] = manifest.build_timestamp.isoformat()
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
