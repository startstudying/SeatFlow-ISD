"""Tests for the dataset manifest (Task 13).

The :class:`DatasetManifest` captures the full provenance of a built
dataset: every policy hash that could affect the rows (identity, cycle,
feature schema, label, split, graph) plus row/node/edge counts and
quality-gate results. ``compute_manifest_hash`` produces a deterministic
SHA-256 over the canonical JSON so two builds with identical inputs yield
identical hashes, and any change to a policy hash changes the digest.
"""

from __future__ import annotations

from datetime import datetime


from seatflow.dataset.manifest import (
    DatasetManifest,
    compute_manifest_hash,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _manifest(**overrides) -> DatasetManifest:
    base = dict(
        dataset_version="seatflow.v3.0",
        build_timestamp=_utc("2026-08-06T00:00:00+00:00"),
        source_manifest_hash="a" * 64,
        identity_policy_hash="b" * 64,
        cycle_policy_hash="c" * 64,
        feature_schema_hash="d" * 64,
        label_policy_hash="e" * 64,
        split_definition_hash="f" * 64,
        graph_schema_hash="0" * 64,
        row_counts={"samples": 1000, "labels": 1000},
        node_counts={"stock": 50, "seat": 200},
        edge_counts={"participates": 500},
        quality_gate_results={"seal": True},
        artifact_uris=["s3://seatflow/dataset/v3/0.parquet"],
    )
    base.update(overrides)
    return DatasetManifest(**base)


# ── Hash determinism ──────────────────────────────────────────────────


class TestManifestHash:
    def test_identical_manifests_yield_identical_hash(self):
        m1 = _manifest()
        m2 = _manifest()
        h1 = compute_manifest_hash(m1)
        h2 = compute_manifest_hash(m2)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_row_counts_yield_different_hash(self):
        m1 = _manifest(row_counts={"samples": 1000})
        m2 = _manifest(row_counts={"samples": 1001})
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_different_identity_policy_hash_yields_different_hash(self):
        m1 = _manifest(identity_policy_hash="b" * 64)
        m2 = _manifest(identity_policy_hash="z" * 64)
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_different_label_policy_hash_yields_different_hash(self):
        m1 = _manifest(label_policy_hash="e" * 64)
        m2 = _manifest(label_policy_hash="y" * 64)
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_different_split_definition_hash_yields_different_hash(self):
        m1 = _manifest(split_definition_hash="f" * 64)
        m2 = _manifest(split_definition_hash="x" * 64)
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_different_build_timestamp_yields_different_hash(self):
        m1 = _manifest(build_timestamp=_utc("2026-08-06T00:00:00+00:00"))
        m2 = _manifest(build_timestamp=_utc("2026-08-07T00:00:00+00:00"))
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_different_artifact_uris_yield_different_hash(self):
        m1 = _manifest(artifact_uris=["s3://a"])
        m2 = _manifest(artifact_uris=["s3://b"])
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_hash_is_hex_sha256(self):
        h = compute_manifest_hash(_manifest())
        assert len(h) == 64
        int(h, 16)  # parses as hex
