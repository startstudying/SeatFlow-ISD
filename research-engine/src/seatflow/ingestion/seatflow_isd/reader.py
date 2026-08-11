"""Verified reader for SeatFlow-ISD V1.0 datasets.

Reads the engineering dataset at
``<dataset_root>/<dataset_version>/`` and validates every file against
the SHA-256 hashes recorded in ``manifests/dataset_manifest.json``.

The reader never falls back to recomputing hashes from raw bytes
without comparing them to the manifest — a tamper or copy error
immediately fails the load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from seatflow.domain.time import assert_aware


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    assert_aware(dt, "datetime")
    return dt


@dataclass(frozen=True)
class ManifestVerification:
    """The result of verifying one dataset_manifest.json."""

    manifest_path: Path
    manifest_hash: str
    file_count: int
    file_hashes_verified: bool
    mismatches: tuple[str, ...]


def verify_manifest(dataset_root: Path) -> ManifestVerification:
    """Recompute every file hash referenced by ``dataset_manifest.json``."""
    manifest_path = dataset_root / "manifests" / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found at {manifest_path}")
    manifest = _read_json(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    mismatches: list[str] = []
    for entry in manifest.get("files", []):
        rel = entry["relative_path"]
        expected = entry["sha256"]
        actual_path = dataset_root / rel
        if not actual_path.exists():
            mismatches.append(f"{rel}: file missing")
            continue
        actual = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"{rel}: hash mismatch")
    return ManifestVerification(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        file_count=len(manifest.get("files", [])),
        file_hashes_verified=len(mismatches) == 0,
        mismatches=tuple(mismatches),
    )


def load_isd_dataset(
    dataset_root: Path,
    dataset_version: str,
) -> dict[str, Any]:
    """Load every record from a verified SeatFlow-ISD V1 dataset.

    Returns a dict with raw JSONL rows for every contract. Conversion
    to V3 dataclasses is handled by :mod:`adapter`.

    The caller is responsible for keeping the returned dict immutable
    (e.g. by converting to :class:`ISDLoadResult` immediately).
    """
    version_dir = dataset_root / dataset_version
    if not version_dir.is_dir():
        raise FileNotFoundError(f"dataset version not found: {version_dir}")
    manifest = verify_manifest(version_dir)
    if not manifest.file_hashes_verified:
        raise ValueError(
            f"manifest hash mismatch in {manifest.manifest_path}: "
            f"{list(manifest.mismatches)}"
        )

    silver = version_dir / "silver"
    gold = version_dir / "gold"
    samples_dir = version_dir / "samples"
    labels_dir = version_dir / "labels"

    return {
        "dataset_version": dataset_version,
        "dataset_root": str(version_dir),
        "manifest_hash": manifest.manifest_hash,
        "manifest_path": str(manifest.manifest_path),
        "manifest_verified": manifest.file_hashes_verified,
        "events": _read_jsonl(silver / "events.jsonl"),
        "participations": _read_jsonl(silver / "participations.jsonl"),
        "market_bars": _read_jsonl(silver / "market_bars.jsonl"),
        "daily_basic": _read_jsonl(silver / "daily_basic.jsonl"),
        "universe": _read_json(gold / "lhb_stock_universe.json"),
        "cycles": _read_jsonl(gold / "cycles.jsonl"),
        "samples": _read_jsonl(samples_dir / "direct_samples.jsonl"),
        "labels": _read_jsonl(labels_dir / "label_rows.jsonl"),
        "quality_report": _read_json(
            version_dir / "quality_reports" / "quality_report.json"
        ),
    }


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return a ``row_counts``-shaped summary of an iterable of records."""
    rows = list(records)
    return {"row_count": len(rows)}
