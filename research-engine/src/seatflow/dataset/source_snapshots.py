"""Deterministic verification and inventory for immutable source snapshots."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping


_DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SnapshotRecord:
    dataset: str
    trade_date: date
    data_path: Path
    manifest_path: Path
    row_count: int
    sha256: str


@dataclass(frozen=True)
class SourceInventoryDataset:
    dataset: str
    relative_path: Path
    start_date: date
    end_date: date
    date_count: int
    row_count: int
    byte_count: int
    fingerprint: str


@dataclass(frozen=True)
class SourceInventory:
    datasets: tuple[SourceInventoryDataset, ...]
    fingerprint: str


def discover_snapshot_records(
    root: Path,
    relative_dataset_path: Path,
) -> tuple[SnapshotRecord, ...]:
    """Return verified complete snapshots, ordered by their trade date."""
    dataset_root = root / relative_dataset_path
    dataset_name = relative_dataset_path.name
    if not dataset_root.exists():
        return ()

    records: list[SnapshotRecord] = []
    for snapshot_dir in dataset_root.iterdir():
        if not snapshot_dir.is_dir():
            continue
        match = _DATE_DIR_RE.fullmatch(snapshot_dir.name)
        if match is None:
            continue
        data_path = snapshot_dir / "data.json"
        manifest_path = snapshot_dir / "manifest.json"
        if not data_path.is_file() or not manifest_path.is_file():
            continue

        trade_date = _parse_trade_date(match.group(1), snapshot_dir)
        manifest = _read_manifest(manifest_path)
        _require_manifest_value(manifest, "dataset", dataset_name, manifest_path)
        # Accept either "date" (legacy format) or "trade_date" (Tushare format).
        manifest_date = manifest.get("date")
        if manifest_date is None:
            manifest_date = manifest.get("trade_date")
        if manifest_date != trade_date.isoformat():
            raise ValueError(
                f"{manifest_path} date mismatch: manifest={manifest_date!r} "
                f"dir={trade_date.isoformat()!r}"
            )

        raw = data_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        manifest_sha256 = manifest.get("sha256")
        if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
            raise ValueError(f"{manifest_path} sha256 must be lowercase SHA-256 hex")
        if not hmac.compare_digest(manifest_sha256, actual_sha256):
            raise ValueError(f"{manifest_path} sha256 mismatch")

        rows = _decode_json_rows(raw, data_path)
        manifest_row_count = manifest.get("row_count")
        if not isinstance(manifest_row_count, int) or isinstance(manifest_row_count, bool):
            raise ValueError(f"{manifest_path} row count must be an integer")
        if manifest_row_count != len(rows):
            raise ValueError(f"{manifest_path} row count mismatch")

        records.append(
            SnapshotRecord(
                dataset=dataset_name,
                trade_date=trade_date,
                data_path=data_path,
                manifest_path=manifest_path,
                row_count=manifest_row_count,
                sha256=manifest_sha256,
            )
        )
    return tuple(sorted(records, key=lambda record: record.trade_date))


def discover_and_load_records(
    root: Path,
    relative_dataset_path: Path,
) -> list[tuple[SnapshotRecord, list[dict[str, object]]]]:
    """Discover snapshots and load their rows in a single pass.

    Unlike calling :func:`discover_snapshot_records` followed by
    :func:`load_verified_rows` per record, this function reads each
    file only once, avoiding the doubled I/O cost on large source
    windows (870+ dates × multiple datasets).
    """
    dataset_root = root / relative_dataset_path
    dataset_name = relative_dataset_path.name
    if not dataset_root.exists():
        return []

    results: list[tuple[SnapshotRecord, list[dict[str, object]]]] = []
    for snapshot_dir in dataset_root.iterdir():
        if not snapshot_dir.is_dir():
            continue
        match = _DATE_DIR_RE.fullmatch(snapshot_dir.name)
        if match is None:
            continue
        data_path = snapshot_dir / "data.json"
        manifest_path = snapshot_dir / "manifest.json"
        if not data_path.is_file() or not manifest_path.is_file():
            continue

        trade_date = _parse_trade_date(match.group(1), snapshot_dir)
        manifest = _read_manifest(manifest_path)
        _require_manifest_value(manifest, "dataset", dataset_name, manifest_path)
        manifest_date = manifest.get("date")
        if manifest_date is None:
            manifest_date = manifest.get("trade_date")
        if manifest_date != trade_date.isoformat():
            raise ValueError(
                f"{manifest_path} date mismatch: manifest={manifest_date!r} "
                f"dir={trade_date.isoformat()!r}"
            )

        raw = data_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        manifest_sha256 = manifest.get("sha256")
        if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
            raise ValueError(f"{manifest_path} sha256 must be lowercase SHA-256 hex")
        if not hmac.compare_digest(manifest_sha256, actual_sha256):
            raise ValueError(f"{manifest_path} sha256 mismatch")

        rows = _decode_json_rows(raw, data_path)
        manifest_row_count = manifest.get("row_count")
        if not isinstance(manifest_row_count, int) or isinstance(manifest_row_count, bool):
            raise ValueError(f"{manifest_path} row count must be an integer")
        if manifest_row_count != len(rows):
            raise ValueError(f"{manifest_path} row count mismatch")

        record = SnapshotRecord(
            dataset=dataset_name,
            trade_date=trade_date,
            data_path=data_path,
            manifest_path=manifest_path,
            row_count=manifest_row_count,
            sha256=manifest_sha256,
        )
        results.append((record, rows))
    results.sort(key=lambda pair: pair[0].trade_date)
    return results


def load_verified_rows(record: SnapshotRecord) -> list[dict[str, object]]:
    """Re-read a record and return its rows only when bytes still verify."""
    raw = record.data_path.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record.sha256):
        raise ValueError(f"{record.data_path} sha256 mismatch")
    rows = _decode_json_rows(raw, record.data_path)
    if len(rows) != record.row_count:
        raise ValueError(f"{record.data_path} row count mismatch")
    return rows


def build_source_inventory(root: Path, dataset_paths: Mapping[str, Path]) -> SourceInventory:
    """Build an immutable canonical summary from all verified source snapshots."""
    summaries: list[SourceInventoryDataset] = []
    for dataset, relative_path in sorted(dataset_paths.items()):
        if relative_path.name != dataset:
            raise ValueError(f"{relative_path} dataset name does not match {dataset}")
        records = discover_snapshot_records(root, relative_path)
        if not records:
            raise ValueError(f"{relative_path} has no verified source snapshots")
        for record in records:
            if record.dataset != dataset:
                raise ValueError(f"{record.manifest_path} dataset mismatch")
            load_verified_rows(record)
        summaries.append(_summarize_dataset(root, relative_path, records))

    payload = [
        {
            "dataset": summary.dataset,
            "relative_path": summary.relative_path.as_posix(),
            "start_date": summary.start_date.isoformat(),
            "end_date": summary.end_date.isoformat(),
            "date_count": summary.date_count,
            "row_count": summary.row_count,
            "byte_count": summary.byte_count,
            "fingerprint": summary.fingerprint,
        }
        for summary in summaries
    ]
    return SourceInventory(datasets=tuple(summaries), fingerprint=_canonical_hash(payload))


def _summarize_dataset(
    root: Path,
    relative_path: Path,
    records: tuple[SnapshotRecord, ...],
) -> SourceInventoryDataset:
    byte_count = 0
    row_count = 0
    payload: list[dict[str, object]] = []
    for record in records:
        record_byte_count = record.data_path.stat().st_size
        byte_count += record_byte_count
        row_count += record.row_count
        payload.append(
            {
                "dataset": record.dataset,
                "trade_date": record.trade_date.isoformat(),
                "relative_data_path": record.data_path.relative_to(root).as_posix(),
                "relative_manifest_path": record.manifest_path.relative_to(root).as_posix(),
                "row_count": record.row_count,
                "byte_count": record_byte_count,
                "sha256": record.sha256,
            }
        )
    return SourceInventoryDataset(
        dataset=records[0].dataset,
        relative_path=relative_path,
        start_date=records[0].trade_date,
        end_date=records[-1].trade_date,
        date_count=len(records),
        row_count=row_count,
        byte_count=byte_count,
        fingerprint=_canonical_hash(payload),
    )


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{manifest_path} manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} manifest must be a JSON object")
    return payload


def _require_manifest_value(
    manifest: Mapping[str, Any], key: str, expected: str, manifest_path: Path
) -> None:
    if manifest.get(key) != expected:
        raise ValueError(f"{manifest_path} {key} mismatch")


def _parse_trade_date(raw: str, snapshot_dir: Path) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{snapshot_dir} date directory is invalid") from exc


def _decode_json_rows(raw: bytes, data_path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{data_path} data payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{data_path} data payload must be a JSON list")
    rows: list[dict[str, object]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"{data_path} data row must be a JSON object")
        rows.append(row)
    return rows


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
