# Commit list
3e97d8a Record Task 1 verification evidence
7d73596 Preserve trustworthy legacy snapshot provenance

# Diff stat
 .../task-1-report.md                               |  50 +++++
 .../src/seatflow/dataset/source_snapshots.py       | 220 ++++++++++++++++++
 .../tests/dataset/test_source_snapshots.py         | 246 +++++++++++++++++++++
 3 files changed, 516 insertions(+)

# Full diff
diff --git a/.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-1-report.md b/.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-1-report.md
new file mode 100644
index 0000000..dcedf47
--- /dev/null
+++ b/.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-1-report.md
@@ -0,0 +1,50 @@
+# Task 1 Report: Deterministic Legacy Snapshot Inventory
+
+## Status
+
+DONE
+
+## TDD evidence
+
+Failing-test command:
+
+```powershell
+$env:PYTHONPATH='research-engine/src'; C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m pytest research-engine/tests/dataset/test_source_snapshots.py -q
+```
+
+Expected failure observed: `ModuleNotFoundError: No module named 'seatflow.dataset.source_snapshots'` during test collection, before production implementation was added.
+
+## Implementation summary
+
+- Added immutable `SnapshotRecord`, `SourceInventoryDataset`, and `SourceInventory` records.
+- Discover complete `date=YYYY-MM-DD` source snapshot pairs in chronological order and validate UTF-8 JSON manifest identity, date, count, and digest fields.
+- Recompute the SHA-256 from exact data bytes before decoding verified JSON-object rows.
+- Build deterministic per-dataset and whole-inventory canonical SHA-256 fingerprints from verified records.
+
+## Passing verification
+
+```powershell
+$env:PYTHONPATH='research-engine/src'; C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m pytest research-engine/tests/dataset/test_source_snapshots.py -q
+```
+
+Output: `11 passed`.
+
+```powershell
+C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m ruff check research-engine/src/seatflow/dataset/source_snapshots.py research-engine/tests/dataset/test_source_snapshots.py
+```
+
+Output: `All checks passed!`.
+
+```powershell
+git diff --check
+```
+
+Output: success (no whitespace errors).
+
+## Commit SHA
+
+`7d73596` — `Preserve trustworthy legacy snapshot provenance`.
+
+## Self-review concerns
+
+- The inventory intentionally rejects a configured dataset with no complete, verified snapshots; this preserves fail-closed provenance but callers must supply populated source roots.
diff --git a/research-engine/src/seatflow/dataset/source_snapshots.py b/research-engine/src/seatflow/dataset/source_snapshots.py
new file mode 100644
index 0000000..1f0b389
--- /dev/null
+++ b/research-engine/src/seatflow/dataset/source_snapshots.py
@@ -0,0 +1,220 @@
+"""Deterministic verification and inventory for immutable source snapshots."""
+
+from __future__ import annotations
+
+import hashlib
+import hmac
+import json
+import re
+from dataclasses import dataclass
+from datetime import date
+from pathlib import Path
+from typing import Any, Mapping
+
+
+_DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")
+_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
+
+
+@dataclass(frozen=True)
+class SnapshotRecord:
+    dataset: str
+    trade_date: date
+    data_path: Path
+    manifest_path: Path
+    row_count: int
+    sha256: str
+
+
+@dataclass(frozen=True)
+class SourceInventoryDataset:
+    dataset: str
+    relative_path: Path
+    start_date: date
+    end_date: date
+    date_count: int
+    row_count: int
+    byte_count: int
+    fingerprint: str
+
+
+@dataclass(frozen=True)
+class SourceInventory:
+    datasets: tuple[SourceInventoryDataset, ...]
+    fingerprint: str
+
+
+def discover_snapshot_records(
+    root: Path,
+    relative_dataset_path: Path,
+) -> tuple[SnapshotRecord, ...]:
+    """Return verified complete snapshots, ordered by their trade date."""
+    dataset_root = root / relative_dataset_path
+    dataset_name = relative_dataset_path.name
+    if not dataset_root.exists():
+        return ()
+
+    records: list[SnapshotRecord] = []
+    for snapshot_dir in dataset_root.iterdir():
+        if not snapshot_dir.is_dir():
+            continue
+        match = _DATE_DIR_RE.fullmatch(snapshot_dir.name)
+        if match is None:
+            continue
+        data_path = snapshot_dir / "data.json"
+        manifest_path = snapshot_dir / "manifest.json"
+        if not data_path.is_file() or not manifest_path.is_file():
+            continue
+
+        trade_date = _parse_trade_date(match.group(1), snapshot_dir)
+        manifest = _read_manifest(manifest_path)
+        _require_manifest_value(manifest, "dataset", dataset_name, manifest_path)
+        _require_manifest_value(manifest, "date", trade_date.isoformat(), manifest_path)
+
+        raw = data_path.read_bytes()
+        actual_sha256 = hashlib.sha256(raw).hexdigest()
+        manifest_sha256 = manifest.get("sha256")
+        if not isinstance(manifest_sha256, str) or not _SHA256_RE.fullmatch(manifest_sha256):
+            raise ValueError(f"{manifest_path} sha256 must be lowercase SHA-256 hex")
+        if not hmac.compare_digest(manifest_sha256, actual_sha256):
+            raise ValueError(f"{manifest_path} sha256 mismatch")
+
+        rows = _decode_json_rows(raw, data_path)
+        manifest_row_count = manifest.get("row_count")
+        if not isinstance(manifest_row_count, int) or isinstance(manifest_row_count, bool):
+            raise ValueError(f"{manifest_path} row count must be an integer")
+        if manifest_row_count != len(rows):
+            raise ValueError(f"{manifest_path} row count mismatch")
+
+        records.append(
+            SnapshotRecord(
+                dataset=dataset_name,
+                trade_date=trade_date,
+                data_path=data_path,
+                manifest_path=manifest_path,
+                row_count=manifest_row_count,
+                sha256=manifest_sha256,
+            )
+        )
+    return tuple(sorted(records, key=lambda record: record.trade_date))
+
+
+def load_verified_rows(record: SnapshotRecord) -> list[dict[str, object]]:
+    """Re-read a record and return its rows only when bytes still verify."""
+    raw = record.data_path.read_bytes()
+    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), record.sha256):
+        raise ValueError(f"{record.data_path} sha256 mismatch")
+    rows = _decode_json_rows(raw, record.data_path)
+    if len(rows) != record.row_count:
+        raise ValueError(f"{record.data_path} row count mismatch")
+    return rows
+
+
+def build_source_inventory(root: Path, dataset_paths: Mapping[str, Path]) -> SourceInventory:
+    """Build an immutable canonical summary from all verified source snapshots."""
+    summaries: list[SourceInventoryDataset] = []
+    for dataset, relative_path in sorted(dataset_paths.items()):
+        if relative_path.name != dataset:
+            raise ValueError(f"{relative_path} dataset name does not match {dataset}")
+        records = discover_snapshot_records(root, relative_path)
+        if not records:
+            raise ValueError(f"{relative_path} has no verified source snapshots")
+        for record in records:
+            if record.dataset != dataset:
+                raise ValueError(f"{record.manifest_path} dataset mismatch")
+            load_verified_rows(record)
+        summaries.append(_summarize_dataset(root, relative_path, records))
+
+    payload = [
+        {
+            "dataset": summary.dataset,
+            "relative_path": summary.relative_path.as_posix(),
+            "start_date": summary.start_date.isoformat(),
+            "end_date": summary.end_date.isoformat(),
+            "date_count": summary.date_count,
+            "row_count": summary.row_count,
+            "byte_count": summary.byte_count,
+            "fingerprint": summary.fingerprint,
+        }
+        for summary in summaries
+    ]
+    return SourceInventory(datasets=tuple(summaries), fingerprint=_canonical_hash(payload))
+
+
+def _summarize_dataset(
+    root: Path,
+    relative_path: Path,
+    records: tuple[SnapshotRecord, ...],
+) -> SourceInventoryDataset:
+    byte_count = 0
+    row_count = 0
+    payload: list[dict[str, object]] = []
+    for record in records:
+        record_byte_count = record.data_path.stat().st_size
+        byte_count += record_byte_count
+        row_count += record.row_count
+        payload.append(
+            {
+                "dataset": record.dataset,
+                "trade_date": record.trade_date.isoformat(),
+                "relative_data_path": record.data_path.relative_to(root).as_posix(),
+                "relative_manifest_path": record.manifest_path.relative_to(root).as_posix(),
+                "row_count": record.row_count,
+                "byte_count": record_byte_count,
+                "sha256": record.sha256,
+            }
+        )
+    return SourceInventoryDataset(
+        dataset=records[0].dataset,
+        relative_path=relative_path,
+        start_date=records[0].trade_date,
+        end_date=records[-1].trade_date,
+        date_count=len(records),
+        row_count=row_count,
+        byte_count=byte_count,
+        fingerprint=_canonical_hash(payload),
+    )
+
+
+def _read_manifest(manifest_path: Path) -> dict[str, Any]:
+    try:
+        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
+    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
+        raise ValueError(f"{manifest_path} manifest is not valid UTF-8 JSON") from exc
+    if not isinstance(payload, dict):
+        raise ValueError(f"{manifest_path} manifest must be a JSON object")
+    return payload
+
+
+def _require_manifest_value(
+    manifest: Mapping[str, Any], key: str, expected: str, manifest_path: Path
+) -> None:
+    if manifest.get(key) != expected:
+        raise ValueError(f"{manifest_path} {key} mismatch")
+
+
+def _parse_trade_date(raw: str, snapshot_dir: Path) -> date:
+    try:
+        return date.fromisoformat(raw)
+    except ValueError as exc:
+        raise ValueError(f"{snapshot_dir} date directory is invalid") from exc
+
+
+def _decode_json_rows(raw: bytes, data_path: Path) -> list[dict[str, object]]:
+    try:
+        payload = json.loads(raw.decode("utf-8"))
+    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
+        raise ValueError(f"{data_path} data payload is not valid UTF-8 JSON") from exc
+    if not isinstance(payload, list):
+        raise ValueError(f"{data_path} data payload must be a JSON list")
+    rows: list[dict[str, object]] = []
+    for row in payload:
+        if not isinstance(row, dict):
+            raise ValueError(f"{data_path} data row must be a JSON object")
+        rows.append(row)
+    return rows
+
+
+def _canonical_hash(payload: object) -> str:
+    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
+    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
diff --git a/research-engine/tests/dataset/test_source_snapshots.py b/research-engine/tests/dataset/test_source_snapshots.py
new file mode 100644
index 0000000..7dfe1ee
--- /dev/null
+++ b/research-engine/tests/dataset/test_source_snapshots.py
@@ -0,0 +1,246 @@
+from __future__ import annotations
+
+import json
+from dataclasses import FrozenInstanceError
+from datetime import date
+from pathlib import Path
+
+import pytest
+
+from seatflow.dataset.source_snapshots import (
+    SnapshotRecord,
+    build_source_inventory,
+    discover_snapshot_records,
+    load_verified_rows,
+)
+
+
+DRAGON_2026_01_01_BYTES = b'[{"id":3,"name":"gamma"}]\n'
+DRAGON_2026_01_01_SHA = "68f4c404cfabf731ca131142edb3760bcd1f1a191acf78b277f7f8059a39f2f8"
+DRAGON_2026_01_02_BYTES = b'[{"id":1,"name":"alpha"},{"id":2,"name":"beta"}]\n'
+DRAGON_2026_01_02_SHA = "ace4ecc8e3450f2af6996f736151ec10ecec709b78bad11042b7fb39a19d8641"
+SEAT_2026_01_03_BYTES = b'[{"id":9}]\n'
+SEAT_2026_01_03_SHA = "34abf74e151cbe419f508568d4881b6b48c08400dc3c3b7f68ac16d533e7d233"
+
+
+def _write_snapshot(
+    root: Path,
+    relative_dataset_path: Path,
+    trade_date: str,
+    data_bytes: bytes,
+    *,
+    dataset: str,
+    row_count: int,
+    sha256: str,
+) -> Path:
+    snapshot_dir = root / relative_dataset_path / f"date={trade_date}"
+    snapshot_dir.mkdir(parents=True)
+    (snapshot_dir / "data.json").write_bytes(data_bytes)
+    manifest = {
+        "dataset": dataset,
+        "date": trade_date,
+        "row_count": row_count,
+        "sha256": sha256,
+    }
+    (snapshot_dir / "manifest.json").write_text(
+        json.dumps(manifest, sort_keys=True),
+        encoding="utf-8",
+    )
+    return snapshot_dir
+
+
+def test_discovers_only_complete_date_snapshots_in_ascending_order(tmp_path: Path) -> None:
+    root = tmp_path
+    dataset_path = Path("snapshots/dragon_tiger")
+    _write_snapshot(
+        root,
+        dataset_path,
+        "2026-01-02",
+        DRAGON_2026_01_02_BYTES,
+        dataset="dragon_tiger",
+        row_count=2,
+        sha256=DRAGON_2026_01_02_SHA,
+    )
+    _write_snapshot(
+        root,
+        dataset_path,
+        "2026-01-01",
+        DRAGON_2026_01_01_BYTES,
+        dataset="dragon_tiger",
+        row_count=1,
+        sha256=DRAGON_2026_01_01_SHA,
+    )
+    (root / dataset_path / "date=2026-01-03").mkdir()
+    (root / dataset_path / "date=bad").mkdir()
+    (root / dataset_path / "2026-01-04").mkdir()
+
+    records = discover_snapshot_records(root, dataset_path)
+
+    assert records == (
+        SnapshotRecord(
+            dataset="dragon_tiger",
+            trade_date=date(2026, 1, 1),
+            data_path=root / dataset_path / "date=2026-01-01" / "data.json",
+            manifest_path=root / dataset_path / "date=2026-01-01" / "manifest.json",
+            row_count=1,
+            sha256=DRAGON_2026_01_01_SHA,
+        ),
+        SnapshotRecord(
+            dataset="dragon_tiger",
+            trade_date=date(2026, 1, 2),
+            data_path=root / dataset_path / "date=2026-01-02" / "data.json",
+            manifest_path=root / dataset_path / "date=2026-01-02" / "manifest.json",
+            row_count=2,
+            sha256=DRAGON_2026_01_02_SHA,
+        ),
+    )
+
+
+@pytest.mark.parametrize(
+    ("manifest_override", "message"),
+    [
+        ({"dataset": "other"}, "dataset"),
+        ({"date": "2026-01-09"}, "date"),
+        ({"row_count": 99}, "row count"),
+        ({"sha256": "0" * 64}, "sha"),
+    ],
+)
+def test_discovery_rejects_manifest_mismatches(
+    tmp_path: Path,
+    manifest_override: dict[str, object],
+    message: str,
+) -> None:
+    dataset_path = Path("snapshots/dragon_tiger")
+    snapshot_dir = _write_snapshot(
+        tmp_path,
+        dataset_path,
+        "2026-01-01",
+        DRAGON_2026_01_01_BYTES,
+        dataset="dragon_tiger",
+        row_count=1,
+        sha256=DRAGON_2026_01_01_SHA,
+    )
+    manifest = {
+        "dataset": "dragon_tiger",
+        "date": "2026-01-01",
+        "row_count": 1,
+        "sha256": DRAGON_2026_01_01_SHA,
+        **manifest_override,
+    }
+    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
+
+    with pytest.raises(ValueError, match=message):
+        discover_snapshot_records(tmp_path, dataset_path)
+
+
+def test_load_verified_rows_returns_json_objects_from_untampered_bytes(tmp_path: Path) -> None:
+    dataset_path = Path("snapshots/dragon_tiger")
+    _write_snapshot(
+        tmp_path,
+        dataset_path,
+        "2026-01-02",
+        DRAGON_2026_01_02_BYTES,
+        dataset="dragon_tiger",
+        row_count=2,
+        sha256=DRAGON_2026_01_02_SHA,
+    )
+    (record,) = discover_snapshot_records(tmp_path, dataset_path)
+
+    assert load_verified_rows(record) == [
+        {"id": 1, "name": "alpha"},
+        {"id": 2, "name": "beta"},
+    ]
+
+
+@pytest.mark.parametrize(
+    ("data_bytes", "row_count", "sha256", "message"),
+    [
+        (b'{"id":1}\n', 1, "51bc513113548e062ada62b03efea153cae2abdf46b051c551c3e62a4dfb88cf", "list"),
+        (b'[1]\n', 1, "acc07b62f23f458923737c4cd4a66bd05d1e71eb4f384003baaf2dcc760d6349", "object"),
+        (SEAT_2026_01_03_BYTES, 2, SEAT_2026_01_03_SHA, "row count"),
+        (b'[{"id":10}]\n', 1, SEAT_2026_01_03_SHA, "sha"),
+    ],
+)
+def test_load_verified_rows_rejects_untrusted_payloads(
+    tmp_path: Path,
+    data_bytes: bytes,
+    row_count: int,
+    sha256: str,
+    message: str,
+) -> None:
+    data_path = tmp_path / "data.json"
+    data_path.write_bytes(data_bytes)
+    record = SnapshotRecord(
+        dataset="institutional_seat",
+        trade_date=date(2026, 1, 3),
+        data_path=data_path,
+        manifest_path=tmp_path / "manifest.json",
+        row_count=row_count,
+        sha256=sha256,
+    )
+
+    with pytest.raises(ValueError, match=message):
+        load_verified_rows(record)
+
+
+def test_build_source_inventory_summarizes_verified_records_deterministically(
+    tmp_path: Path,
+) -> None:
+    _write_snapshot(
+        tmp_path,
+        Path("snapshots/dragon_tiger"),
+        "2026-01-02",
+        DRAGON_2026_01_02_BYTES,
+        dataset="dragon_tiger",
+        row_count=2,
+        sha256=DRAGON_2026_01_02_SHA,
+    )
+    _write_snapshot(
+        tmp_path,
+        Path("snapshots/dragon_tiger"),
+        "2026-01-01",
+        DRAGON_2026_01_01_BYTES,
+        dataset="dragon_tiger",
+        row_count=1,
+        sha256=DRAGON_2026_01_01_SHA,
+    )
+    _write_snapshot(
+        tmp_path,
+        Path("snapshots/institutional_seat"),
+        "2026-01-03",
+        SEAT_2026_01_03_BYTES,
+        dataset="institutional_seat",
+        row_count=1,
+        sha256=SEAT_2026_01_03_SHA,
+    )
+
+    inventory = build_source_inventory(
+        tmp_path,
+        {
+            "institutional_seat": Path("snapshots/institutional_seat"),
+            "dragon_tiger": Path("snapshots/dragon_tiger"),
+        },
+    )
+
+    assert inventory.fingerprint == "1f2f253ce75992a75579e63aa27e9582eff02bc72e5f44d8903518e7263c3a5b"
+    assert inventory.datasets[0].dataset == "dragon_tiger"
+    assert inventory.datasets[0].relative_path == Path("snapshots/dragon_tiger")
+    assert inventory.datasets[0].start_date == date(2026, 1, 1)
+    assert inventory.datasets[0].end_date == date(2026, 1, 2)
+    assert inventory.datasets[0].date_count == 2
+    assert inventory.datasets[0].row_count == 3
+    assert inventory.datasets[0].byte_count == 75
+    assert inventory.datasets[0].fingerprint == "5955be6878186ccad1e13ac1e5ca6ca3df884585a4431bdb2669c8b76079b44b"
+    assert inventory.datasets[1].dataset == "institutional_seat"
+    assert inventory.datasets[1].relative_path == Path("snapshots/institutional_seat")
+    assert inventory.datasets[1].start_date == date(2026, 1, 3)
+    assert inventory.datasets[1].end_date == date(2026, 1, 3)
+    assert inventory.datasets[1].date_count == 1
+    assert inventory.datasets[1].row_count == 1
+    assert inventory.datasets[1].byte_count == 11
+    assert inventory.datasets[1].fingerprint == "51f51a71199ae66bf3786581f1e007baa7d359eae8729da906723fca1f9ef1c6"
+
+    with pytest.raises(FrozenInstanceError):
+        inventory.datasets[0].row_count = 4
+    with pytest.raises(FrozenInstanceError):
+        inventory.fingerprint = "0" * 64
