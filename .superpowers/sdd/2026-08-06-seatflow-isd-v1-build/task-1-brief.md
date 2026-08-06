# Task 1: Deterministic Legacy Snapshot Inventory

## Global constraints

- No new dependencies.
- Source snapshots are read-only and untrusted until their row count and SHA-256 are recomputed.
- Existing files outside the owned paths must not be modified or reverted.
- Use strict TDD: add the test first, run it and observe the expected failure, then write minimal production code.

## Owned files

- Create `research-engine/src/seatflow/dataset/source_snapshots.py`.
- Create `research-engine/tests/dataset/test_source_snapshots.py`.

## Required interfaces

- `SnapshotRecord(dataset: str, trade_date: date, data_path: Path, manifest_path: Path, row_count: int, sha256: str)`.
- `discover_snapshot_records(root: Path, relative_dataset_path: Path) -> tuple[SnapshotRecord, ...]`.
- `load_verified_rows(record: SnapshotRecord) -> list[dict[str, object]]`.
- `build_source_inventory(root: Path, dataset_paths: Mapping[str, Path]) -> SourceInventory`.

## Required behaviors

1. Discover only `date=YYYY-MM-DD/data.json` plus `manifest.json` pairs and return them in ascending date order.
2. Parse every manifest as UTF-8 JSON and require matching dataset name, date, row count, and SHA-256.
3. `load_verified_rows` recomputes SHA-256 from the exact `data.json` bytes, decodes a JSON list of objects, and rejects tampering, non-list payloads, non-object rows, or row-count mismatch.
4. `SourceInventory` is immutable and deterministically summarizes dataset name, relative path, date range, date count, row count, byte count, and a canonical fingerprint over the verified records.
5. Tests derive literal expected values independently and exercise real files, with no mocks.

## Verification

- First run must fail because the module is absent or behavior is missing.
- Final commands:
  - `$env:PYTHONPATH='research-engine/src'; python -m pytest research-engine/tests/dataset/test_source_snapshots.py -q`
  - `python -m ruff check research-engine/src/seatflow/dataset/source_snapshots.py research-engine/tests/dataset/test_source_snapshots.py`
- Run `git diff --check`.
- Commit with a Lore-format message.

## Report contract

Write `.superpowers/sdd/2026-08-06-seatflow-isd-v1-build/task-1-report.md` containing:

- status (`DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`),
- failing-test command and expected failure,
- implementation summary,
- passing-test/lint commands and outputs,
- commit SHA,
- self-review concerns.
