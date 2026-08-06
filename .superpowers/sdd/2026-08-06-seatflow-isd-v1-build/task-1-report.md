# Task 1 Report: Deterministic Legacy Snapshot Inventory

## Status

DONE

## TDD evidence

Failing-test command:

```powershell
$env:PYTHONPATH='research-engine/src'; C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m pytest research-engine/tests/dataset/test_source_snapshots.py -q
```

Expected failure observed: `ModuleNotFoundError: No module named 'seatflow.dataset.source_snapshots'` during test collection, before production implementation was added.

## Implementation summary

- Added immutable `SnapshotRecord`, `SourceInventoryDataset`, and `SourceInventory` records.
- Discover complete `date=YYYY-MM-DD` source snapshot pairs in chronological order and validate UTF-8 JSON manifest identity, date, count, and digest fields.
- Recompute the SHA-256 from exact data bytes before decoding verified JSON-object rows.
- Build deterministic per-dataset and whole-inventory canonical SHA-256 fingerprints from verified records.

## Passing verification

```powershell
$env:PYTHONPATH='research-engine/src'; C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m pytest research-engine/tests/dataset/test_source_snapshots.py -q
```

Output: `11 passed`.

```powershell
C:\Users\wonderful\AppData\Local\Programs\Python\Python312\python.exe -m ruff check research-engine/src/seatflow/dataset/source_snapshots.py research-engine/tests/dataset/test_source_snapshots.py
```

Output: `All checks passed!`.

```powershell
git diff --check
```

Output: success (no whitespace errors).

## Commit SHA

`7d73596` — `Preserve trustworthy legacy snapshot provenance`.

## Self-review concerns

- The inventory intentionally rejects a configured dataset with no complete, verified snapshots; this preserves fail-closed provenance but callers must supply populated source roots.
