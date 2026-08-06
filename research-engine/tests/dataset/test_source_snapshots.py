from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from seatflow.dataset.source_snapshots import (
    SnapshotRecord,
    build_source_inventory,
    discover_snapshot_records,
    load_verified_rows,
)


DRAGON_2026_01_01_BYTES = b'[{"id":3,"name":"gamma"}]\n'
DRAGON_2026_01_01_SHA = "68f4c404cfabf731ca131142edb3760bcd1f1a191acf78b277f7f8059a39f2f8"
DRAGON_2026_01_02_BYTES = b'[{"id":1,"name":"alpha"},{"id":2,"name":"beta"}]\n'
DRAGON_2026_01_02_SHA = "ace4ecc8e3450f2af6996f736151ec10ecec709b78bad11042b7fb39a19d8641"
SEAT_2026_01_03_BYTES = b'[{"id":9}]\n'
SEAT_2026_01_03_SHA = "34abf74e151cbe419f508568d4881b6b48c08400dc3c3b7f68ac16d533e7d233"


def _write_snapshot(
    root: Path,
    relative_dataset_path: Path,
    trade_date: str,
    data_bytes: bytes,
    *,
    dataset: str,
    row_count: int,
    sha256: str,
) -> Path:
    snapshot_dir = root / relative_dataset_path / f"date={trade_date}"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "data.json").write_bytes(data_bytes)
    manifest = {
        "dataset": dataset,
        "date": trade_date,
        "row_count": row_count,
        "sha256": sha256,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_dir


def test_discovers_only_complete_date_snapshots_in_ascending_order(tmp_path: Path) -> None:
    root = tmp_path
    dataset_path = Path("snapshots/dragon_tiger")
    _write_snapshot(
        root,
        dataset_path,
        "2026-01-02",
        DRAGON_2026_01_02_BYTES,
        dataset="dragon_tiger",
        row_count=2,
        sha256=DRAGON_2026_01_02_SHA,
    )
    _write_snapshot(
        root,
        dataset_path,
        "2026-01-01",
        DRAGON_2026_01_01_BYTES,
        dataset="dragon_tiger",
        row_count=1,
        sha256=DRAGON_2026_01_01_SHA,
    )
    (root / dataset_path / "date=2026-01-03").mkdir()
    (root / dataset_path / "date=bad").mkdir()
    (root / dataset_path / "2026-01-04").mkdir()

    records = discover_snapshot_records(root, dataset_path)

    assert records == (
        SnapshotRecord(
            dataset="dragon_tiger",
            trade_date=date(2026, 1, 1),
            data_path=root / dataset_path / "date=2026-01-01" / "data.json",
            manifest_path=root / dataset_path / "date=2026-01-01" / "manifest.json",
            row_count=1,
            sha256=DRAGON_2026_01_01_SHA,
        ),
        SnapshotRecord(
            dataset="dragon_tiger",
            trade_date=date(2026, 1, 2),
            data_path=root / dataset_path / "date=2026-01-02" / "data.json",
            manifest_path=root / dataset_path / "date=2026-01-02" / "manifest.json",
            row_count=2,
            sha256=DRAGON_2026_01_02_SHA,
        ),
    )


@pytest.mark.parametrize(
    ("manifest_override", "message"),
    [
        ({"dataset": "other"}, "dataset"),
        ({"date": "2026-01-09"}, "date"),
        ({"row_count": 99}, "row count"),
        ({"sha256": "0" * 64}, "sha"),
    ],
)
def test_discovery_rejects_manifest_mismatches(
    tmp_path: Path,
    manifest_override: dict[str, object],
    message: str,
) -> None:
    dataset_path = Path("snapshots/dragon_tiger")
    snapshot_dir = _write_snapshot(
        tmp_path,
        dataset_path,
        "2026-01-01",
        DRAGON_2026_01_01_BYTES,
        dataset="dragon_tiger",
        row_count=1,
        sha256=DRAGON_2026_01_01_SHA,
    )
    manifest = {
        "dataset": "dragon_tiger",
        "date": "2026-01-01",
        "row_count": 1,
        "sha256": DRAGON_2026_01_01_SHA,
        **manifest_override,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        discover_snapshot_records(tmp_path, dataset_path)


def test_load_verified_rows_returns_json_objects_from_untampered_bytes(tmp_path: Path) -> None:
    dataset_path = Path("snapshots/dragon_tiger")
    _write_snapshot(
        tmp_path,
        dataset_path,
        "2026-01-02",
        DRAGON_2026_01_02_BYTES,
        dataset="dragon_tiger",
        row_count=2,
        sha256=DRAGON_2026_01_02_SHA,
    )
    (record,) = discover_snapshot_records(tmp_path, dataset_path)

    assert load_verified_rows(record) == [
        {"id": 1, "name": "alpha"},
        {"id": 2, "name": "beta"},
    ]


@pytest.mark.parametrize(
    ("data_bytes", "row_count", "sha256", "message"),
    [
        (b'{"id":1}\n', 1, "51bc513113548e062ada62b03efea153cae2abdf46b051c551c3e62a4dfb88cf", "list"),
        (b'[1]\n', 1, "acc07b62f23f458923737c4cd4a66bd05d1e71eb4f384003baaf2dcc760d6349", "object"),
        (SEAT_2026_01_03_BYTES, 2, SEAT_2026_01_03_SHA, "row count"),
        (b'[{"id":10}]\n', 1, SEAT_2026_01_03_SHA, "sha"),
    ],
)
def test_load_verified_rows_rejects_untrusted_payloads(
    tmp_path: Path,
    data_bytes: bytes,
    row_count: int,
    sha256: str,
    message: str,
) -> None:
    data_path = tmp_path / "data.json"
    data_path.write_bytes(data_bytes)
    record = SnapshotRecord(
        dataset="institutional_seat",
        trade_date=date(2026, 1, 3),
        data_path=data_path,
        manifest_path=tmp_path / "manifest.json",
        row_count=row_count,
        sha256=sha256,
    )

    with pytest.raises(ValueError, match=message):
        load_verified_rows(record)


def test_build_source_inventory_summarizes_verified_records_deterministically(
    tmp_path: Path,
) -> None:
    _write_snapshot(
        tmp_path,
        Path("snapshots/dragon_tiger"),
        "2026-01-02",
        DRAGON_2026_01_02_BYTES,
        dataset="dragon_tiger",
        row_count=2,
        sha256=DRAGON_2026_01_02_SHA,
    )
    _write_snapshot(
        tmp_path,
        Path("snapshots/dragon_tiger"),
        "2026-01-01",
        DRAGON_2026_01_01_BYTES,
        dataset="dragon_tiger",
        row_count=1,
        sha256=DRAGON_2026_01_01_SHA,
    )
    _write_snapshot(
        tmp_path,
        Path("snapshots/institutional_seat"),
        "2026-01-03",
        SEAT_2026_01_03_BYTES,
        dataset="institutional_seat",
        row_count=1,
        sha256=SEAT_2026_01_03_SHA,
    )

    inventory = build_source_inventory(
        tmp_path,
        {
            "institutional_seat": Path("snapshots/institutional_seat"),
            "dragon_tiger": Path("snapshots/dragon_tiger"),
        },
    )

    assert inventory.fingerprint == "1f2f253ce75992a75579e63aa27e9582eff02bc72e5f44d8903518e7263c3a5b"
    assert inventory.datasets[0].dataset == "dragon_tiger"
    assert inventory.datasets[0].relative_path == Path("snapshots/dragon_tiger")
    assert inventory.datasets[0].start_date == date(2026, 1, 1)
    assert inventory.datasets[0].end_date == date(2026, 1, 2)
    assert inventory.datasets[0].date_count == 2
    assert inventory.datasets[0].row_count == 3
    assert inventory.datasets[0].byte_count == 75
    assert inventory.datasets[0].fingerprint == "5955be6878186ccad1e13ac1e5ca6ca3df884585a4431bdb2669c8b76079b44b"
    assert inventory.datasets[1].dataset == "institutional_seat"
    assert inventory.datasets[1].relative_path == Path("snapshots/institutional_seat")
    assert inventory.datasets[1].start_date == date(2026, 1, 3)
    assert inventory.datasets[1].end_date == date(2026, 1, 3)
    assert inventory.datasets[1].date_count == 1
    assert inventory.datasets[1].row_count == 1
    assert inventory.datasets[1].byte_count == 11
    assert inventory.datasets[1].fingerprint == "51f51a71199ae66bf3786581f1e007baa7d359eae8729da906723fca1f9ef1c6"

    with pytest.raises(FrozenInstanceError):
        inventory.datasets[0].row_count = 4
    with pytest.raises(FrozenInstanceError):
        inventory.fingerprint = "0" * 64
