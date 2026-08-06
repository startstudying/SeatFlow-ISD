"""Persistence and integrity tests for point-in-time identity snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from seatflow.identity.repository import IdentityRepository
from seatflow.identity.seat_identity import SeatIdentity, SeatType
from seatflow.identity.stock_identity import StockIdentity


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _stock(stock_id: str = "stk-001", alias: str = "平安银行") -> StockIdentity:
    return StockIdentity(
        stock_id=stock_id,
        ts_code="000001.SZ" if stock_id == "stk-001" else "000002.SZ",
        exchange="SZ",
        board="MAIN",
        aliases=(alias,),
        valid_from=_utc("2020-01-01T00:00:00+00:00"),
        identity_confidence=0.91,
        source_evidence="tushare:stock_basic:manifest-001",
    )


def _seat(seat_id: str = "seat-001", alias: str = "机构专用") -> SeatIdentity:
    return SeatIdentity(
        seat_id=seat_id,
        canonical_name="机构专用" if seat_id == "seat-001" else "另一机构",
        seat_type=SeatType.INSTITUTION,
        aliases=(alias,),
        valid_from=_utc("2020-01-01T00:00:00+00:00"),
        identity_confidence=0.63,
        source_evidence="proxy:top_inst:coverage-limited",
    )


def _rehash_snapshot(snapshot: dict[str, object]) -> None:
    material = {key: value for key, value in snapshot.items() if key != "fingerprint"}
    canonical = json.dumps(
        material,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    snapshot["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_save_load_preserves_pit_evidence_confidence_and_fingerprint(tmp_path: Path) -> None:
    """Dropping persisted evidence or PIT intervals must change loaded behavior."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(_stock())
    repository.register_seat(_seat())

    fingerprint = repository.save(path)
    loaded = IdentityRepository.load(path)

    assert loaded.snapshot_fingerprint() == fingerprint
    stock = loaded.resolve_stock("平安银行", _utc("2025-01-01T00:00:00+00:00"))
    assert stock.identity_confidence == 0.91
    assert stock.source_evidence == "tushare:stock_basic:manifest-001"
    seat = loaded.resolve_seat("机构专用", _utc("2025-01-01T00:00:00+00:00"))
    assert seat.identity_confidence == 0.63
    assert seat.source_evidence == "proxy:top_inst:coverage-limited"


def test_snapshot_fingerprint_is_registration_order_invariant() -> None:
    """Serializing registry insertion order must not alter snapshot identity."""
    first = IdentityRepository()
    first.register_stock(_stock("stk-001", "平安银行"))
    first.register_stock(_stock("stk-002", "万科A"))
    second = IdentityRepository()
    second.register_stock(_stock("stk-002", "万科A"))
    second.register_stock(_stock("stk-001", "平安银行"))

    assert first.snapshot_fingerprint() == second.snapshot_fingerprint()


def test_load_rejects_corrupt_fingerprint(tmp_path: Path) -> None:
    """Skipping snapshot integrity verification must accept this corruption."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(_stock())
    repository.save(path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["stocks"][0]["identity_confidence"] = 0.01
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        IdentityRepository.load(path)


def test_load_rejects_overlapping_stock_alias_intervals(tmp_path: Path) -> None:
    """Loading without registry validation must admit ambiguous PIT resolution."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(_stock())
    repository.save(path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    conflicting = dict(snapshot["stocks"][0])
    conflicting["stock_id"] = "stk-999"
    conflicting["ts_code"] = "999999.SZ"
    snapshot["stocks"].append(conflicting)
    _rehash_snapshot(snapshot)
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Overlapping"):
        IdentityRepository.load(path)


def test_load_rejects_ambiguous_seat_alias_intervals(tmp_path: Path) -> None:
    """Loading two live seats for one alias must fail before resolution."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_seat(_seat())
    repository.save(path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    conflicting = dict(snapshot["seats"][0])
    conflicting["seat_id"] = "seat-999"
    conflicting["canonical_name"] = "另一机构"
    snapshot["seats"].append(conflicting)
    _rehash_snapshot(snapshot)
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Overlapping"):
        IdentityRepository.load(path)


def test_atomic_save_retains_recoverable_previous_snapshot(tmp_path: Path) -> None:
    """Overwriting without a backup must make primary corruption unrecoverable."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(_stock())
    first_fingerprint = repository.save(path)
    repository.register_seat(_seat())
    repository.save(path)
    path.write_text("{corrupt", encoding="utf-8")

    recovered = IdentityRepository.load(path, allow_backup_recovery=True)

    assert recovered.snapshot_fingerprint() == first_fingerprint
    assert recovered.resolve_stock(
        "000001.SZ", _utc("2025-01-01T00:00:00+00:00")
    ).stock_id == "stk-001"


def test_backup_recovery_does_not_mask_fingerprint_corruption(tmp_path: Path) -> None:
    """Recovery must not turn a detected integrity violation into a valid snapshot."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(_stock())
    repository.save(path)
    repository.register_seat(_seat())
    repository.save(path)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["fingerprint"] = "0" * 64
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        IdentityRepository.load(path, allow_backup_recovery=True)


def test_unknown_seat_fallback_keeps_low_confidence_evidence_after_load(tmp_path: Path) -> None:
    """Unknown fallback must remain explicit evidence, never a numeric identity feature."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.save(path)

    resolution = IdentityRepository.load(path).resolve_seat(
        "未收录营业部", _utc("2025-01-01T00:00:00+00:00")
    )

    assert resolution.resolved is False
    assert resolution.identity_confidence == 0.0
    assert resolution.seat_type == SeatType.UNKNOWN
    assert isinstance(resolution.seat_id, str)
    assert "unresolved" in resolution.source_evidence


def test_snapshot_roundtrip_preserves_explicitly_empty_legacy_evidence(tmp_path: Path) -> None:
    """Save must not emit a snapshot that its own loader cannot read."""
    path = tmp_path / "identity-snapshot.json"
    repository = IdentityRepository()
    repository.register_stock(
        StockIdentity(
            stock_id="stk-legacy",
            ts_code="600000.SH",
            exchange="SH",
            board="MAIN",
            valid_from=_utc("2020-01-01T00:00:00+00:00"),
            source_evidence="",
        )
    )

    repository.save(path)
    loaded = IdentityRepository.load(path)

    assert loaded.resolve_stock(
        "600000.SH", _utc("2025-01-01T00:00:00+00:00")
    ).source_evidence == ""
