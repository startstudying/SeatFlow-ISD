"""Atomic, fingerprinted persistence for point-in-time identity registries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from seatflow.identity.seat_identity import (
    SeatIdentity,
    SeatIdentityRegistry,
    SeatIdentityResolution,
    SeatType,
)
from seatflow.identity.stock_identity import StockIdentity, StockIdentityRegistry

_SCHEMA_VERSION = "seatflow.identity.snapshot.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IdentitySnapshotReadError(ValueError):
    """Recoverable failure caused by a truncated or unreadable snapshot."""


class IdentityRepository:
    """Repository for validated PIT stock and seat identity snapshots."""

    def __init__(self) -> None:
        self._stock_registry = StockIdentityRegistry()
        self._seat_registry = SeatIdentityRegistry()

    def register_stock(self, identity: StockIdentity) -> None:
        self._stock_registry.register(identity)

    def register_seat(self, identity: SeatIdentity) -> None:
        self._seat_registry.register(identity)

    def resolve_stock(self, raw_code: str, at_time: datetime) -> StockIdentity:
        return self._stock_registry.resolve_stock(raw_code, at_time)

    def resolve_seat(self, raw_name: str, at_time: datetime) -> SeatIdentityResolution:
        return self._seat_registry.resolve_seat(raw_name, at_time)

    def snapshot_fingerprint(self) -> str:
        """Return a deterministic fingerprint independent of registration order."""
        return _fingerprint(self._snapshot_material())

    def save(self, path: str | Path) -> str:
        """Atomically save a snapshot and retain the previous version as ``.bak``."""
        target = Path(path)
        material = self._snapshot_material()
        fingerprint = _fingerprint(material)
        snapshot = {**material, "fingerprint": fingerprint}
        serialized = (
            json.dumps(
                snapshot,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            _atomic_write(target.with_suffix(target.suffix + ".bak"), target.read_bytes())
        _atomic_write(target, serialized)
        return fingerprint

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_backup_recovery: bool = False,
    ) -> IdentityRepository:
        """Load and validate a snapshot, optionally recovering its prior version."""
        target = Path(path)
        try:
            return cls._load_exact(target)
        except (OSError, UnicodeError, IdentitySnapshotReadError):
            backup = target.with_suffix(target.suffix + ".bak")
            if not allow_backup_recovery or not backup.exists():
                raise
            return cls._load_exact(backup)

    @classmethod
    def _load_exact(cls, path: Path) -> IdentityRepository:
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IdentitySnapshotReadError(f"Invalid identity snapshot JSON: {path}") from exc
        if not isinstance(snapshot, dict):
            raise ValueError("Identity snapshot root must be an object")
        fingerprint = snapshot.get("fingerprint")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError("Identity snapshot fingerprint has invalid SHA-256 shape")
        material = {key: value for key, value in snapshot.items() if key != "fingerprint"}
        expected = _fingerprint(material)
        if not hmac.compare_digest(fingerprint, expected):
            raise ValueError("Identity snapshot fingerprint mismatch")
        if material.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported identity snapshot schema: {material.get('schema_version')!r}")

        repository = cls()
        stocks = material.get("stocks")
        seats = material.get("seats")
        if not isinstance(stocks, list) or not isinstance(seats, list):
            raise ValueError("Identity snapshot stocks and seats must be arrays")
        for item in stocks:
            repository.register_stock(_stock_from_dict(item))
        for item in seats:
            repository.register_seat(_seat_from_dict(item))
        return repository

    def _snapshot_material(self) -> dict[str, Any]:
        stocks = [_stock_to_dict(identity) for identity in self._stock_registry.identities]
        seats = [_seat_to_dict(identity) for identity in self._seat_registry.identities]
        stocks.sort(key=_canonical_json)
        seats.sort(key=_canonical_json)
        return {
            "schema_version": _SCHEMA_VERSION,
            "stocks": stocks,
            "seats": seats,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _fingerprint(material: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _datetime_to_json(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _datetime_from_json(value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 string or null")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _stock_to_dict(identity: StockIdentity) -> dict[str, Any]:
    return {
        "stock_id": identity.stock_id,
        "ts_code": identity.ts_code,
        "exchange": identity.exchange,
        "board": identity.board,
        "listing_date": _datetime_to_json(identity.listing_date),
        "delisting_date": _datetime_to_json(identity.delisting_date),
        "aliases": sorted(identity.aliases),
        "valid_from": _datetime_to_json(identity.valid_from),
        "valid_to": _datetime_to_json(identity.valid_to),
        "identity_confidence": identity.identity_confidence,
        "source_evidence": identity.source_evidence,
    }


def _seat_to_dict(identity: SeatIdentity) -> dict[str, Any]:
    return {
        "seat_id": identity.seat_id,
        "canonical_name": identity.canonical_name,
        "seat_type": identity.seat_type.value,
        "identity_confidence": identity.identity_confidence,
        "valid_from": _datetime_to_json(identity.valid_from),
        "valid_to": _datetime_to_json(identity.valid_to),
        "aliases": sorted(identity.aliases),
        "source_evidence": identity.source_evidence,
    }


def _require_dict(value: Any, entity: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{entity} identity entry must be an object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _aliases_from_json(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(alias, str) for alias in value):
        raise ValueError("aliases must be an array of strings")
    if len(value) != len(set(value)):
        raise ValueError("aliases must not contain duplicates")
    return tuple(value)


def _confidence_from_json(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("identity_confidence must be numeric")
    return float(value)


def _stock_from_dict(value: Any) -> StockIdentity:
    item = _require_dict(value, "Stock")
    return StockIdentity(
        stock_id=_require_string(item.get("stock_id"), "stock_id"),
        ts_code=_require_string(item.get("ts_code"), "ts_code"),
        exchange=_require_string(item.get("exchange"), "exchange"),
        board=_require_string(item.get("board"), "board"),
        listing_date=_datetime_from_json(item.get("listing_date"), "listing_date"),
        delisting_date=_datetime_from_json(item.get("delisting_date"), "delisting_date"),
        aliases=_aliases_from_json(item.get("aliases")),
        valid_from=_datetime_from_json(item.get("valid_from"), "valid_from"),
        valid_to=_datetime_from_json(item.get("valid_to"), "valid_to"),
        identity_confidence=_confidence_from_json(item.get("identity_confidence")),
        source_evidence=_require_text(item.get("source_evidence"), "source_evidence"),
    )


def _seat_from_dict(value: Any) -> SeatIdentity:
    item = _require_dict(value, "Seat")
    try:
        seat_type = SeatType(_require_string(item.get("seat_type"), "seat_type"))
    except ValueError as exc:
        raise ValueError(f"Unknown seat_type: {item.get('seat_type')!r}") from exc
    return SeatIdentity(
        seat_id=_require_string(item.get("seat_id"), "seat_id"),
        canonical_name=_require_string(item.get("canonical_name"), "canonical_name"),
        seat_type=seat_type,
        identity_confidence=_confidence_from_json(item.get("identity_confidence")),
        valid_from=_datetime_from_json(item.get("valid_from"), "valid_from"),
        valid_to=_datetime_from_json(item.get("valid_to"), "valid_to"),
        aliases=_aliases_from_json(item.get("aliases")),
        source_evidence=_require_text(item.get("source_evidence"), "source_evidence"),
    )
