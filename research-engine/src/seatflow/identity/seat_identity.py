"""Point-in-Time seat identity resolution.

Seats (broker branches, institutions, northbound channels) may be renamed,
merged, or reclassified over time. This module resolves a raw seat name to
the canonical identity valid at a given point in time, ensuring that
historical records are not overwritten by current aliases.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SeatType(str, Enum):
    BROKER_BRANCH = "BROKER_BRANCH"
    INSTITUTION = "INSTITUTION"
    NORTHBOUND_SH = "NORTHBOUND_SH"
    NORTHBOUND_SZ = "NORTHBOUND_SZ"
    SPECIAL_SEAT = "SPECIAL_SEAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SeatIdentity:
    """Canonical seat identity."""

    seat_id: str
    canonical_name: str
    seat_type: SeatType
    identity_confidence: float = 1.0
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source_evidence: str = ""


@dataclass(frozen=True)
class SeatIdentityResolution:
    """Result of resolving a raw seat name at a point in time."""

    seat_id: str
    canonical_name: str
    seat_type: SeatType
    identity_confidence: float
    raw_name: str
    resolved: bool
    aliases: tuple[str, ...] = field(default_factory=tuple)
    source_evidence: str = ""


class SeatIdentityRegistry:
    """Point-in-Time registry of seat identities."""

    def __init__(self) -> None:
        self._identities: list[SeatIdentity] = []

    def register(self, identity: SeatIdentity) -> None:
        self._validate_identity(identity)
        self._validate_no_overlap(identity)
        self._identities.append(identity)

    @property
    def identities(self) -> tuple[SeatIdentity, ...]:
        """Return an immutable view used by snapshot persistence."""
        return tuple(self._identities)

    @staticmethod
    def _validate_identity(identity: SeatIdentity) -> None:
        for label, value in (("valid_from", identity.valid_from), ("valid_to", identity.valid_to)):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{label} must be timezone-aware")
        if (
            identity.valid_from is not None
            and identity.valid_to is not None
            and identity.valid_from >= identity.valid_to
        ):
            raise ValueError("valid_from must be earlier than valid_to")
        if not 0.0 <= identity.identity_confidence <= 1.0:
            raise ValueError("identity_confidence must be in [0, 1]")

    def _validate_no_overlap(self, new: SeatIdentity) -> None:
        new_keys = {_normalize_name(name) for name in (new.canonical_name, *new.aliases)}
        new_keys.add(new.seat_id)
        for existing in self._identities:
            existing_keys = {
                _normalize_name(name)
                for name in (existing.canonical_name, *existing.aliases)
            }
            existing_keys.add(existing.seat_id)
            shared = existing_keys & new_keys
            if not shared:
                continue
            if self._intervals_overlap(existing, new):
                raise ValueError(
                    f"Overlapping validity intervals for seat lookup keys {shared}: "
                    f"{existing.seat_id} vs {new.seat_id}"
                )

    @staticmethod
    def _intervals_overlap(a: SeatIdentity, b: SeatIdentity) -> bool:
        a_start = a.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        a_end = a.valid_to or datetime.max.replace(tzinfo=timezone.utc)
        b_start = b.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        b_end = b.valid_to or datetime.max.replace(tzinfo=timezone.utc)
        return a_start < b_end and b_start < a_end

    def resolve_seat(
        self, raw_name: str, at_time: datetime
    ) -> SeatIdentityResolution:
        """Resolve a raw seat name to the canonical identity at ``at_time``.

        Raises ``KeyError`` if the name is known but outside its validity
        period. Returns an UNKNOWN resolution if the name is completely
        unregistered.
        """
        normalized = _normalize_name(raw_name)
        name_known = False
        matches: list[SeatIdentity] = []
        for identity in self._identities:
            identity_names = {
                _normalize_name(name)
                for name in (identity.canonical_name, *identity.aliases)
            }
            if normalized not in identity_names and raw_name != identity.seat_id:
                continue
            name_known = True
            if identity.valid_from and at_time < identity.valid_from:
                continue
            if identity.valid_to and at_time >= identity.valid_to:
                continue
            matches.append(identity)

        if len(matches) == 1:
            ident = matches[0]
            return SeatIdentityResolution(
                seat_id=ident.seat_id,
                canonical_name=ident.canonical_name,
                seat_type=ident.seat_type,
                identity_confidence=ident.identity_confidence,
                raw_name=raw_name,
                resolved=True,
                aliases=ident.aliases,
                source_evidence=ident.source_evidence,
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple seat identities found for {raw_name!r} at {at_time}"
            )

        if name_known and not matches:
            raise KeyError(
                f"Seat {raw_name!r} is known but outside its validity period at {at_time}"
            )

        # Unresolved: generate a stable UNKNOWN identity
        unknown_id = _generate_unknown_id(raw_name)
        return SeatIdentityResolution(
            seat_id=unknown_id,
            canonical_name=raw_name,
            seat_type=SeatType.UNKNOWN,
            identity_confidence=0.0,
            raw_name=raw_name,
            resolved=False,
            source_evidence="unresolved",
        )


def _normalize_name(name: str) -> str:
    """Normalize a seat name: strip whitespace, punctuation, branch suffixes."""
    s = name.strip()
    # Remove common branch suffixes
    s = re.sub(r"[#（\(].*?$", "", s)
    # Normalize whitespace
    s = re.sub(r"\s+", "", s)
    # Normalize punctuation
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("。", ".")
    return s


def _generate_unknown_id(raw_name: str) -> str:
    """Generate a stable UNKNOWN seat ID from the raw name.

    The ID is a string hash, never a numeric continuous feature.
    """
    h = hashlib.sha256(raw_name.encode("utf-8")).hexdigest()[:16]
    return f"UNKNOWN-{h}"
