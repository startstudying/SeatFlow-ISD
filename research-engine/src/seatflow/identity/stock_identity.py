"""Point-in-Time stock identity resolution.

A stock may change its code, name, or exchange over time. This module
resolves a raw code to the canonical identity valid at a given point in
time, ensuring that historical records are not overwritten by current
aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class StockIdentity:
    """Canonical stock identity."""

    stock_id: str
    ts_code: str
    exchange: str
    board: str
    listing_date: Optional[datetime] = None
    delisting_date: Optional[datetime] = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    identity_confidence: float = 1.0
    source_evidence: str = ""


class StockIdentityRegistry:
    """Point-in-Time registry of stock identities.

    Maintains a list of stock identities with validity intervals.
    ``resolve_stock(raw_code, at_time)`` returns the identity valid at
    ``at_time``, not the current alias.
    """

    def __init__(self) -> None:
        self._identities: list[StockIdentity] = []

    def register(self, identity: StockIdentity) -> None:
        """Register a stock identity with a validity interval."""
        self._validate_no_overlap(identity)
        self._identities.append(identity)

    def _validate_no_overlap(self, new: StockIdentity) -> None:
        """Reject overlapping validity intervals for the same alias
        that resolve to different canonical stocks."""
        for existing in self._identities:
            if existing.stock_id == new.stock_id:
                continue
            # Check if aliases overlap
            shared = set(existing.aliases) & set(new.aliases)
            if not shared:
                continue
            # Check if validity intervals overlap
            if self._intervals_overlap(existing, new):
                raise ValueError(
                    f"Overlapping validity intervals for aliases {shared} "
                    f"resolving to different stocks: {existing.stock_id} vs {new.stock_id}"
                )

    @staticmethod
    def _intervals_overlap(a: StockIdentity, b: StockIdentity) -> bool:
        a_start = a.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        a_end = a.valid_to or datetime.max.replace(tzinfo=timezone.utc)
        b_start = b.valid_from or datetime.min.replace(tzinfo=timezone.utc)
        b_end = b.valid_to or datetime.max.replace(tzinfo=timezone.utc)
        return a_start < b_end and b_start < a_end

    def resolve_stock(self, raw_code: str, at_time: datetime) -> StockIdentity:
        """Resolve a raw stock code to the canonical identity at ``at_time``."""
        matches: list[StockIdentity] = []
        for identity in self._identities:
            # Check if raw_code matches stock_id, ts_code, or any alias
            if raw_code not in (identity.stock_id, identity.ts_code, *identity.aliases):
                continue
            # Check validity interval
            if identity.valid_from and at_time < identity.valid_from:
                continue
            if identity.valid_to and at_time > identity.valid_to:
                continue
            matches.append(identity)

        if not matches:
            raise KeyError(f"No stock identity found for {raw_code!r} at {at_time}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple stock identities found for {raw_code!r} at {at_time}"
            )
        return matches[0]
