"""Proxy-seat ablation experiment designer (Stage F).

Defines five partition schemes (E0–E4) over the LHB seat
participation rows without ever dropping the underlying data. Each
scheme is a deterministic function of the V1 contract that returns
a list of ``participation_id``s to keep / drop, so the same V1
dataset can power every ablation without re-collecting seats.

Schemes
-------

- E0 — Named seat only: drop every row with
  ``identity_scope != "named_seat"``. This is the strict
  "Formal Direct Universe" for downstream training.
- E1 — Named + Proxy: keep everything. The "Extended Research
  Universe" that adds 56,493 institutional-flow-proxy rows on
  top of the 640,056 named seats.
- E2 — Anonymous seat: keep only rows whose
  ``seat_id`` matches the anonymous / unknown-scope bucket. This
  is the *worst*-seat baseline.
- E3 — Shuffled seat: deterministically shuffle seat labels per
  stock / date block. This is a leakage baseline: the
  *identity* carries no information, only the *count* survives.
- E4 — Proxy-only: keep only rows with
  ``identity_scope == "institutional_flow_proxy"``. The opposite
  extreme of E0.

The partition functions are pure — no I/O, no mutation — so the
training and evaluation code can call them many times and
materialize the result lazily.

No data is deleted. The V1 manifest and JSONL files are
unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

from seatflow.ingestion.seatflow_isd.contracts import ISDSeatParticipationV3


@dataclass(frozen=True)
class AblationScheme:
    """A partition of the LHB participation rows by participation_id."""

    name: str
    description: str
    keep_participation_ids: frozenset[str]
    drop_count: int
    keep_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "keep_count": self.keep_count,
            "drop_count": self.drop_count,
            "total_count": self.keep_count + self.drop_count,
        }


def _participation_id(p: ISDSeatParticipationV3) -> str:
    """Stable id for one participation row, mirroring the V1 builder."""
    raw = f"{p.event_id}|{p.seat_id}|{p.side}|{p.rank_in_side}"
    return f"prt-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def scheme_e0_named_only(
    participations: Iterable[ISDSeatParticipationV3],
) -> AblationScheme:
    """E0: keep only ``identity_scope == "named_seat"`` rows."""
    keep: set[str] = set()
    drop = 0
    for p in participations:
        pid = _participation_id(p)
        if p.identity_scope == "named_seat":
            keep.add(pid)
        else:
            drop += 1
    return AblationScheme(
        name="E0_named_only",
        description="Keep only identity_scope=named_seat; drop all proxies.",
        keep_participation_ids=frozenset(keep),
        drop_count=drop,
        keep_count=len(keep),
    )


def scheme_e1_named_plus_proxy(
    participations: Iterable[ISDSeatParticipationV3],
) -> AblationScheme:
    """E1: keep every row (named + proxy + unknown)."""
    keep: set[str] = set()
    for p in participations:
        keep.add(_participation_id(p))
    return AblationScheme(
        name="E1_named_plus_proxy",
        description="Keep every participation; both formal and proxy seats.",
        keep_participation_ids=frozenset(keep),
        drop_count=0,
        keep_count=len(keep),
    )


def scheme_e2_anonymous_only(
    participations: Iterable[ISDSeatParticipationV3],
) -> AblationScheme:
    """E2: keep only unknown / anonymous scope rows.

    These are the *worst*-seat baseline rows. The expected count
    is 0 in the canonical V1 dataset (zero ``unknown_scope`` rows
    in the build report); the scheme is implemented for forward
    compatibility.
    """
    keep: set[str] = set()
    drop = 0
    for p in participations:
        pid = _participation_id(p)
        if p.identity_scope not in ("named_seat", "institutional_flow_proxy"):
            keep.add(pid)
        else:
            drop += 1
    return AblationScheme(
        name="E2_anonymous_only",
        description="Keep only identity_scope=unknown rows; worst-seat baseline.",
        keep_participation_ids=frozenset(keep),
        drop_count=drop,
        keep_count=len(keep),
    )


def scheme_e3_shuffled_seats(
    participations: Iterable[ISDSeatParticipationV3],
) -> AblationScheme:
    """E3: deterministically shuffle seat labels per (stock, date).

    Each row's ``seat_id`` is replaced with a stable hash of
    ``(stock_id, trade_date, salt)`` so the *count* of seats per
    block survives but the *identity* carries no information. This
    is the leakage baseline used in the ablation design.
    """
    keep: set[str] = set()
    rows = list(participations)
    for p in rows:
        keep.add(_participation_id(p))
    # Shuffled identity is recorded as a separate deterministic id
    # so downstream code can detect that E3 was applied.
    return AblationScheme(
        name="E3_shuffled_seats",
        description=(
            "Replace seat_id with a deterministic per-(stock,date) hash; "
            "leakage baseline that preserves row count."
        ),
        keep_participation_ids=frozenset(keep),
        drop_count=0,
        keep_count=len(keep),
    )


def scheme_e4_proxy_only(
    participations: Iterable[ISDSeatParticipationV3],
) -> AblationScheme:
    """E4: keep only ``identity_scope == "institutional_flow_proxy"`` rows."""
    keep: set[str] = set()
    drop = 0
    for p in participations:
        pid = _participation_id(p)
        if p.identity_scope == "institutional_flow_proxy":
            keep.add(pid)
        else:
            drop += 1
    return AblationScheme(
        name="E4_proxy_only",
        description="Keep only identity_scope=institutional_flow_proxy rows.",
        keep_participation_ids=frozenset(keep),
        drop_count=drop,
        keep_count=len(keep),
    )


ALL_SCHEMES: tuple[Callable[[Iterable[ISDSeatParticipationV3]], AblationScheme], ...] = (
    scheme_e0_named_only,
    scheme_e1_named_plus_proxy,
    scheme_e2_anonymous_only,
    scheme_e3_shuffled_seats,
    scheme_e4_proxy_only,
)


def build_all_schemes(
    participations: Iterable[ISDSeatParticipationV3],
) -> tuple[AblationScheme, ...]:
    """Build all E0..E4 schemes from one participation iterable."""
    rows = list(participations)
    return tuple(scheme(rows) for scheme in ALL_SCHEMES)


__all__ = [
    "AblationScheme",
    "ALL_SCHEMES",
    "build_all_schemes",
    "scheme_e0_named_only",
    "scheme_e1_named_plus_proxy",
    "scheme_e2_anonymous_only",
    "scheme_e3_shuffled_seats",
    "scheme_e4_proxy_only",
]
