"""Propagation candidate discovery for graph construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from typing import Optional

from seatflow.graph.schema import EdgeType, default_top_k, max_propagation_hops


@dataclass(frozen=True)
class PropagationLink:
    """A discovered propagation link between two stocks."""

    source_stock_id: str
    target_stock_id: str
    relation_type: EdgeType
    hop_count: int
    candidate_score: float
    path_evidence: list[str] = field(default_factory=list)
    available_time: Optional[datetime] = None


class PropagationCandidateFinder:
    """Finds propagation candidates between stocks.

    Enforces:
    - Maximum 2 hops
    - Top-K truncation per relation type per source
    - Only stocks in the Direct pool propagate to Spillover pool
    """

    def __init__(self, *, top_k: int | None = None, max_hops: int | None = None) -> None:
        self._top_k = top_k or default_top_k()
        self._max_hops = max_hops or max_propagation_hops()

    def find_seat_relay(
        self,
        source_stock_id: str,
        seat_participations: list[dict[str, object]],
        all_stock_ids: set[str],
        *,
        cutoff_time: datetime,
    ) -> list[PropagationLink]:
        """Rank shared-seat stocks using only PIT coverage and BUY/SELL agreement."""
        if source_stock_id not in all_stock_ids:
            return []

        source_seats: dict[str, tuple[str, datetime]] = {}
        for p in seat_participations:
            evidence = _seat_evidence(p, cutoff_time=cutoff_time)
            if evidence is None:
                continue
            seat_id, stock_id, side, available_time = evidence
            if stock_id == source_stock_id:
                source_previous = source_seats.get(seat_id)
                if source_previous is None or (available_time, side) < (
                    source_previous[1],
                    source_previous[0],
                ):
                    source_seats[seat_id] = (side, available_time)

        if not source_seats:
            return []

        target_evidence: dict[str, dict[str, tuple[float, datetime]]] = {}
        for p in seat_participations:
            evidence = _seat_evidence(p, cutoff_time=cutoff_time)
            if evidence is None:
                continue
            seat_id, stock_id, side, available_time = evidence
            if stock_id == source_stock_id or stock_id not in all_stock_ids:
                continue

            source_evidence = source_seats.get(seat_id)
            if source_evidence is None or source_evidence[1] > available_time:
                continue

            contribution = 1.0 if source_evidence[0] == side else 0.5
            per_seat = target_evidence.setdefault(stock_id, {})
            target_previous = per_seat.get(seat_id)
            if target_previous is None or contribution > target_previous[0]:
                per_seat[seat_id] = (contribution, available_time)

        links = [
            PropagationLink(
                source_stock_id=source_stock_id,
                target_stock_id=target_id,
                relation_type=EdgeType.SAME_SEAT_RELAY,
                hop_count=1,
                candidate_score=sum(score for score, _ in per_seat.values()) / len(source_seats),
                path_evidence=[f"seat:{seat_id}" for seat_id in sorted(per_seat)],
                available_time=max(available_time for _, available_time in per_seat.values()),
            )
            for target_id, per_seat in target_evidence.items()
            if per_seat
        ]

        links.sort(key=lambda link: (-link.candidate_score, link.target_stock_id))
        return links[: self._top_k]

    def find_theme_diffusion(
        self,
        source_stock_id: str,
        theme_memberships: dict[str, list[dict[str, object]]],
        all_stock_ids: set[str],
        *,
        cutoff_time: datetime,
    ) -> list[PropagationLink]:
        """Rank shared-theme stocks by their strongest PIT membership evidence."""
        if source_stock_id not in all_stock_ids:
            return []

        source_themes: dict[str, float] = {}
        for theme_id, memberships in theme_memberships.items():
            for membership in memberships:
                evidence = _theme_evidence(membership, cutoff_time=cutoff_time)
                if evidence is None:
                    continue
                stock_id, strength, _ = evidence
                if stock_id == source_stock_id:
                    source_themes[theme_id] = max(
                        strength,
                        source_themes.get(theme_id, 0.0),
                    )

        if not source_themes:
            return []

        target_evidence: dict[str, tuple[float, str, datetime]] = {}
        for theme_id in sorted(source_themes):
            for membership in theme_memberships.get(theme_id, []):
                evidence = _theme_evidence(membership, cutoff_time=cutoff_time)
                if evidence is None:
                    continue
                stock_id, strength, available_time = evidence
                if stock_id == source_stock_id or stock_id not in all_stock_ids:
                    continue

                contribution = min(source_themes[theme_id], strength)
                previous = target_evidence.get(stock_id)
                if previous is None or contribution > previous[0]:
                    target_evidence[stock_id] = (
                        contribution,
                        theme_id,
                        available_time,
                    )

        links = [
            PropagationLink(
                source_stock_id=source_stock_id,
                target_stock_id=target_id,
                relation_type=EdgeType.SAME_THEME_DIFFUSION,
                hop_count=1,
                candidate_score=evidence[0],
                path_evidence=[f"theme:{evidence[1]}"],
                available_time=evidence[2],
            )
            for target_id, evidence in target_evidence.items()
        ]

        links.sort(key=lambda link: (-link.candidate_score, link.target_stock_id))
        return links[: self._top_k]

    def enforce_max_hops(self, links: list[PropagationLink]) -> list[PropagationLink]:
        """Filter out links that exceed max_hops."""
        return [link for link in links if link.hop_count <= self._max_hops]


def _seat_evidence(
    participation: dict[str, object],
    *,
    cutoff_time: datetime,
) -> Optional[tuple[str, str, str, datetime]]:
    seat_id = participation.get("seat_id")
    stock_id = participation.get("stock_id")
    side = participation.get("side")
    available_time = participation.get("available_time")

    if not isinstance(seat_id, str) or not seat_id:
        return None
    if not isinstance(stock_id, str) or not stock_id:
        return None
    if not isinstance(side, str) or side.upper() not in {"BUY", "SELL"}:
        return None
    if not isinstance(available_time, datetime):
        return None
    if not _is_available_by(available_time, cutoff_time=cutoff_time):
        return None

    return seat_id, stock_id, side.upper(), available_time


def _theme_evidence(
    membership: dict[str, object],
    *,
    cutoff_time: datetime,
) -> Optional[tuple[str, float, datetime]]:
    stock_id = membership.get("stock_id")
    available_time = membership.get("available_time")
    raw_strength = membership.get("membership_strength")

    if not isinstance(stock_id, str) or not stock_id:
        return None
    if not isinstance(available_time, datetime):
        return None
    if not _is_available_by(available_time, cutoff_time=cutoff_time):
        return None
    if isinstance(raw_strength, bool) or not isinstance(raw_strength, Real):
        return None

    strength = float(raw_strength)
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        return None

    return stock_id, strength, available_time


def _is_available_by(value: datetime, *, cutoff_time: datetime) -> bool:
    try:
        return value <= cutoff_time
    except TypeError:
        return False
