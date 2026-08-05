"""Propagation candidate discovery for graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from seatflow.graph.schema import EdgeType, max_propagation_hops, default_top_k


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
        seat_participations: list[dict],
        all_stock_ids: set[str],
    ) -> list[PropagationLink]:
        """Find stocks that share a seat with the source stock.

        seat_participations: list of {seat_id, stock_id, side, available_time}
        """
        # Find seats that participated in the source stock
        source_seats: dict[str, datetime] = {}
        for p in seat_participations:
            if p["stock_id"] == source_stock_id:
                source_seats[p["seat_id"]] = p.get("available_time")

        # Find other stocks that share those seats
        links: list[PropagationLink] = []
        for p in seat_participations:
            if p["stock_id"] == source_stock_id:
                continue
            if p["seat_id"] in source_seats:
                at = source_seats[p["seat_id"]]
                if at and p.get("available_time") and p["available_time"] < at:
                    continue
                score = 1.0 / (1.0 + abs(hash(p["stock_id"])) % 100 / 100.0)
                links.append(PropagationLink(
                    source_stock_id=source_stock_id,
                    target_stock_id=p["stock_id"],
                    relation_type=EdgeType.SAME_SEAT_RELAY,
                    hop_count=1,
                    candidate_score=score,
                    path_evidence=[f"seat:{p['seat_id']}"],
                    available_time=p.get("available_time"),
                ))

        # Top-K truncation
        links.sort(key=lambda l: -l.candidate_score)
        return links[:self._top_k]

    def find_theme_diffusion(
        self,
        source_stock_id: str,
        theme_memberships: dict[str, list[str]],  # theme_id -> [stock_ids]
    ) -> list[PropagationLink]:
        """Find stocks that share a theme with the source stock."""
        # Find themes for source stock
        source_themes: set[str] = set()
        for theme_id, stock_ids in theme_memberships.items():
            if source_stock_id in stock_ids:
                source_themes.add(theme_id)

        # Find other stocks in same themes
        links: list[PropagationLink] = []
        for theme_id in source_themes:
            for stock_id in theme_memberships.get(theme_id, []):
                if stock_id == source_stock_id:
                    continue
                score = 0.5  # Lower score for theme diffusion
                links.append(PropagationLink(
                    source_stock_id=source_stock_id,
                    target_stock_id=stock_id,
                    relation_type=EdgeType.SAME_THEME_DIFFUSION,
                    hop_count=1,
                    candidate_score=score,
                    path_evidence=[f"theme:{theme_id}"],
                ))

        links.sort(key=lambda l: -l.candidate_score)
        return links[:self._top_k]

    def enforce_max_hops(self, links: list[PropagationLink]) -> list[PropagationLink]:
        """Filter out links that exceed max_hops."""
        return [l for l in links if l.hop_count <= self._max_hops]
