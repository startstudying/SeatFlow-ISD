"""Propagation candidate discovery (Task 9).

This module discovers candidate propagation relations that feed the
Spillover pool. A propagation candidate is a typed relation between a
source stock (on the LHB) and a target stock (not necessarily on the
LHB) with an attached ``candidate_score`` in ``[0, 1]``.

Supported relation types
------------------------
- ``same_seat_relay`` -- two events share at least one seat.
- ``same_coalition_relay`` -- seats co-occur on the same side across events.
- ``same_theme_diffusion`` -- derived from theme/concept membership (when
  a theme-membership lookup is supplied).
- ``same_industry_diffusion`` -- derived from industry membership.
- ``capital_rotation`` -- a seat's net direction flips between events.
- ``repeat_event_path`` -- a stock appears in multiple cycles whose
  intervening gap is short.

Only records with ``available_time <= cutoff_time`` are used (PIT rule).
``candidate_score`` is bounded in ``[0, 1]`` and the default max hops is
2 with a per-relation Top-K truncation (default 20).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from seatflow.dataset.event import LHBEvent, SeatParticipation, SeatSide
from seatflow.domain.time import assert_aware


# --- Relation type constants -----------------------------------------


RELATION_SAME_SEAT_RELAY = "same_seat_relay"
RELATION_SAME_COALITION_RELAY = "same_coalition_relay"
RELATION_SAME_THEME_DIFFUSION = "same_theme_diffusion"
RELATION_SAME_INDUSTRY_DIFFUSION = "same_industry_diffusion"
RELATION_CAPITAL_ROTATION = "capital_rotation"
RELATION_REPEAT_EVENT_PATH = "repeat_event_path"

ALL_RELATION_TYPES = (
    RELATION_SAME_SEAT_RELAY,
    RELATION_SAME_COALITION_RELAY,
    RELATION_SAME_THEME_DIFFUSION,
    RELATION_SAME_INDUSTRY_DIFFUSION,
    RELATION_CAPITAL_ROTATION,
    RELATION_REPEAT_EVENT_PATH,
)


# --- Defaults ---------------------------------------------------------


DEFAULT_MAX_HOPS = 2
DEFAULT_TOP_K = 20


# --- Dataclass --------------------------------------------------------


@dataclass(frozen=True)
class PropagationCandidate:
    """A candidate propagation relation between two stocks.

    Attributes
    ----------
    source_event_id:
        The event that triggered the propagation search.
    source_stock_id:
        The stock on the LHB that originates the propagation.
    target_stock_id:
        The candidate stock receiving the propagation.
    relation_type:
        One of :data:`ALL_RELATION_TYPES`.
    candidate_score:
        Confidence/strength of the candidate, bounded in ``[0, 1]``.
    available_time:
        When the candidate first became observable (PIT).
    candidate_evidence:
        Human-readable evidence string (which seat / theme / etc.).
    """

    source_event_id: str
    source_stock_id: str
    target_stock_id: str
    relation_type: str
    candidate_score: float
    available_time: datetime
    candidate_evidence: str

    def __post_init__(self) -> None:
        assert_aware(self.available_time, "available_time")
        if not math.isfinite(self.candidate_score):
            raise ValueError(
                f"candidate_score must be finite, got {self.candidate_score}"
            )
        if not (0.0 <= self.candidate_score <= 1.0):
            raise ValueError(
                f"candidate_score must be in [0, 1], got {self.candidate_score}"
            )


# --- Membership lookup type ------------------------------------------


MembershipLookup = Callable[[str, datetime], list[str]]
"""A lookup that returns the theme/industry ids a stock belongs to at a
given point in time. Used for ``same_theme_diffusion`` and
``same_industry_diffusion`` candidates."""


# --- find_propagation_candidates -------------------------------------


def find_propagation_candidates(
    source_events: list[LHBEvent],
    seats: list[SeatParticipation],
    memberships: Optional[MembershipLookup] = None,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_k: int = DEFAULT_TOP_K,
) -> list[PropagationCandidate]:
    """Discover propagation candidates from LHB events.

    The discovery is a breadth-first walk over the seat-sharing graph:

    - Hop 1: targets that share a seat with a source event's stock.
    - Hop 2: targets that share a seat with a hop-1 target.
    - ... up to ``max_hops``.

    Candidates are de-duplicated by ``(source_stock_id, target_stock_id,
    relation_type)`` keeping the highest score. Each relation type is
    truncated to ``top_k`` candidates ordered by score descending, with
    ``target_stock_id`` as a stable tie-breaker.

    Self-loops (``source_stock_id == target_stock_id``) are never emitted.

    ``memberships`` is an optional lookup for theme/industry diffusion
    candidates; when ``None`` those relation types are skipped.
    """
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # Index seats by event_id and by seat_id.
    seats_by_event: dict[str, list[SeatParticipation]] = defaultdict(list)
    seats_by_seat_id: dict[str, list[SeatParticipation]] = defaultdict(list)
    stock_by_event: dict[str, str] = {}
    for part in seats:
        seats_by_event[part.event_id].append(part)
        seats_by_seat_id[part.seat_id].append(part)
    for ev in source_events:
        stock_by_event[ev.event_id] = ev.stock_id

    raw_candidates: list[PropagationCandidate] = []

    # --- same_seat_relay via BFS over seat-sharing ---------------------
    raw_candidates.extend(
        _same_seat_relay_bfs(source_events, seats_by_seat_id, max_hops)
    )

    # --- same_coalition_relay (co-buying seats) ------------------------
    raw_candidates.extend(_same_coalition_relay(source_events, seats_by_event))

    # --- capital_rotation (direction flip) -----------------------------
    raw_candidates.extend(_capital_rotation(source_events, seats_by_seat_id))

    # --- repeat_event_path (same stock across nearby events) -----------
    raw_candidates.extend(_repeat_event_path(source_events))

    # --- theme / industry diffusion ------------------------------------
    if memberships is not None:
        raw_candidates.extend(
            _membership_diffusion(
                source_events,
                memberships,
                relation=RELATION_SAME_THEME_DIFFUSION,
                member_kind="theme",
            )
        )
        raw_candidates.extend(
            _membership_diffusion(
                source_events,
                memberships,
                relation=RELATION_SAME_INDUSTRY_DIFFUSION,
                member_kind="industry",
            )
        )

    return _dedupe_and_truncate(raw_candidates, top_k)


# --- same_seat_relay BFS ---------------------------------------------


def _same_seat_relay_bfs(
    source_events: list[LHBEvent],
    seats_by_seat_id: dict[str, list[SeatParticipation]],
    max_hops: int,
) -> list[PropagationCandidate]:
    """BFS over the seat-sharing graph.

    From each source event's stock, walk up to ``max_hops`` hops through
    seats shared with other events. Hop depth attenuates the score:
    ``score = 1.0 / hop``.

    A candidate's ``available_time`` is the latest participation
    ``available_time`` along its propagation chain — the candidate only
    becomes observable once every hop's seat participation is known.
    """
    candidates: list[PropagationCandidate] = []
    best_hop: dict[tuple[str, str], int] = {}

    for ev in source_events:
        source_stock = ev.stock_id
        # Seed: seats that participated in this source event.
        seed_seat_ids: set[str] = set()
        for seat_id, parts in seats_by_seat_id.items():
            for p in parts:
                if p.event_id == ev.event_id:
                    seed_seat_ids.add(seat_id)
                    break

        visited: set[str] = {source_stock}
        # frontier entries: (target_stock, hop, chain_avail_time, evidence)
        frontier: list[tuple[str, int, datetime, str]] = []
        for seat_id in seed_seat_ids:
            for p in seats_by_seat_id[seat_id]:
                if p.event_id == ev.event_id:
                    continue
                tgt = _stock_of_event(p.event_id, source_events)
                if tgt is None or tgt == source_stock:
                    continue
                key = (source_stock, tgt)
                chain_avail = p.available_time
                if key not in best_hop or 1 < best_hop[key]:
                    best_hop[key] = 1
                    frontier.append((tgt, 1, chain_avail, f"seat {seat_id} relay (hop 1)"))
                    candidates.append(
                        PropagationCandidate(
                            source_event_id=ev.event_id,
                            source_stock_id=source_stock,
                            target_stock_id=tgt,
                            relation_type=RELATION_SAME_SEAT_RELAY,
                            candidate_score=_hop_score(1),
                            available_time=chain_avail,
                            candidate_evidence=f"seat {seat_id} relay (hop 1)",
                        )
                    )
                    visited.add(tgt)

        hop = 1
        while hop < max_hops and frontier:
            next_frontier: list[tuple[str, int, datetime, str]] = []
            for current_stock, _, parent_avail, _ in frontier:
                # Seats touching the current stock's events.
                seats_touching: set[str] = set()
                for seat_id, parts in seats_by_seat_id.items():
                    for p in parts:
                        if p.event_id == ev.event_id:
                            continue
                        tgt = _stock_of_event(p.event_id, source_events)
                        if tgt == current_stock:
                            seats_touching.add(seat_id)
                            break
                for seat_id in seats_touching:
                    for p in seats_by_seat_id[seat_id]:
                        if p.event_id == ev.event_id:
                            continue
                        tgt = _stock_of_event(p.event_id, source_events)
                        if tgt is None or tgt in visited or tgt == source_stock:
                            continue
                        key = (source_stock, tgt)
                        new_hop = hop + 1
                        chain_avail = max(parent_avail, p.available_time)
                        if key not in best_hop or new_hop < best_hop[key]:
                            best_hop[key] = new_hop
                            candidates.append(
                                PropagationCandidate(
                                    source_event_id=ev.event_id,
                                    source_stock_id=source_stock,
                                    target_stock_id=tgt,
                                    relation_type=RELATION_SAME_SEAT_RELAY,
                                    candidate_score=_hop_score(new_hop),
                                    available_time=chain_avail,
                                    candidate_evidence=f"seat {seat_id} relay (hop {new_hop})",
                                )
                            )
                            visited.add(tgt)
                            next_frontier.append((tgt, new_hop, chain_avail, f"seat {seat_id} relay (hop {new_hop})"))
            frontier = next_frontier
            hop += 1

    return candidates


def _stock_of_event(event_id: str, events: list[LHBEvent]) -> Optional[str]:
    for ev in events:
        if ev.event_id == event_id:
            return ev.stock_id
    return None


def _hop_score(hop: int) -> float:
    """Score attenuates with hop depth: 1/hop, bounded in (0, 1]."""
    return 1.0 / hop


# --- same_coalition_relay --------------------------------------------


def _same_coalition_relay(
    source_events: list[LHBEvent],
    seats_by_event: dict[str, list[SeatParticipation]],
) -> list[PropagationCandidate]:
    """Two events whose buy-side seat coalitions overlap create a
    ``same_coalition_relay`` candidate.

    The score is the Jaccard similarity of the buy-side seat sets, bounded
    in [0, 1]. The candidate's ``available_time`` is the later of the two
    events' coalition availability.
    """
    candidates: list[PropagationCandidate] = []
    for i, ev_i in enumerate(source_events):
        parts_i = seats_by_event.get(ev_i.event_id, [])
        seats_i = {p.seat_id for p in parts_i if p.side == SeatSide.BUY}
        avail_i = max((p.available_time for p in parts_i), default=ev_i.available_time)
        if not seats_i:
            continue
        for j, ev_j in enumerate(source_events):
            if j <= i:
                continue
            if ev_j.stock_id == ev_i.stock_id:
                continue
            parts_j = seats_by_event.get(ev_j.event_id, [])
            seats_j = {p.seat_id for p in parts_j if p.side == SeatSide.BUY}
            avail_j = max((p.available_time for p in parts_j), default=ev_j.available_time)
            if not seats_j:
                continue
            union = seats_i | seats_j
            inter = seats_i & seats_j
            if not inter:
                continue
            score = len(inter) / len(union)
            candidates.append(
                PropagationCandidate(
                    source_event_id=ev_i.event_id,
                    source_stock_id=ev_i.stock_id,
                    target_stock_id=ev_j.stock_id,
                    relation_type=RELATION_SAME_COALITION_RELAY,
                    candidate_score=score,
                    available_time=max(avail_i, avail_j),
                    candidate_evidence=f"coalition jaccard={score:.2f} (seats {sorted(inter)})",
                )
            )
    return candidates


# --- capital_rotation -------------------------------------------------


def _capital_rotation(
    source_events: list[LHBEvent],
    seats_by_seat_id: dict[str, list[SeatParticipation]],
) -> list[PropagationCandidate]:
    """A seat that buys in one event and sells in another rotates
    capital between the two stocks.

    Score is a constant mid-band 0.4 (rotation is suggestive, not
    conclusive). The candidate's ``available_time`` is the latest
    participation time on that seat (the rotation only becomes observable
    once both legs are known).
    """
    candidates: list[PropagationCandidate] = []
    for seat_id, parts in seats_by_seat_id.items():
        buy_stocks: set[str] = set()
        sell_stocks: set[str] = set()
        avail_times: list[datetime] = []
        for p in parts:
            stock = _stock_of_event(p.event_id, source_events)
            if stock is None:
                continue
            avail_times.append(p.available_time)
            if p.side == SeatSide.BUY:
                buy_stocks.add(stock)
            else:
                sell_stocks.add(stock)
        if not buy_stocks or not sell_stocks:
            continue
        chain_avail = max(avail_times) if avail_times else datetime.fromisoformat("1970-01-01T00:00:00+00:00")
        for source in sorted(buy_stocks):
            for target in sorted(sell_stocks - {source}):
                candidates.append(
                    PropagationCandidate(
                        source_event_id=_first_event_for(source, source_events),
                        source_stock_id=source,
                        target_stock_id=target,
                        relation_type=RELATION_CAPITAL_ROTATION,
                        candidate_score=0.4,
                        available_time=chain_avail,
                        candidate_evidence=f"seat {seat_id} rotation",
                    )
                )
    return candidates


def _first_event_for(stock_id: str, events: list[LHBEvent]) -> str:
    for ev in events:
        if ev.stock_id == stock_id:
            return ev.event_id
    return ""


# --- repeat_event_path ------------------------------------------------


def _repeat_event_path(
    source_events: list[LHBEvent],
    max_gap_days: int = 10,
) -> list[PropagationCandidate]:
    """A stock that appears on the LHB in two events whose trade-date gap
    is short emits a ``repeat_event_path`` candidate to the other stocks
    on its companion event."""
    by_stock: dict[str, list[LHBEvent]] = defaultdict(list)
    for ev in source_events:
        by_stock[ev.stock_id].append(ev)
    candidates: list[PropagationCandidate] = []
    for stock_id, evs in by_stock.items():
        evs = sorted(evs, key=lambda e: e.trade_date)
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                gap = (evs[j].trade_date - evs[i].trade_date).days
                if gap > max_gap_days:
                    break
                # i -> j propagation: a later event of the same stock
                # re-asserts the propagation path.
                candidates.append(
                    PropagationCandidate(
                        source_event_id=evs[i].event_id,
                        source_stock_id=stock_id,
                        target_stock_id=stock_id,
                        relation_type=RELATION_REPEAT_EVENT_PATH,
                        candidate_score=1.0 - gap / max_gap_days,
                        available_time=evs[i].available_time,
                        candidate_evidence=f"repeat path gap={gap}d",
                    )
                )
    return candidates


# --- membership diffusion --------------------------------------------


def _membership_diffusion(
    source_events: list[LHBEvent],
    memberships: MembershipLookup,
    *,
    relation: str,
    member_kind: str,
) -> list[PropagationCandidate]:
    """For each source event, find stocks sharing a theme/industry at the
    event's available_time and emit a diffusion candidate.
    """
    candidates: list[PropagationCandidate] = []
    for ev in source_events:
        members = memberships(member_kind, ev.available_time)
        my_id = members.get(ev.stock_id) if isinstance(members, dict) else None
        if my_id is None:
            continue
        # All stocks sharing ``my_id``.
        for stock_id, m_id in members.items():
            if stock_id == ev.stock_id:
                continue
            if m_id != my_id:
                continue
            candidates.append(
                PropagationCandidate(
                    source_event_id=ev.event_id,
                    source_stock_id=ev.stock_id,
                    target_stock_id=stock_id,
                    relation_type=relation,
                    candidate_score=0.3,
                    available_time=ev.available_time,
                    candidate_evidence=f"{member_kind} {my_id}",
                )
            )
    return candidates


# --- Dedup and truncate ----------------------------------------------


def _dedupe_and_truncate(
    candidates: list[PropagationCandidate],
    top_k: int,
) -> list[PropagationCandidate]:
    """Per (relation_type, source_stock_id) keep the top_k candidates.

    Ordering: ``candidate_score`` desc, then ``target_stock_id`` asc
    (stable tie-breaker), then ``candidate_evidence`` asc.
    Self-loops are dropped.
    """
    # Filter self-loops and keep the maximum score per (relation, src, tgt).
    best: dict[tuple[str, str, str], PropagationCandidate] = {}
    for c in candidates:
        if c.source_stock_id == c.target_stock_id:
            continue
        key = (c.relation_type, c.source_stock_id, c.target_stock_id)
        if key not in best or c.candidate_score > best[key].candidate_score:
            best[key] = c

    by_group: dict[tuple[str, str], list[PropagationCandidate]] = defaultdict(list)
    for c in best.values():
        by_group[(c.relation_type, c.source_stock_id)].append(c)

    final: list[PropagationCandidate] = []
    for (relation, src), group in by_group.items():
        group.sort(
            key=lambda c: (
                -c.candidate_score,
                c.target_stock_id,
                c.candidate_evidence,
            )
        )
        final.extend(group[:top_k])
    return final
