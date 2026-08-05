"""Tests for graph snapshot and state store (Task 14)."""

from datetime import datetime, timezone

import pytest

from seatflow.graph.schema import EdgeType, NodeType
from seatflow.graph.snapshot import GraphSnapshotBuilder, GraphSnapshot
from seatflow.graph.state_store import GraphStateStore
from seatflow.graph.candidates import PropagationCandidateFinder


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestGraphSnapshot:
    def test_basic_snapshot(self):
        builder = GraphSnapshotBuilder()
        builder.add_stock_node("stock-1")
        builder.add_seat_node("seat-1")
        builder.add_event_node("event-1")
        builder.add_edge(EdgeType.SEAT_BUY_EVENT, "seat-1", "event-1", {"buy_amount": 100.0})
        builder.add_edge(EdgeType.EVENT_IMPACTS_STOCK, "event-1", "stock-1")
        snap = builder.build(snapshot_id="snap-1", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        assert snap.node_count == 3
        assert len(snap.edges) == 2

    def test_snapshot_hash_deterministic(self):
        def make() -> GraphSnapshot:
            b = GraphSnapshotBuilder()
            b.add_stock_node("stock-1")
            b.add_event_node("event-1")
            b.add_edge(EdgeType.EVENT_IMPACTS_STOCK, "event-1", "stock-1")
            return b.build(snapshot_id="snap-x", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        s1 = make()
        s2 = make()
        assert s1.snapshot_hash == s2.snapshot_hash

    def test_top_k_truncation(self):
        builder = GraphSnapshotBuilder(top_k=2)
        builder.add_stock_node("stock-src")
        for i in range(10):
            builder.add_stock_node(f"stock-dst-{i}")
            builder.add_edge(
                EdgeType.SAME_SEAT_RELAY,
                "stock-src",
                f"stock-dst-{i}",
                {"candidate_score": float(i)},
            )
        builder.apply_top_k_truncation()
        snap = builder.build(snapshot_id="snap-2", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        # Should keep only top 2
        relay_edges = [e for e in snap.edges if e.edge_type == EdgeType.SAME_SEAT_RELAY]
        assert len(relay_edges) == 2
        # Highest scores kept
        scores = [e.properties.get("candidate_score") for e in relay_edges]
        assert max(scores) == 9.0

    def test_node_and_edge_counts(self):
        builder = GraphSnapshotBuilder()
        for i in range(3):
            builder.add_stock_node(f"s{i}")
        for i in range(2):
            builder.add_seat_node(f"seat{i}")
        builder.add_event_node("e1")
        builder.add_edge(EdgeType.EVENT_IMPACTS_STOCK, "e1", "s0")
        builder.add_edge(EdgeType.SEAT_BUY_EVENT, "seat0", "e1")
        snap = builder.build(snapshot_id="snap-3", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        assert snap.node_counts.get("Stock") == 3
        assert snap.node_counts.get("Seat") == 2
        assert snap.node_counts.get("Event") == 1
        assert snap.edge_counts.get("event_impacts_stock") == 1


class TestGraphStateStore:
    def test_save_and_retrieve(self):
        store = GraphStateStore()
        b = GraphSnapshotBuilder()
        b.add_stock_node("s1")
        snap = b.build(snapshot_id="snap-s1", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        store.save(snap)
        retrieved = store.get("snap-s1")
        assert retrieved.snapshot_id == "snap-s1"

    def test_duplicate_save_rejected(self):
        store = GraphStateStore()
        b = GraphSnapshotBuilder()
        b.add_stock_node("s1")
        snap = b.build(snapshot_id="snap-dup", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        store.save(snap)
        with pytest.raises(ValueError, match="already exists"):
            store.save(snap)

    def test_get_by_cutoff(self):
        store = GraphStateStore()
        b = GraphSnapshotBuilder()
        b.add_stock_node("s1")

        snap1 = b.build(snapshot_id="snap-1", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        store.save(snap1)
        snap2 = b.build(snapshot_id="snap-2", cutoff_time=_utc("2025-02-01T00:00:00+00:00"))
        store.save(snap2)

        latest = store.get_by_cutoff(_utc("2025-03-01T00:00:00+00:00"))
        assert latest.snapshot_id == "snap-2"

        early = store.get_by_cutoff(_utc("2025-01-15T00:00:00+00:00"))
        assert early.snapshot_id == "snap-1"


class TestPropagationCandidates:
    def test_seat_relay_finds_shared_seat(self):
        finder = PropagationCandidateFinder()
        participations = [
            {"seat_id": "seat-A", "stock_id": "stock-src", "side": "BUY", "available_time": _utc("2025-01-01T00:00:00+00:00")},
            {"seat_id": "seat-A", "stock_id": "stock-tgt-1", "side": "BUY", "available_time": _utc("2025-01-02T00:00:00+00:00")},
        ]
        links = finder.find_seat_relay("stock-src", participations, {"stock-src", "stock-tgt-1"})
        assert len(links) == 1
        assert links[0].target_stock_id == "stock-tgt-1"
        assert links[0].relation_type == EdgeType.SAME_SEAT_RELAY

    def test_top_k_truncation(self):
        finder = PropagationCandidateFinder(top_k=3)
        participations = []
        for i in range(10):
            participations.extend([
                {"seat_id": "seat-A", "stock_id": "stock-src", "side": "BUY", "available_time": _utc("2025-01-01T00:00:00+00:00")},
                {"seat_id": "seat-A", "stock_id": f"stock-{i}", "side": "BUY", "available_time": _utc("2025-01-02T00:00:00+00:00")},
            ])
        links = finder.find_seat_relay("stock-src", participations, {"stock-src", *[f"stock-{i}" for i in range(10)]})
        assert len(links) == 3

    def test_max_hops_enforcement(self):
        finder = PropagationCandidateFinder(max_hops=2)
        participations = [
            {"seat_id": "seat-A", "stock_id": "stock-src", "side": "BUY", "available_time": _utc("2025-01-01T00:00:00+00:00")},
            {"seat_id": "seat-A", "stock_id": "stock-tgt", "side": "BUY", "available_time": _utc("2025-01-02T00:00:00+00:00")},
        ]
        links = finder.find_seat_relay("stock-src", participations, {"stock-src", "stock-tgt"})
        filtered = finder.enforce_max_hops(links)
        assert all(l.hop_count <= 2 for l in filtered)

    def test_theme_diffusion(self):
        finder = PropagationCandidateFinder()
        memberships = {
            "theme-tech": ["stock-src", "stock-tech-1", "stock-tech-2"],
            "theme-energy": ["stock-other"],
        }
        links = finder.find_theme_diffusion("stock-src", memberships)
        target_ids = {l.target_stock_id for l in links}
        assert "stock-tech-1" in target_ids
        assert "stock-tech-2" in target_ids
        assert "stock-other" not in target_ids
