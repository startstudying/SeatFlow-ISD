"""Tests for graph tensorization (Task 14)."""

from datetime import datetime, timezone

import pytest

from seatflow.graph.schema import EdgeType, NodeType
from seatflow.graph.snapshot import GraphSnapshotBuilder
from seatflow.graph.tensorize import tensorize_snapshot, TensorizedGraph


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


class TestTensorize:
    def _make_snapshot(self):
        builder = GraphSnapshotBuilder()
        builder.add_stock_node("stock-1")
        builder.add_stock_node("stock-2")
        builder.add_seat_node("seat-1")
        builder.add_event_node("event-1")
        builder.add_edge(EdgeType.SEAT_BUY_EVENT, "seat-1", "event-1", {"buy_amount": 100.0})
        builder.add_edge(EdgeType.EVENT_IMPACTS_STOCK, "event-1", "stock-1")
        builder.add_edge(EdgeType.SAME_SEAT_RELAY, "stock-1", "stock-2", {"candidate_score": 0.8})
        return builder.build(snapshot_id="snap-t", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))

    def test_node_id_map_built(self):
        snap = self._make_snapshot()
        tg = tensorize_snapshot(snap)
        assert "Stock" in tg.node_id_maps
        assert "Seat" in tg.node_id_maps
        assert "Event" in tg.node_id_maps
        assert tg.node_id_maps["Stock"]["stock-1"] == 0
        assert tg.node_id_maps["Stock"]["stock-2"] == 1

    def test_edge_indices_built(self):
        snap = self._make_snapshot()
        tg = tensorize_snapshot(snap)
        assert "seat_buy_event" in tg.edge_indices
        assert "event_impacts_stock" in tg.edge_indices
        assert "same_seat_relay" in tg.edge_indices
        # Check seat_buy_event: seat-1 (idx 0) -> event-1 (idx 0)
        assert (0, 0) in tg.edge_indices["seat_buy_event"]

    def test_snapshot_hash_preserved(self):
        snap = self._make_snapshot()
        tg = tensorize_snapshot(snap)
        assert tg.snapshot_hash == snap.snapshot_hash

    def test_node_features_dim(self):
        snap = self._make_snapshot()
        tg = tensorize_snapshot(snap, feature_dim=16)
        for nt, feats in tg.node_features.items():
            for row in feats:
                assert len(row) == 16

    def test_edge_features_built(self):
        snap = self._make_snapshot()
        tg = tensorize_snapshot(snap)
        assert "seat_buy_event" in tg.edge_features
        # buy_amount=100.0 should be in the first edge feature
        assert 100.0 in tg.edge_features["seat_buy_event"][0]

    def test_dangling_edge_skipped(self):
        """Edges referencing non-existent nodes are skipped."""
        builder = GraphSnapshotBuilder()
        builder.add_stock_node("stock-1")
        builder.add_edge(EdgeType.SAME_SEAT_RELAY, "stock-1", "nonexistent")
        snap = builder.build(snapshot_id="snap-d", cutoff_time=_utc("2025-01-01T00:00:00+00:00"))
        tg = tensorize_snapshot(snap)
        assert len(tg.edge_indices.get("same_seat_relay", [])) == 0
