"""Tests for graph tensorization (Task 14)."""

from datetime import datetime

import pytest

from seatflow.graph.schema import EdgeType
from seatflow.graph.snapshot import GraphSnapshotBuilder
from seatflow.graph.tensorize import tensorize_snapshot


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


_CUTOFF = _utc("2025-01-01T00:00:00+00:00")


def _properties(**features: object) -> dict[str, object]:
    return {"continuous_features": features}


def _stock_builder(*feature_rows: dict[str, object] | None) -> GraphSnapshotBuilder:
    builder = GraphSnapshotBuilder()
    for index, features in enumerate(feature_rows, start=1):
        builder.add_stock_node(
            f"stock-{index}",
            {} if features is None else _properties(**features),
        )
    return builder


def _seat_buy_snapshot(edge_features: dict[str, object]):
    builder = GraphSnapshotBuilder()
    builder.add_seat_node("seat-1", _properties(seat_win_rate=0.6))
    builder.add_event_node("event-1", _properties(event_net_amount=100.0))
    builder.add_edge(
        EdgeType.SEAT_BUY_EVENT,
        "seat-1",
        "event-1",
        {"continuous_features": edge_features},
    )
    return builder.build(snapshot_id="snap-edge-schema", cutoff_time=_CUTOFF)


class TestTensorize:
    def _make_snapshot(
        self,
        *,
        stock_ids: tuple[str, str] = ("stock-1", "stock-2"),
        seat_id: str = "seat-1",
        event_id: str = "event-1",
    ):
        builder = GraphSnapshotBuilder()
        builder.add_stock_node(
            stock_ids[0],
            _properties(stock_volatility_5d=0.1),
        )
        builder.add_stock_node(
            stock_ids[1],
            _properties(stock_volatility_5d=0.2),
        )
        builder.add_seat_node(
            seat_id,
            _properties(seat_win_rate=0.6),
        )
        builder.add_event_node(
            event_id,
            _properties(event_net_amount=100.0),
        )
        builder.add_edge(
            EdgeType.SEAT_BUY_EVENT,
            seat_id,
            event_id,
            _properties(buy_amount=100.0),
        )
        builder.add_edge(
            EdgeType.EVENT_IMPACTS_STOCK,
            event_id,
            stock_ids[0],
            _properties(net_amount=100.0),
        )
        builder.add_edge(
            EdgeType.SAME_SEAT_RELAY,
            stock_ids[0],
            stock_ids[1],
            _properties(candidate_score=0.8),
        )
        return builder.build(snapshot_id="snap-t", cutoff_time=_CUTOFF)

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

    def test_explicit_semantic_features_are_tensorized(self):
        tg = tensorize_snapshot(self._make_snapshot())
        assert tg.node_features["Stock"] == [[0.1], [0.2]]
        assert tg.edge_features["seat_buy_event"] == [[100.0]]

    def test_renaming_identity_ids_preserves_all_continuous_features(self):
        """Reintroducing an ID-derived value would change one of these matrices."""
        original = tensorize_snapshot(self._make_snapshot())
        renamed = tensorize_snapshot(
            self._make_snapshot(
                stock_ids=("renamed-stock-a", "renamed-stock-b"),
                seat_id="renamed-seat",
                event_id="renamed-event",
            )
        )

        assert renamed.node_features == original.node_features
        assert renamed.edge_features == original.edge_features
        assert renamed.edge_indices == original.edge_indices

    @pytest.mark.parametrize("identity_name", ["stock_id", "node_id", "id_hash"])
    def test_identity_field_rejected_as_node_continuous_feature(self, identity_name: str):
        builder = _stock_builder({identity_name: 1.0})
        snapshot = builder.build(
            snapshot_id="snap-identity-node",
            cutoff_time=_CUTOFF,
        )

        with pytest.raises(ValueError, match="identity"):
            tensorize_snapshot(snapshot)

    @pytest.mark.parametrize(
        ("features", "message"),
        [
            pytest.param(None, "continuous_features", id="missing"),
            pytest.param({"stock_volatility_5d": float("nan")}, "finite", id="nan"),
        ],
    )
    def test_invalid_continuous_features_fail_closed(self, features, message):
        builder = _stock_builder(features)
        snapshot = builder.build(
            snapshot_id="snap-invalid-features",
            cutoff_time=_CUTOFF,
        )

        with pytest.raises(ValueError, match=message):
            tensorize_snapshot(snapshot)

    def test_unknown_first_row_feature_fails_versioned_schema(self):
        builder = _stock_builder({"unknown_signal": 0.1})
        snapshot = builder.build(
            snapshot_id="snap-unknown-first-row",
            cutoff_time=_CUTOFF,
        )

        with pytest.raises(ValueError, match="graph-continuous-v1"):
            tensorize_snapshot(snapshot)

    def test_extra_first_row_dimension_fails_versioned_schema(self):
        builder = _stock_builder(
            {"stock_volatility_5d": 0.1, "unexpected_extra": 0.2}
        )
        snapshot = builder.build(snapshot_id="snap-extra-first-row", cutoff_time=_CUTOFF)

        with pytest.raises(ValueError, match="graph-continuous-v1"):
            tensorize_snapshot(snapshot)

    @pytest.mark.parametrize(
        "proxy_name",
        ["stock_code", "ticker_embedding", "identity_embedding"],
    )
    def test_identity_proxy_name_fails_versioned_schema(self, proxy_name: str):
        builder = _stock_builder({proxy_name: 1.0})
        snapshot = builder.build(snapshot_id="snap-identity-proxy", cutoff_time=_CUTOFF)

        with pytest.raises(ValueError, match="graph-continuous-v1"):
            tensorize_snapshot(snapshot)

    def test_extra_edge_dimension_fails_versioned_schema(self):
        snapshot = _seat_buy_snapshot({"buy_amount": 100.0, "unexpected_extra": 1.0})

        with pytest.raises(ValueError, match="graph-continuous-v1"):
            tensorize_snapshot(snapshot)

    def test_dangling_edge_skipped(self):
        """Edges referencing non-existent nodes are skipped."""
        builder = _stock_builder({"stock_volatility_5d": 0.1})
        builder.add_edge(EdgeType.SAME_SEAT_RELAY, "stock-1", "nonexistent")
        snap = builder.build(snapshot_id="snap-d", cutoff_time=_CUTOFF)
        tg = tensorize_snapshot(snap)
        assert len(tg.edge_indices.get("same_seat_relay", [])) == 0
