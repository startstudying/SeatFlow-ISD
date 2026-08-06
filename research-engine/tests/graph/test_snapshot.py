"""Tests for graph snapshot and state store (Task 14)."""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import pytest

from seatflow.graph.schema import EdgeType
from seatflow.graph.snapshot import GraphSnapshotBuilder, GraphSnapshot
from seatflow.graph.state_store import GraphStateStore
from seatflow.graph.candidates import PropagationCandidateFinder, PropagationLink


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


_CUTOFF = _utc("2025-01-03T00:00:00+00:00")
_SOURCE_TIME = _utc("2025-01-01T00:00:00+00:00")
_TARGET_TIME = _utc("2025-01-02T00:00:00+00:00")


def _participation(seat: str, stock: str, side: str, at: datetime | None) -> dict[str, object]:
    return {"seat_id": seat, "stock_id": stock, "side": side, "available_time": at}


def _semantic_relay_case(
    prefix: str = "",
) -> tuple[str, list[dict[str, object]], set[str]]:
    source = f"{prefix}source"
    strong, medium, weak = (f"{prefix}{name}" for name in ("z-strong", "m-medium", "a-weak"))
    seat_a, seat_b = f"{prefix}seat-A", f"{prefix}seat-B"
    parts = [
        _participation(seat_a, source, "BUY", _SOURCE_TIME),
        _participation(seat_b, source, "BUY", _SOURCE_TIME),
        _participation(seat_a, strong, "BUY", _TARGET_TIME),
        _participation(seat_b, strong, "BUY", _TARGET_TIME),
        _participation(seat_a, medium, "BUY", _TARGET_TIME),
        _participation(seat_a, weak, "SELL", _TARGET_TIME),
    ]
    return source, parts, {source, strong, medium, weak}


class TestPropagationCandidates:
    def test_seat_relay_finds_shared_seat(self):
        parts = [
            _participation("seat-A", "source", "BUY", _SOURCE_TIME),
            _participation("seat-A", "target", "BUY", _TARGET_TIME),
        ]
        links = PropagationCandidateFinder().find_seat_relay(
            "source", parts, {"source", "target"}, cutoff_time=_CUTOFF
        )
        assert [(link.target_stock_id, link.relation_type) for link in links] == [
            ("target", EdgeType.SAME_SEAT_RELAY)
        ]

    def test_top_k_retains_complete_semantic_tie_boundary(self):
        parts = [_participation("seat-A", "source", "BUY", _SOURCE_TIME)]
        targets = [f"stock-{index}" for index in range(10)]
        parts.extend(_participation("seat-A", target, "BUY", _TARGET_TIME) for target in targets)
        links = PropagationCandidateFinder(top_k=3).find_seat_relay(
            "source", parts, {"source", *targets}, cutoff_time=_CUTOFF
        )
        assert {link.target_stock_id for link in links} == set(targets)

    def test_equal_semantic_evidence_does_not_reorder_renamed_targets(self):
        def ordered_targets(first_id: str, second_id: str) -> list[str]:
            parts = [
                _participation("seat-A", "source", "BUY", _SOURCE_TIME),
                _participation("seat-A", first_id, "BUY", _TARGET_TIME),
                _participation("seat-A", second_id, "BUY", _TARGET_TIME),
            ]
            return [
                link.target_stock_id
                for link in PropagationCandidateFinder(top_k=1).find_seat_relay(
                    "source",
                    parts,
                    {"source", first_id, second_id},
                    cutoff_time=_CUTOFF,
                )
            ]

        assert ordered_targets("z-first", "a-second") == ["z-first", "a-second"]
        assert ordered_targets("a-renamed-first", "z-renamed-second") == [
            "a-renamed-first",
            "z-renamed-second",
        ]

    def test_max_hops_enforcement(self):
        links = [PropagationLink("source", "target", EdgeType.SAME_SEAT_RELAY, 3, 1.0)]
        assert PropagationCandidateFinder(max_hops=2).enforce_max_hops(links) == []

    def test_theme_diffusion_uses_pit_membership_strength(self):
        memberships = {
            "theme-tech": [
                {"stock_id": "source", "available_time": _SOURCE_TIME, "membership_strength": 0.9},
                {"stock_id": "strong", "available_time": _TARGET_TIME, "membership_strength": 0.8},
                {"stock_id": "weak", "available_time": _TARGET_TIME, "membership_strength": 0.4},
            ]
        }
        links = PropagationCandidateFinder().find_theme_diffusion(
            "source", memberships, {"source", "strong", "weak"}, cutoff_time=_CUTOFF
        )
        assert [(link.target_stock_id, link.candidate_score) for link in links] == [
            ("strong", 0.8),
            ("weak", 0.4),
        ]

    def test_theme_top_k_retains_complete_semantic_tie_boundary(self):
        memberships = {
            "theme-tech": [
                {"stock_id": "source", "available_time": _SOURCE_TIME, "membership_strength": 0.9},
                {"stock_id": "target-b", "available_time": _TARGET_TIME, "membership_strength": 0.5},
                {"stock_id": "target-a", "available_time": _TARGET_TIME, "membership_strength": 0.5},
            ]
        }
        links = PropagationCandidateFinder(top_k=1).find_theme_diffusion(
            "source",
            memberships,
            {"source", "target-a", "target-b"},
            cutoff_time=_CUTOFF,
        )
        assert {link.target_stock_id for link in links} == {"target-a", "target-b"}

    def test_identity_renaming_does_not_change_candidate_scores(self):
        finder = PropagationCandidateFinder()

        def scores(prefix: str) -> list[float]:
            source, parts, universe = _semantic_relay_case(prefix)
            return [
                link.candidate_score
                for link in finder.find_seat_relay(source, parts, universe, cutoff_time=_CUTOFF)
            ]

        assert scores("") == scores("renamed-") == [1.0, 0.5, 0.25]

    def test_candidate_order_is_stable_across_python_hash_seeds(self):
        script = """
import json
from tests.graph.test_snapshot import _CUTOFF, _semantic_relay_case
from seatflow.graph.candidates import PropagationCandidateFinder
source, parts, universe = _semantic_relay_case()
links = PropagationCandidateFinder().find_seat_relay(
    source, parts, universe, cutoff_time=_CUTOFF
)
print(json.dumps([(link.target_stock_id, link.candidate_score) for link in links]))
"""
        project_root = Path(__file__).resolve().parents[2]
        outputs = []
        for seed in ("1", "999"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(project_root / "src")}
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            outputs.append(json.loads(completed.stdout))

        expected = [["z-strong", 1.0], ["m-medium", 0.5], ["a-weak", 0.25]]
        assert outputs == [expected, expected]

    @pytest.mark.parametrize("unavailable_time", [None, _utc("2025-01-04T00:00:00+00:00")])
    def test_unavailable_candidate_evidence_is_rejected(self, unavailable_time):
        parts = [
            _participation("seat-A", "source", "BUY", _SOURCE_TIME),
            _participation("seat-A", "target", "BUY", unavailable_time),
        ]
        links = PropagationCandidateFinder().find_seat_relay(
            "source", parts, {"source", "target"}, cutoff_time=_CUTOFF
        )
        assert links == []
