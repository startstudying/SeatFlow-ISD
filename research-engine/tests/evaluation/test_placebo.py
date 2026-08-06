"""Tests for placebo tests (Task 18)."""

import numpy as np

from seatflow.evaluation.placebo import (
    shuffle_seat_assignments,
    shuffle_graph_edges,
    verify_placebo_reproducibility,
    verify_degree_preserved,
)


class TestShuffledSeat:
    def test_reproducibility(self):
        """Same seed must produce same shuffle."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-2", "stock-3", "stock-4"]
        assert verify_placebo_reproducibility(seat_ids, stock_ids, seed=42)

    def test_different_seeds_differ(self):
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-2", "stock-3", "stock-4"]
        m1 = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        m2 = shuffle_seat_assignments(seat_ids, stock_ids, seed=99)
        assert m1 != m2

    def test_row_count_preserved(self):
        """Shuffling must preserve the number of seat participations."""
        seat_ids = ["seat-a", "seat-b", "seat-c"]
        stock_ids = ["stock-1", "stock-2", "stock-3"]
        mapping = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        assert len(mapping) == len(set(seat_ids))

    def test_seat_mapping_is_permutation(self):
        """Mapping must be a permutation of unique seat IDs."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-a", "seat-b"]
        stock_ids = ["s1", "s2", "s3", "s4", "s5"]
        mapping = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        unique_seats = set(seat_ids)
        assert set(mapping.keys()) == unique_seats
        assert set(mapping.values()) == unique_seats


class TestShuffledGraph:
    def test_reproducibility(self):
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        shuf1 = shuffle_graph_edges(edge_index, seed=42)
        shuf2 = shuffle_graph_edges(edge_index, seed=42)
        np.testing.assert_array_equal(shuf1, shuf2)

    def test_degree_preserved(self):
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        shuf = shuffle_graph_edges(edge_index, seed=42, preserve_degree=True)
        assert verify_degree_preserved(edge_index, shuf)

    def test_row_count_preserved(self):
        """Shuffling must preserve the number of edges."""
        edge_index = np.array([[0, 1, 2], [3, 4, 5]])
        shuf = shuffle_graph_edges(edge_index, seed=42)
        assert shuf.shape == edge_index.shape

    def test_empty_edges(self):
        edge_index = np.array([[], []], dtype=int)
        shuf = shuffle_graph_edges(edge_index, seed=42)
        assert shuf.shape[1] == 0

    def test_semantic_assignment_changes(self):
        """Shuffling must change at least some target assignments."""
        edge_index = np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2]])
        shuf = shuffle_graph_edges(edge_index, seed=42, preserve_degree=True)
        # At least one target should differ
        assert not np.array_equal(edge_index[1], shuf[1])
