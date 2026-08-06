"""Tests for placebo tests (Task 18)."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from seatflow.evaluation.placebo import (
    shuffle_seat_assignments,
    shuffle_graph_edges,
    verify_placebo_reproducibility,
    verify_degree_preserved,
    shuffle_seat_rows,
)


def _edge_multiset(edge_index: np.ndarray[Any, Any]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for source, target in edge_index.T.tolist():
        edge = (int(source), int(target))
        counts[edge] = counts.get(edge, 0) + 1
    return counts


class TestShuffledSeat:
    def test_row_aligned_shuffle_is_stable_across_pythonhashseed_processes(
        self,
        tmp_path: Path,
    ) -> None:
        """Catches Python hash-order dependence in seat placebo assignment."""
        script = tmp_path / "seat_shuffle_probe.py"
        script.write_text(
            textwrap.dedent(
                """
                import json
                from seatflow.evaluation.placebo import shuffle_seat_rows

                seat_ids = ["seat-b", "seat-a", "seat-c", "seat-a", "seat-d", "seat-b"]
                stock_ids = ["stock-2", "stock-1", "stock-2", "stock-1", "stock-3", "stock-3"]
                rows = shuffle_seat_rows(seat_ids, stock_ids, seed=20260806)
                print(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))
                """
            ),
            encoding="utf-8",
        )
        env_base = os.environ.copy()
        env_base["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        outputs: list[bytes] = []
        for hash_seed in ("1", "987654"):
            env = env_base | {"PYTHONHASHSEED": hash_seed}
            result = subprocess.run(
                [sys.executable, str(script)],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            outputs.append(result.stdout)

        assert outputs[0] == outputs[1]
        assert json.loads(outputs[0].decode("utf-8")) == [
            "seat-a",
            "seat-b",
            "seat-a",
            "seat-b",
            "seat-c",
            "seat-d",
        ]

    def test_row_aligned_shuffle_preserves_stock_rows_and_global_seat_multiset(self) -> None:
        """Catches dropping seats or ignoring stock grouping during row-aligned shuffle."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-a", "seat-d", "seat-e"]
        stock_ids = ["stock-1", "stock-1", "stock-2", "stock-2", "stock-2", "stock-3"]

        shuffled = shuffle_seat_rows(seat_ids, stock_ids, seed=17)

        assert len(shuffled) == len(seat_ids)
        assert sorted(shuffled) == sorted(seat_ids)
        for stock_id in {"stock-1", "stock-2", "stock-3"}:
            assert sum(1 for stock in stock_ids if stock == stock_id) == len(
                [seat for seat, stock in zip(shuffled, stock_ids) if stock == stock_id]
            )
        assert dict(zip(stock_ids, shuffled)) != dict(zip(stock_ids, seat_ids))

    def test_row_aligned_shuffle_different_seed_changes_semantic_assignment(self) -> None:
        """Catches seed parameters that do not affect a permutable row assignment."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d", "seat-e", "seat-f"]
        stock_ids = ["stock-1", "stock-1", "stock-2", "stock-2", "stock-3", "stock-3"]

        assert shuffle_seat_rows(seat_ids, stock_ids, seed=10) != shuffle_seat_rows(
            seat_ids,
            stock_ids,
            seed=11,
        )

    def test_row_aligned_shuffle_never_returns_identity_when_permutable(self) -> None:
        """Catches seed values that leave the placebo semantically unchanged."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-1", "stock-2", "stock-2"]

        assert shuffle_seat_rows(seat_ids, stock_ids, seed=4) != seat_ids

    @pytest.mark.parametrize(
        ("seat_ids", "stock_ids", "match"),
        [
            ([], [], "empty"),
            (["seat-a"], [], "same length"),
            ([""], ["stock-1"], "seat_ids"),
            (["seat-a"], [""], "stock_ids"),
            (["seat-a"], ["stock-1"], "at least two rows"),
        ],
    )
    def test_row_aligned_shuffle_rejects_invalid_inputs(
        self,
        seat_ids: list[str],
        stock_ids: list[str],
        match: str,
    ) -> None:
        """Catches placebo inputs that silently produce meaningless assignments."""
        with pytest.raises(ValueError, match=match):
            shuffle_seat_rows(seat_ids, stock_ids, seed=42)

    @pytest.mark.parametrize("bad_seed", [True, False, -1])
    def test_seat_shuffle_rejects_bool_or_negative_seed(self, bad_seed: int) -> None:
        """Catches Python bool-as-int and negative placebo seeds."""
        seat_ids = ["seat-a", "seat-b"]
        stock_ids = ["stock-1", "stock-2"]

        with pytest.raises(ValueError, match="seed"):
            shuffle_seat_rows(seat_ids, stock_ids, seed=bad_seed)
        with pytest.raises(ValueError, match="seed"):
            shuffle_seat_assignments(seat_ids, stock_ids, seed=bad_seed)

    def test_reproducibility(self) -> None:
        """Same seed must produce same shuffle."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-2", "stock-3", "stock-4"]
        assert verify_placebo_reproducibility(seat_ids, stock_ids, seed=42)

    def test_different_seeds_differ(self) -> None:
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-2", "stock-3", "stock-4"]
        m1 = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        m2 = shuffle_seat_assignments(seat_ids, stock_ids, seed=99)
        assert m1 != m2

    def test_row_count_preserved(self) -> None:
        """Shuffling must preserve the number of seat participations."""
        seat_ids = ["seat-a", "seat-b", "seat-c"]
        stock_ids = ["stock-1", "stock-2", "stock-3"]
        mapping = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        assert len(mapping) == len(set(seat_ids))

    def test_seat_mapping_is_permutation(self) -> None:
        """Mapping must be a permutation of unique seat IDs."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-a", "seat-b"]
        stock_ids = ["s1", "s2", "s3", "s4", "s5"]
        mapping = shuffle_seat_assignments(seat_ids, stock_ids, seed=42)
        unique_seats = set(seat_ids)
        assert set(mapping.keys()) == unique_seats
        assert set(mapping.values()) == unique_seats

    def test_legacy_mapping_never_returns_identity_when_permutable(self) -> None:
        """Catches compatibility mapping seeds that leave all seats unchanged."""
        seat_ids = ["seat-a", "seat-b", "seat-c", "seat-d"]
        stock_ids = ["stock-1", "stock-1", "stock-2", "stock-2"]

        mapping = shuffle_seat_assignments(seat_ids, stock_ids, seed=4)

        assert any(original != shuffled for original, shuffled in mapping.items())


class TestShuffledGraph:
    def test_reproducibility(self) -> None:
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        shuf1 = shuffle_graph_edges(edge_index, seed=42)
        shuf2 = shuffle_graph_edges(edge_index, seed=42)
        np.testing.assert_array_equal(shuf1, shuf2)

    def test_degree_preserved(self) -> None:
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        shuf = shuffle_graph_edges(edge_index, seed=42, preserve_degree=True)
        assert verify_degree_preserved(edge_index, shuf)
        assert dict(zip(*np.unique(edge_index[1], return_counts=True))) == dict(
            zip(*np.unique(shuf[1], return_counts=True))
        )

    def test_row_count_preserved(self) -> None:
        """Shuffling must preserve the number of edges."""
        edge_index = np.array([[0, 1, 2], [3, 4, 5]])
        shuf = shuffle_graph_edges(edge_index, seed=42)
        assert shuf.shape == edge_index.shape

    def test_empty_edges(self) -> None:
        edge_index = np.array([[], []], dtype=int)
        shuf = shuffle_graph_edges(edge_index, seed=42)
        assert shuf.shape[1] == 0

    def test_semantic_assignment_changes(self) -> None:
        """Shuffling must change at least some target assignments."""
        edge_index = np.array([[0, 1, 2, 3, 4, 5], [3, 4, 5, 0, 1, 2]])
        shuf = shuffle_graph_edges(edge_index, seed=42, preserve_degree=True)
        # At least one target should differ
        assert not np.array_equal(edge_index[1], shuf[1])

    @pytest.mark.parametrize(
        ("edge_index", "seed"),
        [
            (np.array([[0, 1, 2, 3], [1, 2, 3, 0]]), 1),
            (np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]), 228),
        ],
    )
    def test_degree_preserving_graph_shuffle_never_accepts_identity_when_permutable(
        self,
        edge_index: np.ndarray[Any, Any],
        seed: int,
    ) -> None:
        """Catches RNG identity draws that leave a permutable graph unchanged."""
        shuf1 = shuffle_graph_edges(edge_index, seed=seed, preserve_degree=True)
        shuf2 = shuffle_graph_edges(edge_index, seed=seed, preserve_degree=True)

        np.testing.assert_array_equal(shuf1, shuf2)
        assert verify_degree_preserved(edge_index, shuf1)
        assert _edge_multiset(shuf1) != _edge_multiset(edge_index)
        assert sorted(shuf1[1].tolist()) == sorted(edge_index[1].tolist())
        assert not np.any(shuf1[0] == shuf1[1])
        assert not np.array_equal(edge_index[1], shuf1[1])

    def test_degree_preserving_graph_shuffle_fallback_handles_1024_edges_iteratively(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Catches recursive fallback implementations that overflow on normal large graphs."""

        class IdentityShuffleGenerator:
            def shuffle(self, values: np.ndarray[Any, Any]) -> None:
                return None

        monkeypatch.setattr(
            np.random,
            "default_rng",
            lambda seed: cast(np.random.Generator, IdentityShuffleGenerator()),
        )
        node_count = 1024
        edge_index = np.stack(
            [
                np.arange(node_count, dtype=np.int64),
                (np.arange(node_count, dtype=np.int64) + 1) % node_count,
            ],
        )

        shuffled = shuffle_graph_edges(edge_index, seed=42, preserve_degree=True)

        assert verify_degree_preserved(edge_index, shuffled)
        assert sorted(shuffled[1].tolist()) == sorted(edge_index[1].tolist())
        assert _edge_multiset(shuffled) != _edge_multiset(edge_index)
        assert not np.any(shuffled[0] == shuffled[1])

    def test_allow_self_loops_graph_shuffle_never_accepts_identity_when_permutable(self) -> None:
        """Catches semantic identity in the explicit self-loop-allowed path."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])

        shuffled = shuffle_graph_edges(edge_index, seed=1, allow_self_loops=True)

        assert verify_degree_preserved(edge_index, shuffled)
        assert sorted(shuffled[1].tolist()) == sorted(edge_index[1].tolist())
        assert _edge_multiset(shuffled) != _edge_multiset(edge_index)

    def test_degree_preserving_graph_shuffle_rejects_nonpermutable_without_identity(self) -> None:
        """Catches labeling an unchanged reciprocal pair as a graph shuffle."""
        with pytest.raises(ValueError, match="semantic"):
            shuffle_graph_edges(np.array([[0, 1], [1, 0]]), seed=42, preserve_degree=True)

    @pytest.mark.parametrize(
        ("edge_index", "match"),
        [
            (np.array([0, 1, 2]), "shape"),
            (np.array([[0, 1], [1, 2]], dtype=float), "integer"),
            (np.array([[-1, 1], [1, 2]]), "non-negative"),
            (np.array([[0, 1], [0, 2]]), "self-loop"),
        ],
    )
    def test_graph_shuffle_rejects_invalid_shape_values_and_self_loops(
        self,
        edge_index: np.ndarray[Any, Any],
        match: str,
    ) -> None:
        """Catches malformed graph placebo inputs before they reach model evaluation."""
        with pytest.raises(ValueError, match=match):
            shuffle_graph_edges(edge_index, seed=42)

    @pytest.mark.parametrize("bad_seed", [True, False, -1])
    def test_graph_shuffle_rejects_bool_or_negative_seed(self, bad_seed: int) -> None:
        """Catches Python bool-as-int and negative graph placebo seeds."""
        with pytest.raises(ValueError, match="seed"):
            shuffle_graph_edges(np.array([[0, 1], [1, 0]]), seed=bad_seed)

    def test_graph_shuffle_can_keep_self_loops_when_explicitly_allowed(self) -> None:
        """Catches implicit self-loop policy changes."""
        edge_index = np.array([[0, 1, 2], [0, 2, 1]])

        shuffled = shuffle_graph_edges(edge_index, seed=42, allow_self_loops=True)

        assert shuffled.shape == edge_index.shape
