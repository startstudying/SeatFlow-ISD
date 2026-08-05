"""Placebo tests: Shuffled Seat and Shuffled Graph (Task 18).

These tests verify that the model's performance is NOT due to spurious
patterns. By shuffling seat identities or graph edges, we expect a
significant performance drop if the model truly uses the structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PlaceboResult:
    """Result of a placebo test."""

    variant: str  # "shuffled_seat" or "shuffled_graph"
    seed: int
    row_count: int
    degree_preserved: bool
    metric_delta: float  # original_metric - placebo_metric


def shuffle_seat_assignments(
    seat_ids: list[str],
    stock_ids: list[str],
    *,
    seed: int = 42,
) -> dict[str, str]:
    """Shuffle seat-to-stock assignments deterministically.

    Preserves the count of seats per stock but reassigns which seat
    participates in which stock. This breaks the semantic meaning of
    seat identity while keeping the data shape identical.

    Args:
        seat_ids: Original seat IDs in order
        stock_ids: Original stock IDs (one per seat participation)
        seed: Random seed for deterministic shuffling

    Returns:
        Mapping from original seat_id -> shuffled seat_id
    """
    rng = np.random.default_rng(seed)
    unique_seats = list(set(seat_ids))
    shuffled = unique_seats.copy()
    rng.shuffle(shuffled)
    return dict(zip(unique_seats, shuffled))


def shuffle_graph_edges(
    edge_index: np.ndarray,
    *,
    seed: int = 42,
    preserve_degree: bool = True,
) -> np.ndarray:
    """Shuffle graph edges while optionally preserving degree distribution.

    Args:
        edge_index: [2, num_edges] array of (source, target) pairs
        seed: Random seed
        preserve_degree: If True, preserve in-degree and out-degree

    Returns:
        Shuffled edge_index [2, num_edges]
    """
    rng = np.random.default_rng(seed)
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return edge_index.copy()

    if preserve_degree:
        # Preserve out-degree: shuffle targets only
        sources = edge_index[0].copy()
        targets = edge_index[1].copy()
        rng.shuffle(targets)
        return np.stack([sources, targets])
    else:
        # Full shuffle: randomly reassign both sources and targets
        all_nodes = np.unique(edge_index)
        new_sources = rng.choice(all_nodes, size=num_edges)
        new_targets = rng.choice(all_nodes, size=num_edges)
        return np.stack([new_sources, new_targets])


def verify_placebo_reproducibility(
    seat_ids: list[str],
    stock_ids: list[str],
    seed: int = 42,
) -> bool:
    """Verify that the same seed produces the same shuffle."""
    mapping1 = shuffle_seat_assignments(seat_ids, stock_ids, seed=seed)
    mapping2 = shuffle_seat_assignments(seat_ids, stock_ids, seed=seed)
    return mapping1 == mapping2


def verify_degree_preserved(
    edge_index: np.ndarray,
    shuffled: np.ndarray,
) -> bool:
    """Verify that degree distribution is preserved after shuffling."""
    if edge_index.shape[1] == 0:
        return True
    orig_out_degree: dict[int, int] = {}
    for s in edge_index[0]:
        orig_out_degree[int(s)] = orig_out_degree.get(int(s), 0) + 1

    shuf_out_degree: dict[int, int] = {}
    for s in shuffled[0]:
        shuf_out_degree[int(s)] = shuf_out_degree.get(int(s), 0) + 1

    return orig_out_degree == shuf_out_degree
