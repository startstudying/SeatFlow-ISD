"""Placebo tests: Shuffled Seat and Shuffled Graph (Task 18).

These tests verify that the model's performance is NOT due to spurious
patterns. By shuffling seat identities or graph edges, we expect a
significant performance drop if the model truly uses the structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

NDArrayAny = np.ndarray[Any, Any]


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
    """Shuffle unique seat assignments deterministically.

    This compatibility wrapper returns the legacy ``seat_id -> shuffled_seat_id``
    mapping. New evaluation code should prefer :func:`shuffle_seat_rows`, which
    is row-aligned and validates stock grouping.

    Args:
        seat_ids: Original seat IDs in order
        stock_ids: Original stock IDs (one per seat participation)
        seed: Random seed for deterministic shuffling

    Returns:
        Mapping from original seat_id -> shuffled seat_id
    """
    _validate_seed(seed)
    _validate_seat_rows(seat_ids, stock_ids)
    unique_seats = _stable_unique(seat_ids)
    if len(unique_seats) < 2:
        raise ValueError("seat_ids must contain at least two unique seats")
    shuffled = _rotate(sorted(unique_seats), seed)
    if all(original == placebo for original, placebo in zip(unique_seats, shuffled)):
        shuffled = _rotate(sorted(unique_seats), seed + 1)
    return dict(zip(unique_seats, shuffled))


def shuffle_seat_rows(
    seat_ids: Sequence[str],
    stock_ids: Sequence[str],
    *,
    seed: int = 42,
) -> list[str]:
    """Return a deterministic row-aligned seat placebo assignment.

    The output has exactly one shuffled seat ID per input row. It preserves the
    sample count, the global seat multiset, and the number of participating rows
    for every stock. Stock IDs are validated and used to define deterministic
    first-seen groups; no Python hash ordering participates in the shuffle.
    """
    _validate_seed(seed)
    _validate_seat_rows(seat_ids, stock_ids)
    grouped_row_indices: dict[str, list[int]] = {}
    for index, stock_id in enumerate(stock_ids):
        grouped_row_indices.setdefault(stock_id, []).append(index)

    shuffled_multiset = _rotate(sorted(seat_ids), seed)
    result = [""] * len(seat_ids)
    cursor = 0
    for row_indices in grouped_row_indices.values():
        group_seats = shuffled_multiset[cursor : cursor + len(row_indices)]
        cursor += len(row_indices)
        for row_index, seat_id in zip(row_indices, group_seats):
            result[row_index] = seat_id
    if len(set(seat_ids)) >= 2 and result == list(seat_ids):
        return shuffle_seat_rows(seat_ids, stock_ids, seed=seed + 1)
    return result


def shuffle_graph_edges(
    edge_index: NDArrayAny,
    *,
    seed: int = 42,
    preserve_degree: bool = True,
    allow_self_loops: bool = False,
) -> NDArrayAny:
    """Shuffle graph edges while optionally preserving degree distribution.

    Args:
        edge_index: [2, num_edges] array of (source, target) pairs
        seed: Random seed
        preserve_degree: If True, preserve in-degree and out-degree
        allow_self_loops: If False, reject input self-loops and avoid creating
            shuffled self-loops when a valid permutation exists.

    Returns:
        Shuffled edge_index [2, num_edges]
    """
    _validate_seed(seed)
    _validate_edge_index(edge_index, allow_self_loops=allow_self_loops)
    rng = np.random.default_rng(seed)
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return edge_index.copy()

    if preserve_degree:
        sources = edge_index[0].copy()
        targets = edge_index[1].copy()
        targets = _shuffle_targets_without_self_loops(
            sources,
            targets,
            rng=rng,
            allow_self_loops=allow_self_loops,
        )
        return cast(NDArrayAny, np.stack([sources, targets]))
    else:
        all_nodes = np.unique(edge_index)
        new_sources = rng.choice(all_nodes, size=num_edges)
        new_targets = rng.choice(all_nodes, size=num_edges)
        shuffled = cast(NDArrayAny, np.stack([new_sources, new_targets]))
        if not allow_self_loops and np.any(shuffled[0] == shuffled[1]):
            raise ValueError("full graph shuffle created a self-loop; use allow_self_loops=True")
        return shuffled


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
    edge_index: NDArrayAny,
    shuffled: NDArrayAny,
) -> bool:
    """Verify that degree distribution is preserved after shuffling."""
    if edge_index.shape[1] == 0:
        return True
    return _degree_counts(edge_index[0]) == _degree_counts(shuffled[0]) and _degree_counts(
        edge_index[1]
    ) == _degree_counts(shuffled[1])


def _validate_seat_rows(seat_ids: Sequence[str], stock_ids: Sequence[str]) -> None:
    if len(seat_ids) != len(stock_ids):
        raise ValueError("seat_ids and stock_ids must have the same length")
    if not seat_ids:
        raise ValueError("seat_ids and stock_ids cannot be empty")
    if any(not isinstance(seat_id, str) or not seat_id.strip() for seat_id in seat_ids):
        raise ValueError("seat_ids must contain non-empty strings")
    if any(not isinstance(stock_id, str) or not stock_id.strip() for stock_id in stock_ids):
        raise ValueError("stock_ids must contain non-empty strings")
    if len(seat_ids) < 2:
        raise ValueError("seat placebo requires at least two rows")


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _rotate(values: list[str], seed: int) -> list[str]:
    _validate_seed(seed)
    if not values:
        return []
    offset = seed % len(values)
    return values[offset:] + values[:offset]


def _validate_edge_index(edge_index: NDArrayAny, *, allow_self_loops: bool) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if not np.issubdtype(edge_index.dtype, np.integer):
        raise ValueError("edge_index must use an integer dtype")
    if np.any(edge_index < 0):
        raise ValueError("edge_index node IDs must be non-negative")
    if not allow_self_loops and np.any(edge_index[0] == edge_index[1]):
        raise ValueError("edge_index contains a self-loop; use allow_self_loops=True")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _shuffle_targets_without_self_loops(
    sources: NDArrayAny,
    targets: NDArrayAny,
    *,
    rng: np.random.Generator,
    allow_self_loops: bool,
) -> NDArrayAny:
    if allow_self_loops:
        shuffled = targets.copy()
        rng.shuffle(shuffled)
        return shuffled

    for _ in range(128):
        shuffled = targets.copy()
        rng.shuffle(shuffled)
        if _is_valid_semantic_graph_shuffle(sources, targets, shuffled):
            return shuffled
    fallback = _find_deterministic_target_shuffle(sources, targets)
    if fallback is not None:
        return fallback
    raise ValueError("cannot create a semantic graph shuffle without creating a self-loop")


def _is_valid_semantic_graph_shuffle(
    sources: NDArrayAny,
    original_targets: NDArrayAny,
    candidate_targets: NDArrayAny,
) -> bool:
    if np.any(sources == candidate_targets):
        return False
    return _edge_multiset(sources, candidate_targets) != _edge_multiset(sources, original_targets)


def _find_deterministic_target_shuffle(sources: NDArrayAny, targets: NDArrayAny) -> NDArrayAny | None:
    target_counts: dict[int, int] = _degree_counts(targets)
    source_values = [int(source) for source in sources.tolist()]
    remaining_edges = _edge_multiset(sources, targets)
    assignment: list[int] = []
    memo: set[
        tuple[
            int,
            tuple[tuple[int, int], ...],
            tuple[tuple[tuple[int, int], int], ...],
        ]
    ] = set()

    def search(position: int) -> list[int] | None:
        if position == len(source_values):
            return assignment.copy() if remaining_edges else None

        state = (
            position,
            tuple(sorted(target_counts.items())),
            tuple(sorted(remaining_edges.items())),
        )
        if state in memo:
            return None

        source = source_values[position]
        for target in sorted(target_counts):
            if target == source or target_counts[target] == 0:
                continue
            edge = (source, target)
            edge_count = remaining_edges.get(edge, 0)
            target_counts[target] -= 1
            if target_counts[target] == 0:
                del target_counts[target]
            if edge_count > 0:
                if edge_count == 1:
                    del remaining_edges[edge]
                else:
                    remaining_edges[edge] = edge_count - 1
            assignment.append(target)

            found = search(position + 1)
            if found is not None:
                return found

            assignment.pop()
            if edge_count > 0:
                remaining_edges[edge] = edge_count
            target_counts[target] = target_counts.get(target, 0) + 1

        memo.add(state)
        return None

    found_assignment = search(0)
    if found_assignment is None:
        return None
    return cast(NDArrayAny, np.array(found_assignment, dtype=targets.dtype))


def _edge_multiset(sources: NDArrayAny, targets: NDArrayAny) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for source, target in zip(sources, targets):
        edge = (int(source), int(target))
        counts[edge] = counts.get(edge, 0) + 1
    return counts


def _degree_counts(nodes: NDArrayAny) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        node_id = int(node)
        counts[node_id] = counts.get(node_id, 0) + 1
    return counts
