"""Walk-forward split construction (Task 13).

The cycle is the *split atom*: a whole cycle must appear in exactly one
of train / validation / test / blind, never split across them. The
:class:`WalkForwardFold` records which cycle_ids belong to each segment
plus an embargo of at least 5 calendar days between train and validation
to prevent label leakage from the 5-day horizon.

Design
------
A single chronological partition is produced from the end backward:

1. Sort all cycles by ``cycle_start``.
2. If the dataset is large enough (at least 50 cycles), reserve the
   last ``blind_days`` calendar days as the **blind** holdout. These
   cycles are not assigned to any segment. For smaller datasets no
   blind window is reserved so all cycles participate in development.
3. Of the development cycles, working from the end backward:
   - The final ``test_days`` calendar days form the **test** segment.
   - The ``validation_days`` calendar days immediately before the test
     segment form the **validation** segment.
   - An ``embargo_days`` gap (>= 5) separates validation from the
     training cycles that precede it.
   - All earlier cycles form the **train** segment.

The key invariant: every cycle appears in exactly one of train /
validation / test / blind. A cycle is "unassigned" only if it falls in
the blind window; with a small dataset the blind window is empty so all
cycles are assigned to development.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from seatflow.dataset.cycle import EventCycle
from seatflow.domain.time import assert_aware


# ── Minimum cycles to justify a blind holdout ─────────────────────────

_MIN_CYCLES_FOR_BLIND = 50


# ── Dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward split fold.

    Attributes
    ----------
    fold_id:
        Zero-based integer identifier.
    train_cycle_ids:
        Cycle IDs assigned to the training set.
    validation_cycle_ids:
        Cycle IDs assigned to the validation set.
    test_cycle_ids:
        Cycle IDs assigned to the test set.
    embargo_days:
        Calendar days of embargo between train and validation (>= 5).
    train_start:
        Earliest ``cycle_start`` in the train set.
    train_end:
        Latest ``cycle_start`` in the train set.
    validation_end:
        End of the validation window (latest ``cycle_start`` in val).
    test_end:
        End of the test window (latest ``cycle_start`` in test).
    """

    fold_id: int
    train_cycle_ids: frozenset[str]
    validation_cycle_ids: frozenset[str]
    test_cycle_ids: frozenset[str]
    embargo_days: int
    train_start: Optional[datetime]
    train_end: Optional[datetime]
    validation_end: Optional[datetime]
    test_end: Optional[datetime]


# ── Public API ────────────────────────────────────────────────────────


def create_walk_forward_splits(
    cycles: Sequence[EventCycle],
    *,
    embargo_days: int = 5,
    validation_days: int = 60,
    test_days: int = 20,
    blind_days: int = 120,
) -> list[WalkForwardFold]:
    """Create walk-forward folds from a sequence of cycles.

    Parameters
    ----------
    cycles:
        Sorted or unsorted sequence of :class:`EventCycle`. They will be
        sorted by ``cycle_start`` internally.
    embargo_days:
        Calendar-day gap between the train window and the validation
        window. Must be >= 5 to prevent label leakage.
    validation_days:
        Calendar-day span for the validation window.
    test_days:
        Calendar-day span for the test window.
    blind_days:
        Calendar-day span reserved at the end as the blind holdout.
        Cycles falling within the blind period are not assigned to any
        segment. A blind window is only created when the dataset has at
        least 50 cycles; otherwise all cycles participate in development.

    Returns
    -------
    list[WalkForwardFold]
        A single fold partitioning the cycles into train / validation /
        test (plus an implicit blind tail for large datasets). The list
        is empty if there are too few cycles to fill validation + test.
    """
    if embargo_days < 5:
        raise ValueError(f"embargo_days must be >= 5, got {embargo_days}")
    if not cycles:
        return []

    sorted_cycles = sorted(cycles, key=lambda c: c.cycle_start)
    for c in sorted_cycles:
        assert_aware(c.cycle_start, "cycle_start")

    overall_end = sorted_cycles[-1].cycle_start
    n_cycles = len(sorted_cycles)

    # ── Blind window ─────────────────────────────────────────────────
    effective_blind = blind_days if n_cycles >= _MIN_CYCLES_FOR_BLIND else 0
    blind_start = overall_end - timedelta(days=effective_blind)

    # Development cycles: everything before the blind window (or all
    # cycles if the blind window is empty).
    dev_cycles = [c for c in sorted_cycles if c.cycle_start <= blind_start]
    if effective_blind == 0:
        dev_cycles = list(sorted_cycles)
    if not dev_cycles:
        return []

    dev_end = dev_cycles[-1].cycle_start

    # ── Test window (from the end of dev backward) ───────────────────
    test_start = dev_end - timedelta(days=test_days)
    test_cycles = [c for c in dev_cycles if c.cycle_start > test_start]
    if not test_cycles:
        return []

    # ── Validation window (immediately before test) ─────────────────
    val_end = test_start
    val_start = val_end - timedelta(days=validation_days)
    val_cycles = [c for c in dev_cycles if val_start <= c.cycle_start <= val_end]
    if not val_cycles:
        return []

    # ── Train window (everything before val, minus embargo) ─────────
    train_cutoff = val_start - timedelta(days=embargo_days)
    train_cycles = [c for c in dev_cycles if c.cycle_start < train_cutoff]
    if not train_cycles:
        return []

    return [
        WalkForwardFold(
            fold_id=0,
            train_cycle_ids=frozenset(c.cycle_id for c in train_cycles),
            validation_cycle_ids=frozenset(c.cycle_id for c in val_cycles),
            test_cycle_ids=frozenset(c.cycle_id for c in test_cycles),
            embargo_days=embargo_days,
            train_start=train_cycles[0].cycle_start,
            train_end=train_cycles[-1].cycle_start,
            validation_end=val_cycles[-1].cycle_start if val_cycles else None,
            test_end=test_cycles[-1].cycle_start,
        )
    ]
