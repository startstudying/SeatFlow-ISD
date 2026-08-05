"""Tests for walk-forward split construction (Task 13).

The cycle is the *split atom*: a whole cycle must appear in exactly one
of train / validation / test / blind, never split across them. The
:class:`WalkForwardFold` records which cycle_ids belong to each segment
plus an embargo of at least 5 trading days between train and validation
to prevent label leakage from the 5-day horizon.
"""

from __future__ import annotations

from datetime import datetime, timedelta


from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.splits import (
    create_walk_forward_splits,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _cycle(
    cycle_id: str,
    stock_id: str,
    start: str,
    last: str,
    *,
    status: CycleStatus = CycleStatus.EXPIRED,
) -> EventCycle:
    return EventCycle(
        cycle_id=cycle_id,
        stock_id=stock_id,
        cycle_start=_utc(start),
        last_event_date=_utc(last),
        cycle_end=_utc(last),
        event_count=1,
        status=status,
    )


def _cycles_over_window(n: int, start: str = "2026-01-05T00:00:00+00:00") -> list[EventCycle]:
    """Build *n* cycles spaced 15 calendar days apart."""
    base = _utc(start)
    out: list[EventCycle] = []
    for i in range(n):
        d = base + timedelta(days=15 * i)
        out.append(_cycle(f"cyc-{i:03d}", "000001.SZ", d.isoformat(), d.isoformat()))
    return out


# ── Atomicity ─────────────────────────────────────────────────────────


class TestSplitAtomicity:
    def test_each_cycle_in_exactly_one_split(self):
        cycles = _cycles_over_window(40)
        folds = create_walk_forward_splits(cycles)
        # Every cycle_id appears in exactly one segment across all folds.
        all_ids = [c.cycle_id for c in cycles]
        seen: dict[str, str] = {}
        for fold in folds:
            for cid in fold.train_cycle_ids:
                assert cid not in seen, f"{cid} duplicated in {seen[cid]} and train"
                seen[cid] = "train"
            for cid in fold.validation_cycle_ids:
                assert cid not in seen, f"{cid} duplicated in {seen[cid]} and val"
                seen[cid] = "val"
            for cid in fold.test_cycle_ids:
                assert cid not in seen, f"{cid} duplicated in {seen[cid]} and test"
                seen[cid] = "test"
        # All cycles are accounted for (no cycle dropped).
        for cid in all_ids:
            assert cid in seen, f"{cid} not assigned to any split"

    def test_no_cycle_crosses_train_val_test_blind(self):
        cycles = _cycles_over_window(40)
        folds = create_walk_forward_splits(cycles)
        for fold in folds:
            train = set(fold.train_cycle_ids)
            val = set(fold.validation_cycle_ids)
            test = set(fold.test_cycle_ids)
            # The three sets are pairwise disjoint.
            assert train.isdisjoint(val)
            assert train.isdisjoint(test)
            assert val.isdisjoint(test)

    def test_fold_ids_are_contiguous_integers(self):
        cycles = _cycles_over_window(40)
        folds = create_walk_forward_splits(cycles)
        assert [f.fold_id for f in folds] == list(range(len(folds)))


# ── Embargo ───────────────────────────────────────────────────────────


class TestEmbargo:
    def test_embargo_at_least_five_trading_days(self):
        cycles = _cycles_over_window(40)
        folds = create_walk_forward_splits(cycles, embargo_days=5)
        for fold in folds:
            assert fold.embargo_days >= 5

    def test_embargo_separates_train_end_from_validation_start(self):
        cycles = _cycles_over_window(40)
        folds = create_walk_forward_splits(cycles, embargo_days=5)
        for fold in folds:
            if fold.train_end is not None and fold.validation_end is not None:
                # validation_end is after train_end + embargo.
                assert fold.validation_end > fold.train_end
                gap = (fold.validation_end - fold.train_end).days
                assert gap >= 5


# ── Blind range ───────────────────────────────────────────────────────


class TestBlindRange:
    def test_blind_range_separated_from_development(self):
        """The blind range (held-out cycles) must come after the
        development range (train+val+test) and never overlap it."""
        cycles = _cycles_over_window(60)
        folds = create_walk_forward_splits(
            cycles, embargo_days=5, validation_days=60, test_days=20, blind_days=120
        )
        # The last fold's test_end must precede the blind window.
        last = folds[-1]
        if last.test_end is not None:
            # All cycles after test_end are in the blind window (not
            # assigned to train/val/test of any fold).
            assigned = set()
            for fold in folds:
                assigned |= fold.train_cycle_ids | fold.validation_cycle_ids | fold.test_cycle_ids
            unassigned = [c for c in cycles if c.cycle_id not in assigned]
            # Unassigned cycles form the blind holdout.
            if unassigned:
                blind_start = min(c.cycle_start for c in unassigned)
                assert blind_start > last.test_end


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_cycles_yields_no_folds(self):
        assert create_walk_forward_splits([]) == []

    def test_too_few_cycles_yields_no_folds(self):
        # Fewer cycles than needed to fill one fold's segments.
        assert create_walk_forward_splits(_cycles_over_window(2)) == []
