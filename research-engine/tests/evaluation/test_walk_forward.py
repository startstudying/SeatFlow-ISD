"""Tests for Walk-Forward evaluation (Task 18)."""

from datetime import datetime, timezone, timedelta


from seatflow.evaluation.walk_forward import (
    WalkForwardFold,
    create_walk_forward_splits,
    verify_cycle_atomicity,
)


def _utc(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


class TestWalkForward:
    def test_chronological_order(self):
        """Train must end before validation starts, which must end before test starts."""
        cycles = [
            (f"c{i}", _utc(2024, 1, 1) + timedelta(days=i * 10),
             _utc(2024, 1, 1) + timedelta(days=i * 10 + 5))
            for i in range(30)
        ]
        folds = create_walk_forward_splits(cycles, embargo_days=5)
        assert len(folds) >= 1
        for fold in folds:
            assert fold.train_end <= fold.validation_start
            assert fold.validation_end <= fold.test_start

    def test_embargo_at_least_5_days(self):
        """Embargo between train/val/test must be at least 5 trading days."""
        # Create a trading calendar with enough span
        calendar = [
            _utc(2024, 1, 1) + timedelta(days=i)
            for i in range(500)
            if (_utc(2024, 1, 1) + timedelta(days=i)).weekday() < 5  # weekdays
        ]
        # Create cycles spanning a long period with gaps
        cycles = [
            (f"c{i}", calendar[i * 5], calendar[i * 5 + 3])
            for i in range(50)
        ]
        folds = create_walk_forward_splits(
            cycles, embargo_days=5, validation_days=60, test_days=20
        )
        assert len(folds) >= 1
        for fold in folds:
            assert fold.embargo_days >= 5

    def test_cycle_atomicity(self):
        """Every cycle_id must belong to exactly one split."""
        cycles = [
            (f"c{i}", _utc(2024, 1, 1) + timedelta(days=i * 10),
             _utc(2024, 1, 1) + timedelta(days=i * 10 + 5))
            for i in range(30)
        ]
        folds = create_walk_forward_splits(cycles, embargo_days=5)
        assert len(folds) >= 1
        all_ids = {c[0] for c in cycles}
        for fold in folds:
            assert verify_cycle_atomicity(fold, all_ids)

    def test_no_cycle_in_multiple_splits(self):
        """A cycle cannot be in both train and test."""
        fold = WalkForwardFold(
            fold_id=0,
            train_cycle_ids=frozenset({"c1", "c2"}),
            validation_cycle_ids=frozenset({"c3"}),
            test_cycle_ids=frozenset({"c4"}),
            embargo_days=5,
            train_start=_utc(2024, 1, 1),
            train_end=_utc(2024, 6, 1),
            validation_end=_utc(2024, 8, 1),
            test_end=_utc(2024, 9, 1),
        )
        assert verify_cycle_atomicity(fold, {"c1", "c2", "c3", "c4"})

    def test_cycle_overlap_detected(self):
        """Overlapping cycles should fail atomicity check."""
        fold = WalkForwardFold(
            fold_id=0,
            train_cycle_ids=frozenset({"c1", "c2"}),
            validation_cycle_ids=frozenset({"c2", "c3"}),  # c2 overlaps
            test_cycle_ids=frozenset({"c4"}),
            embargo_days=5,
            train_start=_utc(2024, 1, 1),
            train_end=_utc(2024, 6, 1),
            validation_end=_utc(2024, 8, 1),
            test_end=_utc(2024, 9, 1),
        )
        assert not verify_cycle_atomicity(fold, {"c1", "c2", "c3", "c4"})

    def test_empty_input(self):
        folds = create_walk_forward_splits([], embargo_days=5)
        assert folds == []

    def test_insufficient_data(self):
        """Too few cycles should produce no folds."""
        cycles = [("c0", _utc(2024, 1, 1), _utc(2024, 1, 5))]
        folds = create_walk_forward_splits(cycles, embargo_days=5)
        assert folds == []
