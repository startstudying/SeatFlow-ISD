"""Compatibility tests for the canonical walk-forward contract."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.splits import (
    WalkForwardConfig as DatasetWalkForwardConfig,
    WalkForwardFold as DatasetWalkForwardFold,
    create_walk_forward_plan as dataset_create_walk_forward_plan,
)
from seatflow.evaluation.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    create_walk_forward_plan,
    create_walk_forward_splits,
    validate_walk_forward_plan,
)


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _calendar(days: int = 150) -> list[datetime]:
    out: list[datetime] = []
    cursor = _utc(2026, 1, 2)
    while len(out) < days:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _cycle(cycle_id: str, day: datetime) -> EventCycle:
    return EventCycle(
        cycle_id=cycle_id,
        stock_id="000001.SZ",
        cycle_start=day,
        last_event_date=day,
        cycle_end=day,
        event_count=1,
        status=CycleStatus.EXPIRED,
    )


def _cycles(calendar: list[datetime], count: int = 110) -> list[EventCycle]:
    return [_cycle(f"cyc-{i:03d}", calendar[i]) for i in range(count)]


def _config() -> WalkForwardConfig:
    return WalkForwardConfig(
        min_train_trading_days=30,
        validation_trading_days=8,
        test_trading_days=5,
        label_horizon_trading_days=5,
        purge_trading_days=5,
        embargo_trading_days=5,
        blind_trading_days=20,
    )


def test_evaluation_imports_are_same_canonical_symbols_as_dataset() -> None:
    assert WalkForwardConfig is DatasetWalkForwardConfig
    assert WalkForwardFold is DatasetWalkForwardFold
    assert create_walk_forward_plan is dataset_create_walk_forward_plan


def test_legacy_tuple_api_requires_explicit_trading_calendar() -> None:
    calendar = _calendar()
    cycles = [(f"cyc-{i:03d}", calendar[i], calendar[i]) for i in range(60)]

    with pytest.raises(ValueError, match="trading_calendar"):
        create_walk_forward_splits(cycles, config=_config())


def test_legacy_tuple_api_uses_canonical_multi_fold_logic_when_calendar_is_explicit() -> None:
    calendar = _calendar()
    tuple_cycles = [(f"cyc-{i:03d}", calendar[i], calendar[i]) for i in range(110)]

    folds = create_walk_forward_splits(
        tuple_cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint="calendar-source-a",
    )

    assert len(folds) >= 3
    assert all(isinstance(fold, DatasetWalkForwardFold) for fold in folds)


def test_validator_rejects_train_prediction_overlap_and_oof_duplicates() -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint="calendar-source-a",
    )

    first = plan.folds[0]
    bad_fold = replace(
        first,
        train_cycle_ids=frozenset([*first.train_cycle_ids, next(iter(first.test_cycle_ids))]),
    )
    bad_plan = replace(
        plan,
        folds=(bad_fold, *plan.folds[1:]),
    )
    with pytest.raises(ValueError, match="training.*prediction"):
        validate_walk_forward_plan(bad_plan, cycles=cycles, trading_calendar=calendar)

    if len(plan.folds) < 2:
        pytest.fail("fixture must produce at least two folds")
    second = plan.folds[1]
    duplicated_first = replace(
        first,
        validation_cycle_ids=frozenset(
            [*first.validation_cycle_ids, next(iter(second.validation_cycle_ids))]
        ),
    )
    duplicate_plan = replace(
        plan,
        folds=(duplicated_first, *plan.folds[1:]),
    )
    with pytest.raises(ValueError, match="OOF"):
        validate_walk_forward_plan(duplicate_plan, cycles=cycles, trading_calendar=calendar)


def test_validator_rejects_calendar_tamper_and_embargo_shortfall() -> None:
    calendar = _calendar()
    cycles = _cycles(calendar)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint="calendar-source-a",
    )

    with pytest.raises(ValueError, match="calendar fingerprint"):
        validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=calendar[:-1])

    first = plan.folds[0]
    bad_fold = replace(
        first,
        validation_start=calendar[calendar.index(first.train_end) + 2],
    )
    bad_plan = replace(
        plan,
        folds=(bad_fold, *plan.folds[1:]),
    )
    with pytest.raises(ValueError, match="embargo"):
        validate_walk_forward_plan(bad_plan, cycles=cycles, trading_calendar=calendar)
