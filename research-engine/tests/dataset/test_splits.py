"""Canonical expanding walk-forward split contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import MutableMapping, cast

import pytest

from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.dataset.splits import (
    CycleDisposition,
    WalkForwardConfig,
    WalkForwardPlan,
    create_walk_forward_plan,
    create_walk_forward_splits,
    validate_walk_forward_plan,
    verify_cycle_atomicity,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _trading_calendar(
    start: datetime = _utc(2026, 1, 2),
    *,
    days: int = 180,
    holidays: set[datetime] | None = None,
) -> list[datetime]:
    holidays = holidays or set()
    out: list[datetime] = []
    cursor = start
    while len(out) < days:
        if cursor.weekday() < 5 and cursor not in holidays:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _cycle(cycle_id: str, day: datetime, *, span: int = 0) -> EventCycle:
    return EventCycle(
        cycle_id=cycle_id,
        stock_id="000001.SZ",
        cycle_start=day,
        last_event_date=day + timedelta(days=span),
        cycle_end=day + timedelta(days=span),
        event_count=1,
        status=CycleStatus.EXPIRED,
    )


def _cycles(calendar: list[datetime], start_index: int = 0, stop_index: int = 150) -> list[EventCycle]:
    return [_cycle(f"cyc-{i:03d}", calendar[i]) for i in range(start_index, stop_index)]


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


def test_expanding_plan_has_three_or_more_monotonic_folds_and_unique_oof_segments() -> None:
    calendar = _trading_calendar(days=180)
    cycles = _cycles(calendar, stop_index=150)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )

    assert len(plan.folds) >= 3
    train_sizes = [len(f.train_cycle_ids) for f in plan.folds]
    assert train_sizes == sorted(train_sizes)

    previous_prediction_ids: set[str] = set()
    previous_val_start: datetime | None = None
    previous_test_start: datetime | None = None
    for fold in plan.folds:
        assert previous_prediction_ids.isdisjoint(fold.prediction_cycle_ids)
        previous_prediction_ids.update(fold.prediction_cycle_ids)
        if previous_val_start is not None and previous_test_start is not None:
            assert fold.validation_start > previous_val_start
            assert fold.test_start > previous_test_start
        previous_val_start = fold.validation_start
        previous_test_start = fold.test_start

    validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=calendar)


def test_exact_classification_keeps_cycles_atomic_and_blind_isolated() -> None:
    calendar = _trading_calendar(days=100)
    cycles = _cycles(calendar, stop_index=90)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )

    first = plan.folds[0]
    classification = plan.classification_by_fold[first.fold_id]
    for cycle in cycles:
        assert cycle.cycle_id in classification
        assigned_segments = [
            cycle.cycle_id in first.train_cycle_ids,
            cycle.cycle_id in first.validation_cycle_ids,
            cycle.cycle_id in first.test_cycle_ids,
            cycle.cycle_id in first.blind_cycle_ids,
            classification[cycle.cycle_id] == CycleDisposition.PURGED,
            classification[cycle.cycle_id] == CycleDisposition.OUTSIDE,
        ]
        assert sum(1 for assigned in assigned_segments if assigned) == 1

    assert first.blind_cycle_ids == plan.blind_cycle_ids
    assert first.blind_cycle_ids.isdisjoint(first.train_cycle_ids)
    assert first.blind_cycle_ids.isdisjoint(first.validation_cycle_ids)
    assert first.blind_cycle_ids.isdisjoint(first.test_cycle_ids)


def test_embargo_uses_trading_days_across_weekend_and_holiday() -> None:
    holiday = _utc(2026, 2, 16)
    calendar = _trading_calendar(days=120, holidays={holiday})
    cycles = _cycles(calendar, stop_index=90)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )

    saw_holiday_boundary = False
    for fold in plan.folds:
        between_train_val = [d for d in calendar if fold.train_end < d < fold.validation_start]
        between_val_test = [d for d in calendar if fold.validation_end < d < fold.test_start]
        assert len(between_train_val) >= 5
        assert len(between_val_test) >= 5
        assert holiday not in between_train_val
        assert holiday not in between_val_test
        saw_holiday_boundary = saw_holiday_boundary or (
            fold.train_end < holiday < fold.validation_start
            or fold.validation_end < holiday < fold.test_start
        )
    assert saw_holiday_boundary


def test_long_cycle_or_label_maturity_overlap_is_purged_from_prediction_segment() -> None:
    calendar = _trading_calendar(days=120)
    cycles = _cycles(calendar, stop_index=90)
    cycles.append(_cycle("long-cycle", calendar[34], span=(calendar[50] - calendar[34]).days))

    plan = create_walk_forward_plan(
        list(reversed(cycles)),
        trading_calendar=list(reversed(calendar)),
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )

    first = plan.folds[0]
    assert "long-cycle" not in first.train_cycle_ids
    assert "long-cycle" not in first.validation_cycle_ids
    assert "long-cycle" not in first.test_cycle_ids
    assert plan.classification_by_fold[first.fold_id]["long-cycle"] == CycleDisposition.PURGED
    validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=calendar)


def test_input_order_fingerprint_determinism_and_tamper_sensitivity() -> None:
    calendar = _trading_calendar(days=140)
    cycles = _cycles(calendar, stop_index=110)
    config = _config()

    plan_a = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )
    plan_b = create_walk_forward_plan(
        list(reversed(cycles)),
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )
    assert plan_a.fingerprint == plan_b.fingerprint

    assert (
        create_walk_forward_plan(
            cycles,
            trading_calendar=calendar,
            config=config,
            calendar_evidence_fingerprint=HASH_B,
        ).fingerprint
        != plan_a.fingerprint
    )
    changed_config = WalkForwardConfig(**{**_config().as_dict(), "test_trading_days": 6})
    assert (
        create_walk_forward_plan(
            cycles,
            trading_calendar=calendar,
            config=changed_config,
            calendar_evidence_fingerprint=HASH_A,
        ).fingerprint
        != plan_a.fingerprint
    )
    tampered = [*cycles[:-1], _cycle(cycles[-1].cycle_id, calendar[111])]
    assert (
        create_walk_forward_plan(
            tampered,
            trading_calendar=calendar,
            config=config,
            calendar_evidence_fingerprint=HASH_A,
        ).fingerprint
        != plan_a.fingerprint
    )


def test_calendar_validation_is_fail_closed_for_naive_missing_and_duplicate_dates() -> None:
    calendar = _trading_calendar(days=80)
    cycles = _cycles(calendar, stop_index=50)
    config = _config()

    with pytest.raises(ValueError, match="timezone-aware"):
        create_walk_forward_plan(
            cycles,
            trading_calendar=[d.replace(tzinfo=None) for d in calendar],
            config=config,
            calendar_evidence_fingerprint=HASH_A,
        )

    with pytest.raises(ValueError, match="duplicate"):
        create_walk_forward_plan(
            cycles,
            trading_calendar=[*calendar, calendar[0]],
            config=config,
            calendar_evidence_fingerprint=HASH_A,
        )

    with pytest.raises(ValueError, match="missing"):
        create_walk_forward_plan(
            cycles,
            trading_calendar=[d for d in calendar if d != cycles[10].cycle_start],
            config=config,
            calendar_evidence_fingerprint=HASH_A,
        )


def test_unsorted_calendar_canonicalizes_to_same_fingerprint() -> None:
    calendar = _trading_calendar(days=140)
    cycles = _cycles(calendar, stop_index=110)
    plan_a = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )
    plan_b = create_walk_forward_plan(
        cycles,
        trading_calendar=list(reversed(calendar)),
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )
    assert plan_a.calendar_fingerprint == plan_b.calendar_fingerprint
    assert plan_a.fingerprint == plan_b.fingerprint


def test_insufficient_data_returns_empty_plan_and_invalid_config_fails() -> None:
    calendar = _trading_calendar(days=40)
    empty = create_walk_forward_plan(
        _cycles(calendar, stop_index=10),
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )
    assert empty.folds == ()

    with pytest.raises(ValueError, match="label horizon"):
        WalkForwardConfig(label_horizon_trading_days=5, purge_trading_days=4)


def test_legacy_splits_wrapper_uses_canonical_plan() -> None:
    calendar = _trading_calendar(days=140)
    cycles = _cycles(calendar, stop_index=110)
    config = _config()
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert create_walk_forward_splits(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    ) == list(plan.folds)


def test_label_maturity_beyond_calendar_is_immature_outside_not_clamped() -> None:
    calendar = _trading_calendar(days=80)
    cycles = [*_cycles(calendar, stop_index=70), _cycle("tail-cycle", calendar[-3])]
    config = WalkForwardConfig(
        min_train_trading_days=10,
        validation_trading_days=5,
        test_trading_days=5,
        label_horizon_trading_days=5,
        purge_trading_days=5,
        embargo_trading_days=5,
        blind_trading_days=0,
    )

    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert plan.classification_by_fold[0]["tail-cycle"] == CycleDisposition.IMMATURE_OUTSIDE
    assert "tail-cycle" not in plan.folds[0].blind_cycle_ids


def test_zero_blind_keeps_last_day_immature_cycle_out_of_blind() -> None:
    calendar = _trading_calendar(days=80)
    cycles = [*_cycles(calendar, stop_index=70), _cycle("last-day-cycle", calendar[-1])]
    config = WalkForwardConfig(
        min_train_trading_days=10,
        validation_trading_days=5,
        test_trading_days=5,
        label_horizon_trading_days=5,
        purge_trading_days=5,
        embargo_trading_days=5,
        blind_trading_days=0,
    )

    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert plan.blind_cycle_ids == frozenset()
    assert all(fold.blind_cycle_ids == frozenset() for fold in plan.folds)
    assert plan.classification_by_fold[0]["last-day-cycle"] == CycleDisposition.IMMATURE_OUTSIDE


def test_raw_maturity_index_beyond_calendar_still_marks_blind_tail_touch() -> None:
    calendar = _trading_calendar(days=40)
    config = WalkForwardConfig(
        min_train_trading_days=5,
        validation_trading_days=2,
        test_trading_days=2,
        label_horizon_trading_days=10,
        purge_trading_days=10,
        embargo_trading_days=10,
        blind_trading_days=5,
    )
    cycle = _cycle("raw-maturity-crosses-blind", calendar[-config.blind_trading_days - 5])

    plan = create_walk_forward_plan(
        [*_cycles(calendar, stop_index=20), cycle],
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert "raw-maturity-crosses-blind" in plan.blind_cycle_ids


def test_blind_covers_cycle_start_end_or_label_maturity_touching_blind_tail() -> None:
    calendar = _trading_calendar(days=100)
    config = _config()
    blind_start = calendar[-config.blind_trading_days]
    cycles = [
        _cycle("start-blind", blind_start),
        _cycle("end-blind", calendar[-config.blind_trading_days - 2], span=7),
        _cycle("maturity-blind", calendar[-config.blind_trading_days - 4]),
        *_cycles(calendar, stop_index=70),
    ]

    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=config,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert {"start-blind", "end-blind", "maturity-blind"}.issubset(plan.blind_cycle_ids)
    for fold in plan.folds:
        assert {"start-blind", "end-blind", "maturity-blind"}.issubset(fold.blind_cycle_ids)
        assert fold.blind_cycle_ids.isdisjoint(fold.train_cycle_ids)
        assert fold.blind_cycle_ids.isdisjoint(fold.prediction_cycle_ids)


def test_calendar_evidence_fingerprint_must_be_sha256_hex() -> None:
    calendar = _trading_calendar(days=80)
    with pytest.raises(ValueError, match="calendar_evidence_fingerprint"):
        create_walk_forward_plan(
            _cycles(calendar, stop_index=50),
            trading_calendar=calendar,
            config=_config(),
            calendar_evidence_fingerprint="calendar-source-a",
        )


def test_plan_is_deeply_immutable_and_validator_regenerates_boundaries() -> None:
    calendar = _trading_calendar(days=140)
    cycles = _cycles(calendar, stop_index=110)
    plan = create_walk_forward_plan(
        cycles,
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )

    mutable_classification = cast(
        MutableMapping[str, CycleDisposition],
        plan.classification_by_fold[plan.folds[0].fold_id],
    )
    with pytest.raises(TypeError):
        mutable_classification["cyc-000"] = CycleDisposition.TEST

    mutated_classification = {
        fid: dict(mapping) for fid, mapping in plan.classification_by_fold.items()
    }
    mutated_classification[plan.folds[0].fold_id]["cyc-000"] = CycleDisposition.OUTSIDE
    bad_plan = WalkForwardPlan(
        config=plan.config,
        folds=plan.folds,
        calendar_fingerprint=plan.calendar_fingerprint,
        calendar_evidence_fingerprint=plan.calendar_evidence_fingerprint,
        blind_cycle_ids=plan.blind_cycle_ids,
        classification_by_fold=mutated_classification,
        fingerprint=plan.fingerprint,
    )
    with pytest.raises(ValueError, match="classification"):
        validate_walk_forward_plan(bad_plan, cycles=cycles, trading_calendar=calendar)

    moved_fold = replace(plan.folds[0], validation_start=calendar[40])
    bad_boundary_plan = WalkForwardPlan(
        config=plan.config,
        folds=(moved_fold, *plan.folds[1:]),
        calendar_fingerprint=plan.calendar_fingerprint,
        calendar_evidence_fingerprint=plan.calendar_evidence_fingerprint,
        blind_cycle_ids=plan.blind_cycle_ids,
        classification_by_fold=plan.classification_by_fold,
        fingerprint=plan.fingerprint,
    )
    with pytest.raises(ValueError, match="boundaries"):
        validate_walk_forward_plan(bad_boundary_plan, cycles=cycles, trading_calendar=calendar)


def test_verify_cycle_atomicity_requires_exact_accounting_with_outside_classification() -> None:
    calendar = _trading_calendar(days=100)
    plan = create_walk_forward_plan(
        _cycles(calendar, stop_index=80),
        trading_calendar=calendar,
        config=_config(),
        calendar_evidence_fingerprint=HASH_A,
    )
    fold = plan.folds[0]
    all_ids = set(plan.classification_by_fold[fold.fold_id])

    assert verify_cycle_atomicity(fold, all_ids, plan.classification_by_fold[fold.fold_id])
    assert not verify_cycle_atomicity(fold, all_ids | {"unknown"}, plan.classification_by_fold[fold.fold_id])


def test_legacy_wrapper_preserves_explicit_zero_blind_days() -> None:
    calendar = _trading_calendar(days=500)
    cycles = _cycles(calendar, stop_index=450)

    folds = create_walk_forward_splits(
        cycles,
        trading_calendar=calendar,
        blind_days=0,
        calendar_evidence_fingerprint=HASH_A,
    )

    assert folds
    assert all(not fold.blind_cycle_ids for fold in folds)


def test_open_cycle_and_intraday_boundary_fail_closed() -> None:
    calendar = _trading_calendar(days=80)
    open_cycle = EventCycle(
        cycle_id="open-cycle",
        stock_id="000001.SZ",
        cycle_start=calendar[10],
        last_event_date=calendar[10],
        cycle_end=None,
        event_count=1,
        status=CycleStatus.ACTIVE,
    )
    with pytest.raises(ValueError, match="open cycle"):
        create_walk_forward_plan(
            [open_cycle],
            trading_calendar=calendar,
            config=_config(),
            calendar_evidence_fingerprint=HASH_A,
        )

    intraday = _cycle("intraday", calendar[10].replace(hour=10))
    with pytest.raises(ValueError, match="calendar missing cycle_start"):
        create_walk_forward_plan(
            [intraday],
            trading_calendar=calendar,
            config=_config(),
            calendar_evidence_fingerprint=HASH_A,
        )
