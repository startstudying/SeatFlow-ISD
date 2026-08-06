"""Canonical expanding walk-forward split construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.domain.time import assert_aware


class CycleDisposition(str, Enum):
    """Per-fold cycle assignment."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BLIND = "blind"
    PURGED = "purged"
    IMMATURE_OUTSIDE = "immature_outside"
    OUTSIDE = "outside"


@dataclass(frozen=True)
class WalkForwardConfig:
    """Frozen walk-forward split parameters expressed in trading days."""

    min_train_trading_days: int = 252
    validation_trading_days: int = 60
    test_trading_days: int = 20
    label_horizon_trading_days: int = 5
    purge_trading_days: int = 5
    embargo_trading_days: int = 5
    blind_trading_days: int = 120

    def __post_init__(self) -> None:
        positive_fields = {
            "min_train_trading_days": self.min_train_trading_days,
            "validation_trading_days": self.validation_trading_days,
            "test_trading_days": self.test_trading_days,
            "label_horizon_trading_days": self.label_horizon_trading_days,
            "purge_trading_days": self.purge_trading_days,
            "embargo_trading_days": self.embargo_trading_days,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.blind_trading_days < 0:
            raise ValueError("blind_trading_days must be non-negative")
        if self.purge_trading_days < self.label_horizon_trading_days:
            raise ValueError(
                "purge_trading_days must be at least the label horizon trading days"
            )
        if self.embargo_trading_days < self.label_horizon_trading_days:
            raise ValueError(
                "embargo_trading_days must be at least the label horizon trading days"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "min_train_trading_days": self.min_train_trading_days,
            "validation_trading_days": self.validation_trading_days,
            "test_trading_days": self.test_trading_days,
            "label_horizon_trading_days": self.label_horizon_trading_days,
            "purge_trading_days": self.purge_trading_days,
            "embargo_trading_days": self.embargo_trading_days,
            "blind_trading_days": self.blind_trading_days,
        }


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding walk-forward fold."""

    fold_id: int
    train_cycle_ids: frozenset[str]
    validation_cycle_ids: frozenset[str]
    test_cycle_ids: frozenset[str]
    blind_cycle_ids: frozenset[str]
    purged_cycle_ids: frozenset[str]
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    purge_trading_days: int
    embargo_trading_days: int
    label_horizon_trading_days: int

    @property
    def embargo_days(self) -> int:
        """Legacy alias for trading-day embargo."""
        return self.embargo_trading_days

    @property
    def prediction_cycle_ids(self) -> frozenset[str]:
        return self.validation_cycle_ids | self.test_cycle_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_cycle_ids": self.train_cycle_ids,
            "validation_cycle_ids": self.validation_cycle_ids,
            "test_cycle_ids": self.test_cycle_ids,
            "blind_cycle_ids": self.blind_cycle_ids,
            "purged_cycle_ids": self.purged_cycle_ids,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "purge_trading_days": self.purge_trading_days,
            "embargo_trading_days": self.embargo_trading_days,
            "label_horizon_trading_days": self.label_horizon_trading_days,
        }


@dataclass(frozen=True)
class WalkForwardPlan:
    """Auditable split plan with deterministic fingerprints."""

    config: WalkForwardConfig
    folds: tuple[WalkForwardFold, ...]
    calendar_fingerprint: str
    calendar_evidence_fingerprint: str
    blind_cycle_ids: frozenset[str]
    classification_by_fold: Mapping[int, Mapping[str, CycleDisposition]]
    fingerprint: str

    def __post_init__(self) -> None:
        immutable = {
            fold_id: MappingProxyType(dict(mapping))
            for fold_id, mapping in self.classification_by_fold.items()
        }
        object.__setattr__(self, "classification_by_fold", MappingProxyType(immutable))

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "folds": [_fold_payload(fold) for fold in self.folds],
            "calendar_fingerprint": self.calendar_fingerprint,
            "calendar_evidence_fingerprint": self.calendar_evidence_fingerprint,
            "blind_cycle_ids": sorted(self.blind_cycle_ids),
            "classification_by_fold": _classification_payload(self.classification_by_fold),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class _CycleRecord:
    cycle_id: str
    stock_id: str
    cycle_start: datetime
    cycle_end: datetime
    label_maturity: datetime | None
    label_maturity_index: int | None
    raw_label_maturity_index: int
    status: str


CycleInput = EventCycle | tuple[str, datetime, datetime]


def create_walk_forward_plan(
    cycles: Sequence[CycleInput],
    *,
    trading_calendar: Sequence[datetime],
    config: WalkForwardConfig | None = None,
    calendar_evidence_fingerprint: str,
) -> WalkForwardPlan:
    """Create a canonical expanding walk-forward split plan.

    All temporal parameters are trading-day counts over the explicit
    timezone-aware trading calendar. The sealed blind tail is excluded before
    fold generation.
    """
    _validate_sha256_hex(calendar_evidence_fingerprint, "calendar_evidence_fingerprint")
    cfg = config or WalkForwardConfig()
    calendar = _canonical_calendar(trading_calendar)
    calendar_index = {day: idx for idx, day in enumerate(calendar)}
    records = _canonical_cycles(cycles, calendar, calendar_index, cfg)
    calendar_fingerprint = _fingerprint({"trading_calendar": [_dt(day) for day in calendar]})
    plan = _generate_plan(
        config=cfg,
        records=records,
        calendar=calendar,
        calendar_index=calendar_index,
        calendar_fingerprint=calendar_fingerprint,
        calendar_evidence_fingerprint=calendar_evidence_fingerprint,
    )
    validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=calendar)
    return plan


def create_walk_forward_splits(
    cycles: Sequence[CycleInput],
    *,
    trading_calendar: Sequence[datetime] | None = None,
    config: WalkForwardConfig | None = None,
    calendar_evidence_fingerprint: str = "0" * 64,
    embargo_days: int | None = None,
    validation_days: int | None = None,
    test_days: int | None = None,
    blind_days: int | None = None,
) -> list[WalkForwardFold]:
    """Compatibility wrapper around :func:`create_walk_forward_plan`.

    Legacy callers must now provide an explicit trading calendar. Silent
    calendar-day single-fold fallback is intentionally not supported.
    """
    if trading_calendar is None:
        raise ValueError("trading_calendar is required for canonical walk-forward splits")
    base_config = WalkForwardConfig()
    cfg = config or WalkForwardConfig(
        min_train_trading_days=base_config.min_train_trading_days,
        validation_trading_days=(
            validation_days
            if validation_days is not None
            else base_config.validation_trading_days
        ),
        test_trading_days=test_days if test_days is not None else base_config.test_trading_days,
        label_horizon_trading_days=base_config.label_horizon_trading_days,
        blind_trading_days=(
            blind_days if blind_days is not None else base_config.blind_trading_days
        ),
        embargo_trading_days=(
            embargo_days if embargo_days is not None else base_config.embargo_trading_days
        ),
        purge_trading_days=(
            embargo_days if embargo_days is not None else base_config.purge_trading_days
        ),
    )
    return list(
        create_walk_forward_plan(
            cycles,
            trading_calendar=trading_calendar,
            config=cfg,
            calendar_evidence_fingerprint=calendar_evidence_fingerprint,
        ).folds
    )


def _generate_plan(
    *,
    config: WalkForwardConfig,
    records: tuple[_CycleRecord, ...],
    calendar: tuple[datetime, ...],
    calendar_index: dict[datetime, int],
    calendar_fingerprint: str,
    calendar_evidence_fingerprint: str,
) -> WalkForwardPlan:
    if not records:
        return _build_plan(
            config=config,
            folds=(),
            calendar_fingerprint=calendar_fingerprint,
            calendar_evidence_fingerprint=calendar_evidence_fingerprint,
            blind_cycle_ids=frozenset(),
            classification_by_fold={},
            cycle_records=records,
        )

    blind_start_idx = len(calendar) if config.blind_trading_days == 0 else max(
        0, len(calendar) - config.blind_trading_days
    )
    blind_cycle_ids = _blind_cycle_ids(records, calendar_index, blind_start_idx)

    fold_boundaries = _fold_boundaries(calendar, config, blind_start_idx)
    folds: list[WalkForwardFold] = []
    classifications: dict[int, dict[str, CycleDisposition]] = {}
    for fold_id, boundary in enumerate(fold_boundaries):
        fold, classification = _build_fold(
            fold_id,
            records,
            calendar_index,
            calendar,
            config,
            boundary,
            blind_cycle_ids,
        )
        if not fold.validation_cycle_ids or not fold.test_cycle_ids or not fold.train_cycle_ids:
            continue
        folds.append(fold)
        classifications[fold_id] = classification

    return _build_plan(
        config=config,
        folds=tuple(folds),
        calendar_fingerprint=calendar_fingerprint,
        calendar_evidence_fingerprint=calendar_evidence_fingerprint,
        blind_cycle_ids=blind_cycle_ids,
        classification_by_fold=classifications,
        cycle_records=records,
    )


def validate_walk_forward_plan(
    plan: WalkForwardPlan,
    *,
    cycles: Sequence[CycleInput],
    trading_calendar: Sequence[datetime],
) -> None:
    """Fail closed on leakage, overlap, calendar tamper, or OOF duplication."""
    _validate_sha256_hex(plan.calendar_evidence_fingerprint, "calendar_evidence_fingerprint")
    calendar = _canonical_calendar(trading_calendar)
    calendar_index = {day: idx for idx, day in enumerate(calendar)}
    expected_calendar_fingerprint = _fingerprint({"trading_calendar": [_dt(day) for day in calendar]})
    if expected_calendar_fingerprint != plan.calendar_fingerprint:
        raise ValueError("calendar fingerprint mismatch")

    records = _canonical_cycles(cycles, calendar, calendar_index, plan.config)
    all_ids = {record.cycle_id for record in records}
    expected = _generate_plan(
        config=plan.config,
        records=records,
        calendar=calendar,
        calendar_index=calendar_index,
        calendar_fingerprint=plan.calendar_fingerprint,
        calendar_evidence_fingerprint=plan.calendar_evidence_fingerprint,
    )
    if plan.blind_cycle_ids != expected.blind_cycle_ids:
        raise ValueError("blind cycle ids mismatch")
    classification_matches_expected = _classification_payload(
        plan.classification_by_fold
    ) == _classification_payload(expected.classification_by_fold)

    prediction_counts: dict[str, int] = {}
    for fold in plan.folds:
        for cycle_id in fold.prediction_cycle_ids:
            prediction_counts[cycle_id] = prediction_counts.get(cycle_id, 0) + 1
    duplicated_predictions = sorted(
        cycle_id for cycle_id, count in prediction_counts.items() if count > 1
    )
    if duplicated_predictions:
        raise ValueError(f"OOF prediction cycle duplicated: {duplicated_predictions}")

    seen_oof: set[str] = set()
    previous_train_size = -1
    previous_validation_start: datetime | None = None
    previous_test_start: datetime | None = None
    for fold in plan.folds:
        train = set(fold.train_cycle_ids)
        val = set(fold.validation_cycle_ids)
        test = set(fold.test_cycle_ids)
        blind = set(fold.blind_cycle_ids)
        purged = set(fold.purged_cycle_ids)
        classification = plan.classification_by_fold.get(fold.fold_id)
        if classification is None:
            raise ValueError("classification missing for fold")
        if set(classification) != all_ids:
            raise ValueError("classification does not exactly account for all cycles")
        prediction = val | test
        if previous_train_size > len(train):
            raise ValueError("train set is not monotonic expanding")
        if previous_validation_start is not None and fold.validation_start <= previous_validation_start:
            raise ValueError("validation boundaries do not move forward")
        if previous_test_start is not None and fold.test_start <= previous_test_start:
            raise ValueError("test boundaries do not move forward")
        previous_train_size = len(train)
        previous_validation_start = fold.validation_start
        previous_test_start = fold.test_start
        if train & prediction:
            raise ValueError("training set contains prediction cycle")
        if val & test:
            raise ValueError("validation/test overlap")
        if blind & (train | val | test):
            raise ValueError("blind cycle appears in development fold")
        if purged & (train | val | test | blind):
            raise ValueError("purged cycle also assigned to a concrete segment")
        if not (train | val | test | blind | purged).issubset(all_ids):
            raise ValueError("fold contains unknown cycle id")
        duplicate_oof = seen_oof & prediction
        if duplicate_oof:
            raise ValueError(f"OOF prediction cycle duplicated: {sorted(duplicate_oof)}")
        if not verify_cycle_atomicity(fold, all_ids, classification):
            if not classification_matches_expected:
                raise ValueError("classification differs from canonical regeneration")
            raise ValueError("cycle atomicity/accounting violation")
        seen_oof.update(prediction)
        _validate_embargo(fold, calendar_index, plan.config)

    if not classification_matches_expected:
        raise ValueError("classification differs from canonical regeneration")
    if tuple(_fold_payload(fold) for fold in plan.folds) != tuple(
        _fold_payload(fold) for fold in expected.folds
    ):
        raise ValueError("fold boundaries or assignments differ from canonical regeneration")
    if expected.fingerprint != plan.fingerprint:
        raise ValueError("split fingerprint mismatch")


def verify_embargo(fold: WalkForwardFold, trading_calendar: Sequence[datetime]) -> bool:
    """Legacy boolean embargo verifier."""
    try:
        calendar = _canonical_calendar(trading_calendar)
        index = {day: idx for idx, day in enumerate(calendar)}
        _validate_embargo(
            fold,
            index,
            WalkForwardConfig(
                label_horizon_trading_days=fold.label_horizon_trading_days,
                purge_trading_days=fold.purge_trading_days,
                embargo_trading_days=fold.embargo_trading_days,
            ),
        )
    except ValueError:
        return False
    return True


def verify_cycle_atomicity(
    fold: WalkForwardFold,
    all_cycle_ids: set[str],
    classification: Mapping[str, CycleDisposition] | None = None,
) -> bool:
    """Legacy boolean cycle-overlap verifier."""
    train = set(fold.train_cycle_ids)
    val = set(fold.validation_cycle_ids)
    test = set(fold.test_cycle_ids)
    blind = set(fold.blind_cycle_ids)
    purged = set(fold.purged_cycle_ids)
    immature = (
        {cid for cid, disp in classification.items() if disp is CycleDisposition.IMMATURE_OUTSIDE}
        if classification is not None
        else set()
    )
    outside = (
        {cid for cid, disp in classification.items() if disp is CycleDisposition.OUTSIDE}
        if classification is not None
        else set()
    )
    segments = [train, val, test, blind, purged, immature, outside]
    for left_idx, left in enumerate(segments):
        for right in segments[left_idx + 1 :]:
            if left & right:
                return False
    accounted = set().union(*segments)
    if classification is not None and set(classification) != all_cycle_ids:
        return False
    return accounted == all_cycle_ids


def _canonical_calendar(trading_calendar: Sequence[datetime]) -> tuple[datetime, ...]:
    if not trading_calendar:
        raise ValueError("trading_calendar is missing")
    for day in trading_calendar:
        assert_aware(day, "trading_calendar")
    unique = set(trading_calendar)
    if len(unique) != len(trading_calendar):
        raise ValueError("trading_calendar contains duplicate dates")
    return tuple(sorted(trading_calendar))


def _validate_sha256_hex(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a 64-character sha256 hex fingerprint")


def _canonical_cycles(
    cycles: Sequence[CycleInput],
    calendar: tuple[datetime, ...],
    calendar_index: dict[datetime, int],
    config: WalkForwardConfig,
) -> tuple[_CycleRecord, ...]:
    records = [_coerce_cycle(cycle, calendar, calendar_index, config) for cycle in cycles]
    cycle_ids = [record.cycle_id for record in records]
    if len(cycle_ids) != len(set(cycle_ids)):
        raise ValueError("cycles contain duplicate cycle_id")
    return tuple(sorted(records, key=lambda record: (record.cycle_start, record.cycle_id)))


def _coerce_cycle(
    cycle: CycleInput,
    calendar: tuple[datetime, ...],
    calendar_index: dict[datetime, int],
    config: WalkForwardConfig,
) -> _CycleRecord:
    if isinstance(cycle, tuple):
        cycle_id, start, end = cycle
        stock_id = ""
        status = CycleStatus.EXPIRED.value
    else:
        cycle_id = cycle.cycle_id
        stock_id = cycle.stock_id
        if cycle.status is CycleStatus.ACTIVE or cycle.cycle_end is None:
            raise ValueError(f"open cycle cannot be split: {cycle_id}")
        start = cycle.cycle_start
        end = cycle.cycle_end
        status = cycle.status.value if isinstance(cycle.status, CycleStatus) else str(cycle.status)
    assert_aware(start, "cycle_start")
    assert_aware(end, "cycle_end")
    if start not in calendar_index:
        raise ValueError(f"calendar missing cycle_start for {cycle_id}")
    if end not in calendar_index:
        raise ValueError(f"calendar missing cycle_end for {cycle_id}")
    if calendar_index[end] < calendar_index[start]:
        raise ValueError(f"cycle_end precedes cycle_start for {cycle_id}")
    maturity_index = calendar_index[end] + config.label_horizon_trading_days
    if maturity_index >= len(calendar):
        maturity = None
        maturity_record_index = None
    else:
        maturity = calendar[maturity_index]
        maturity_record_index = maturity_index
    return _CycleRecord(
        cycle_id=cycle_id,
        stock_id=stock_id,
        cycle_start=start,
        cycle_end=end,
        label_maturity=maturity,
        label_maturity_index=maturity_record_index,
        raw_label_maturity_index=maturity_index,
        status=status,
    )


def _blind_cycle_ids(
    records: tuple[_CycleRecord, ...],
    calendar_index: dict[datetime, int],
    blind_start_idx: int,
) -> frozenset[str]:
    if blind_start_idx >= len(calendar_index):
        return frozenset()
    ids: set[str] = set()
    for record in records:
        start_idx = calendar_index[record.cycle_start]
        end_idx = calendar_index[record.cycle_end]
        if start_idx >= blind_start_idx or end_idx >= blind_start_idx:
            ids.add(record.cycle_id)
        elif record.raw_label_maturity_index >= blind_start_idx:
            ids.add(record.cycle_id)
    return frozenset(sorted(ids))


def _fold_boundaries(
    calendar: tuple[datetime, ...],
    config: WalkForwardConfig,
    blind_start_idx: int,
) -> list[tuple[int, int, int, int, int, int]]:
    boundaries: list[tuple[int, int, int, int, int, int]] = []
    gap = max(config.purge_trading_days, config.embargo_trading_days)
    train_start_idx = 0
    train_end_idx = config.min_train_trading_days - 1
    while True:
        validation_start_idx = train_end_idx + gap + 1
        validation_end_idx = validation_start_idx + config.validation_trading_days - 1
        test_start_idx = validation_end_idx + gap + 1
        test_end_idx = test_start_idx + config.test_trading_days - 1
        if test_end_idx >= blind_start_idx:
            break
        boundaries.append(
            (
                train_start_idx,
                train_end_idx,
                validation_start_idx,
                validation_end_idx,
                test_start_idx,
                test_end_idx,
            )
        )
        train_end_idx = test_end_idx
    return boundaries


def _build_fold(
    fold_id: int,
    records: tuple[_CycleRecord, ...],
    calendar_index: dict[datetime, int],
    calendar: tuple[datetime, ...],
    config: WalkForwardConfig,
    boundary: tuple[int, int, int, int, int, int],
    blind_cycle_ids: frozenset[str],
) -> tuple[WalkForwardFold, dict[str, CycleDisposition]]:
    (
        train_start_idx,
        train_end_idx,
        validation_start_idx,
        validation_end_idx,
        test_start_idx,
        test_end_idx,
    ) = boundary
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    purged_ids: set[str] = set()
    classification: dict[str, CycleDisposition] = {}

    for record in records:
        start_idx = calendar_index[record.cycle_start]
        end_idx = calendar_index[record.cycle_end]
        maturity_idx = record.label_maturity_index
        if record.cycle_id in blind_cycle_ids:
            classification[record.cycle_id] = CycleDisposition.BLIND
        elif maturity_idx is None:
            classification[record.cycle_id] = CycleDisposition.IMMATURE_OUTSIDE
        elif start_idx >= train_start_idx and maturity_idx <= train_end_idx:
            train_ids.add(record.cycle_id)
            classification[record.cycle_id] = CycleDisposition.TRAIN
        elif start_idx >= validation_start_idx and end_idx <= validation_end_idx:
            validation_ids.add(record.cycle_id)
            classification[record.cycle_id] = CycleDisposition.VALIDATION
        elif start_idx >= test_start_idx and end_idx <= test_end_idx:
            test_ids.add(record.cycle_id)
            classification[record.cycle_id] = CycleDisposition.TEST
        elif _touches_guard_or_prediction(
            start_idx,
            end_idx,
            maturity_idx,
            train_end_idx,
            validation_start_idx,
            validation_end_idx,
            test_start_idx,
            test_end_idx,
        ):
            purged_ids.add(record.cycle_id)
            classification[record.cycle_id] = CycleDisposition.PURGED
        else:
            classification[record.cycle_id] = CycleDisposition.OUTSIDE

    fold = WalkForwardFold(
        fold_id=fold_id,
        train_cycle_ids=frozenset(sorted(train_ids)),
        validation_cycle_ids=frozenset(sorted(validation_ids)),
        test_cycle_ids=frozenset(sorted(test_ids)),
        blind_cycle_ids=blind_cycle_ids,
        purged_cycle_ids=frozenset(sorted(purged_ids)),
        train_start=calendar[train_start_idx],
        train_end=calendar[train_end_idx],
        validation_start=calendar[validation_start_idx],
        validation_end=calendar[validation_end_idx],
        test_start=calendar[test_start_idx],
        test_end=calendar[test_end_idx],
        purge_trading_days=config.purge_trading_days,
        embargo_trading_days=config.embargo_trading_days,
        label_horizon_trading_days=config.label_horizon_trading_days,
    )
    return fold, classification


def _touches_guard_or_prediction(
    start_idx: int,
    end_idx: int,
    maturity_idx: int,
    train_end_idx: int,
    validation_start_idx: int,
    validation_end_idx: int,
    test_start_idx: int,
    test_end_idx: int,
) -> bool:
    if train_end_idx < maturity_idx and start_idx <= test_end_idx:
        return True
    if start_idx <= validation_end_idx and maturity_idx >= validation_start_idx:
        return True
    return start_idx <= test_end_idx and maturity_idx >= test_start_idx


def _validate_embargo(
    fold: WalkForwardFold,
    calendar_index: dict[datetime, int],
    config: WalkForwardConfig,
) -> None:
    try:
        train_end = calendar_index[fold.train_end]
        validation_start = calendar_index[fold.validation_start]
        validation_end = calendar_index[fold.validation_end]
        test_start = calendar_index[fold.test_start]
    except KeyError as exc:
        raise ValueError("fold boundary is missing from trading calendar") from exc
    train_val_gap = validation_start - train_end - 1
    val_test_gap = test_start - validation_end - 1
    required = max(config.purge_trading_days, config.embargo_trading_days)
    if train_val_gap < required or val_test_gap < required:
        raise ValueError("embargo/purge shortfall between fold windows")


def _build_plan(
    *,
    config: WalkForwardConfig,
    folds: tuple[WalkForwardFold, ...],
    calendar_fingerprint: str,
    calendar_evidence_fingerprint: str,
    blind_cycle_ids: frozenset[str],
    classification_by_fold: dict[int, dict[str, CycleDisposition]],
    cycle_records: tuple[_CycleRecord, ...],
) -> WalkForwardPlan:
    payload = {
        "config": config.as_dict(),
        "calendar_fingerprint": calendar_fingerprint,
        "calendar_evidence_fingerprint": calendar_evidence_fingerprint,
        "cycle_records": [
            {
                "cycle_id": record.cycle_id,
                "stock_id": record.stock_id,
                "cycle_start": _dt(record.cycle_start),
                "cycle_end": _dt(record.cycle_end),
                "label_maturity": _dt(record.label_maturity)
                if record.label_maturity is not None
                else None,
                "label_maturity_index": record.label_maturity_index,
                "raw_label_maturity_index": record.raw_label_maturity_index,
                "status": record.status,
            }
            for record in cycle_records
        ],
        "folds": [_fold_payload(fold) for fold in folds],
        "blind_cycle_ids": sorted(blind_cycle_ids),
        "classification_by_fold": _classification_payload(classification_by_fold),
    }
    return WalkForwardPlan(
        config=config,
        folds=folds,
        calendar_fingerprint=calendar_fingerprint,
        calendar_evidence_fingerprint=calendar_evidence_fingerprint,
        blind_cycle_ids=blind_cycle_ids,
        classification_by_fold=classification_by_fold,
        fingerprint=_fingerprint(payload),
    )


def _fold_payload(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "train_cycle_ids": sorted(fold.train_cycle_ids),
        "validation_cycle_ids": sorted(fold.validation_cycle_ids),
        "test_cycle_ids": sorted(fold.test_cycle_ids),
        "blind_cycle_ids": sorted(fold.blind_cycle_ids),
        "purged_cycle_ids": sorted(fold.purged_cycle_ids),
        "train_start": _dt(fold.train_start),
        "train_end": _dt(fold.train_end),
        "validation_start": _dt(fold.validation_start),
        "validation_end": _dt(fold.validation_end),
        "test_start": _dt(fold.test_start),
        "test_end": _dt(fold.test_end),
        "purge_trading_days": fold.purge_trading_days,
        "embargo_trading_days": fold.embargo_trading_days,
        "label_horizon_trading_days": fold.label_horizon_trading_days,
    }


def _classification_payload(
    classification_by_fold: Mapping[int, Mapping[str, CycleDisposition]],
) -> dict[str, dict[str, str]]:
    return {
        str(fid): {cid: disp.value for cid, disp in sorted(mapping.items())}
        for fid, mapping in sorted(classification_by_fold.items())
    }


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dt(value: datetime) -> str:
    return value.isoformat()
