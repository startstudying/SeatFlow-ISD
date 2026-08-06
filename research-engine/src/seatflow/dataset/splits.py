"""Canonical expanding walk-forward split construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

from seatflow.dataset.cycle import CycleStatus, EventCycle
from seatflow.domain.time import assert_aware


class CycleDisposition(str, Enum):
    """Per-fold cycle assignment."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BLIND = "blind"
    PURGED = "purged"
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
    classification_by_fold: dict[int, dict[str, CycleDisposition]]
    fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "folds": [_fold_payload(fold) for fold in self.folds],
            "calendar_fingerprint": self.calendar_fingerprint,
            "calendar_evidence_fingerprint": self.calendar_evidence_fingerprint,
            "blind_cycle_ids": sorted(self.blind_cycle_ids),
            "classification_by_fold": {
                str(fid): {cid: disp.value for cid, disp in sorted(mapping.items())}
                for fid, mapping in sorted(self.classification_by_fold.items())
            },
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class _CycleRecord:
    cycle_id: str
    stock_id: str
    cycle_start: datetime
    cycle_end: datetime
    label_maturity: datetime
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
    if not calendar_evidence_fingerprint:
        raise ValueError("calendar_evidence_fingerprint is required")
    cfg = config or WalkForwardConfig()
    calendar = _canonical_calendar(trading_calendar)
    calendar_index = {day: idx for idx, day in enumerate(calendar)}
    records = _canonical_cycles(cycles, calendar, calendar_index, cfg)
    calendar_fingerprint = _fingerprint({"trading_calendar": [_dt(day) for day in calendar]})

    if not records:
        return _build_plan(
            config=cfg,
            folds=(),
            calendar_fingerprint=calendar_fingerprint,
            calendar_evidence_fingerprint=calendar_evidence_fingerprint,
            blind_cycle_ids=frozenset(),
            classification_by_fold={},
            cycle_records=records,
        )

    blind_start_idx = max(0, len(calendar) - cfg.blind_trading_days)
    blind_start = calendar[blind_start_idx] if cfg.blind_trading_days else None
    blind_cycle_ids = frozenset(
        record.cycle_id
        for record in records
        if blind_start is not None and record.cycle_start >= blind_start
    )

    fold_boundaries = _fold_boundaries(calendar, cfg, blind_start_idx)
    folds: list[WalkForwardFold] = []
    classifications: dict[int, dict[str, CycleDisposition]] = {}
    for fold_id, boundary in enumerate(fold_boundaries):
        fold, classification = _build_fold(
            fold_id,
            records,
            calendar_index,
            calendar,
            cfg,
            boundary,
            blind_cycle_ids,
        )
        if not fold.validation_cycle_ids or not fold.test_cycle_ids or not fold.train_cycle_ids:
            continue
        folds.append(fold)
        classifications[fold_id] = classification

    plan = _build_plan(
        config=cfg,
        folds=tuple(folds),
        calendar_fingerprint=calendar_fingerprint,
        calendar_evidence_fingerprint=calendar_evidence_fingerprint,
        blind_cycle_ids=blind_cycle_ids,
        classification_by_fold=classifications,
        cycle_records=records,
    )
    validate_walk_forward_plan(plan, cycles=cycles, trading_calendar=calendar)
    return plan


def create_walk_forward_splits(
    cycles: Sequence[CycleInput],
    *,
    trading_calendar: Sequence[datetime] | None = None,
    config: WalkForwardConfig | None = None,
    calendar_evidence_fingerprint: str = "legacy-explicit-calendar",
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
        validation_trading_days=validation_days or base_config.validation_trading_days,
        test_trading_days=test_days or base_config.test_trading_days,
        label_horizon_trading_days=base_config.label_horizon_trading_days,
        blind_trading_days=blind_days or base_config.blind_trading_days,
        embargo_trading_days=embargo_days or base_config.embargo_trading_days,
        purge_trading_days=embargo_days or base_config.purge_trading_days,
    )
    return list(
        create_walk_forward_plan(
            cycles,
            trading_calendar=trading_calendar,
            config=cfg,
            calendar_evidence_fingerprint=calendar_evidence_fingerprint,
        ).folds
    )


def validate_walk_forward_plan(
    plan: WalkForwardPlan,
    *,
    cycles: Sequence[CycleInput],
    trading_calendar: Sequence[datetime],
) -> None:
    """Fail closed on leakage, overlap, calendar tamper, or OOF duplication."""
    calendar = _canonical_calendar(trading_calendar)
    calendar_index = {day: idx for idx, day in enumerate(calendar)}
    expected_calendar_fingerprint = _fingerprint({"trading_calendar": [_dt(day) for day in calendar]})
    if expected_calendar_fingerprint != plan.calendar_fingerprint:
        raise ValueError("calendar fingerprint mismatch")

    records = _canonical_cycles(cycles, calendar, calendar_index, plan.config)
    all_ids = {record.cycle_id for record in records}
    seen_oof: set[str] = set()
    for fold in plan.folds:
        train = set(fold.train_cycle_ids)
        val = set(fold.validation_cycle_ids)
        test = set(fold.test_cycle_ids)
        blind = set(fold.blind_cycle_ids)
        purged = set(fold.purged_cycle_ids)
        prediction = val | test
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
        seen_oof.update(prediction)
        _validate_embargo(fold, calendar_index, plan.config)

    rebuilt = _build_plan(
        config=plan.config,
        folds=plan.folds,
        calendar_fingerprint=plan.calendar_fingerprint,
        calendar_evidence_fingerprint=plan.calendar_evidence_fingerprint,
        blind_cycle_ids=plan.blind_cycle_ids,
        classification_by_fold=plan.classification_by_fold,
        cycle_records=records,
    )
    if rebuilt.fingerprint != plan.fingerprint:
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


def verify_cycle_atomicity(fold: WalkForwardFold, all_cycle_ids: set[str]) -> bool:
    """Legacy boolean cycle-overlap verifier."""
    train = set(fold.train_cycle_ids)
    val = set(fold.validation_cycle_ids)
    test = set(fold.test_cycle_ids)
    blind = set(fold.blind_cycle_ids)
    purged = set(fold.purged_cycle_ids)
    segments = [train, val, test, blind, purged]
    for left_idx, left in enumerate(segments):
        for right in segments[left_idx + 1 :]:
            if left & right:
                return False
    return set().union(*segments).issubset(all_cycle_ids)


def _canonical_calendar(trading_calendar: Sequence[datetime]) -> tuple[datetime, ...]:
    if not trading_calendar:
        raise ValueError("trading_calendar is missing")
    for day in trading_calendar:
        assert_aware(day, "trading_calendar")
    unique = set(trading_calendar)
    if len(unique) != len(trading_calendar):
        raise ValueError("trading_calendar contains duplicate dates")
    return tuple(sorted(trading_calendar))


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
        start = cycle.cycle_start
        end = cycle.cycle_end or cycle.last_event_date
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
        maturity = calendar[-1]
    else:
        maturity = calendar[maturity_index]
    return _CycleRecord(
        cycle_id=cycle_id,
        stock_id=stock_id,
        cycle_start=start,
        cycle_end=end,
        label_maturity=maturity,
        status=status,
    )


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
        maturity_idx = calendar_index[record.label_maturity]
        if record.cycle_id in blind_cycle_ids:
            classification[record.cycle_id] = CycleDisposition.BLIND
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
                "label_maturity": _dt(record.label_maturity),
                "status": record.status,
            }
            for record in cycle_records
        ],
        "folds": [_fold_payload(fold) for fold in folds],
        "blind_cycle_ids": sorted(blind_cycle_ids),
        "classification_by_fold": {
            str(fid): {cid: disp.value for cid, disp in sorted(mapping.items())}
            for fid, mapping in sorted(classification_by_fold.items())
        },
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


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dt(value: datetime) -> str:
    return value.isoformat()
