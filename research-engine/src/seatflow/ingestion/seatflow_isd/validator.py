"""Row-level integrity checks for SeatFlow-ISD V1 records.

Validators in this module return human-readable issues; they never
mutate the input. The caller can decide whether to raise, log, or
record the issue in the quality report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from seatflow.ingestion.seatflow_isd.contracts import (
    ISDEventV3,
    ISDLabelRowV3,
    ISDLhbUniverseV3,
    ISDMarketBarV3,
    ISDSeatParticipationV3,
    ISDSampleV3,
)


@dataclass(frozen=True)
class ValidationIssue:
    row_id: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": [
                {"row_id": i.row_id, "field": i.field, "message": i.message}
                for i in self.issues
            ],
        }


def validate_events(events: Iterable[ISDEventV3]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        if not event.event_id:
            issues.append(ValidationIssue(
                event.event_id, "event_id", "missing event_id"
            ))
        if event.available_time < event.trade_date:
            issues.append(ValidationIssue(
                event.event_id, "available_time",
                "available_time precedes trade_date",
            ))
        key = (event.stock_id, event.trade_date.isoformat(), ",".join(event.reason_codes))
        if key in seen:
            issues.append(ValidationIssue(
                event.event_id, "natural_key",
                f"duplicate event for {key}",
            ))
        seen.add(key)
    return ValidationReport(tuple(issues))


def validate_participations(
    participations: Iterable[ISDSeatParticipationV3],
    known_event_ids: set[str],
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for p in participations:
        if not p.event_id:
            issues.append(ValidationIssue(
                p.seat_id, "event_id",
                "participation has no event_id (unbound at build time)",
            ))
            continue
        if p.event_id not in known_event_ids:
            issues.append(ValidationIssue(
                p.seat_id, "event_id",
                f"participation references unknown event {p.event_id}",
            ))
    return ValidationReport(tuple(issues))


def validate_market_bars(bars: Iterable[ISDMarketBarV3]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for b in bars:
        if b.open <= 0 or b.close <= 0:
            issues.append(ValidationIssue(
                f"{b.stock_id}@{b.trade_date.date().isoformat()}",
                "price",
                "non-positive price in market bar",
            ))
    return ValidationReport(tuple(issues))


def validate_samples(samples: Iterable[ISDSampleV3]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for s in samples:
        if s.pool_type != "DIRECT":
            issues.append(ValidationIssue(
                s.sample_id, "pool_type",
                f"unexpected pool_type={s.pool_type}; V1 must be DIRECT-only",
            ))
        if s.cutoff_time < s.available_time:
            issues.append(ValidationIssue(
                s.sample_id, "cutoff_time",
                "cutoff_time precedes available_time",
            ))
    return ValidationReport(tuple(issues))


def validate_labels(labels: Iterable[ISDLabelRowV3]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for label in labels:
        if label.maturity.value == "UNMATURE":
            if label.exit_1d_return is not None or label.exit_3d_return is not None:
                issues.append(ValidationIssue(
                    label.sample_id, "maturity",
                    "UNMATURE label must have null returns",
                ))
        if label.maturity.value == "MATURE":
            if label.exit_1d_return is None or label.exit_3d_return is None or label.exit_5d_return is None:
                issues.append(ValidationIssue(
                    label.sample_id, "maturity",
                    "MATURE label is missing required return fields",
                ))
    return ValidationReport(tuple(issues))


def validate_universe(
    universe: ISDLhbUniverseV3,
    samples: Iterable[ISDSampleV3],
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for s in samples:
        if s.stock_id not in universe.stock_ids:
            issues.append(ValidationIssue(
                s.sample_id, "stock_id",
                f"sample {s.sample_id} stock {s.stock_id} not in LHB universe",
            ))
    return ValidationReport(tuple(issues))


def validate_all(
    events: Iterable[ISDEventV3],
    participations: Iterable[ISDSeatParticipationV3],
    market_bars: Iterable[ISDMarketBarV3],
    samples: Iterable[ISDSampleV3],
    labels: Iterable[ISDLabelRowV3],
    universe: ISDLhbUniverseV3,
) -> ValidationReport:
    events = tuple(events)
    participations = tuple(participations)
    market_bars = tuple(market_bars)
    samples = tuple(samples)
    labels = tuple(labels)
    known_event_ids = {e.event_id for e in events}
    reports = [
        validate_events(events),
        validate_participations(participations, known_event_ids),
        validate_market_bars(market_bars),
        validate_samples(samples),
        validate_labels(labels),
        validate_universe(universe, samples),
    ]
    merged_issues: list[ValidationIssue] = []
    for r in reports:
        merged_issues.extend(r.issues)
    return ValidationReport(tuple(merged_issues))


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_all",
    "validate_events",
    "validate_labels",
    "validate_market_bars",
    "validate_participations",
    "validate_samples",
    "validate_universe",
]
