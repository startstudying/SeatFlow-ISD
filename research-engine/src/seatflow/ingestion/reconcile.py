"""Reconciliation of conflicting records from multiple providers.

When two sources disagree on a field (e.g. buy amount), both Bronze
records are preserved. A versioned reconciliation rule selects one value
for the Silver layer and produces a ConflictReport documenting the
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from seatflow.ingestion.models import BronzeRecord, SourceConfidence


@dataclass(frozen=True)
class SourceValue:
    """A value observed from a specific source."""

    source: str
    source_record_id: str
    source_version: str
    value: Any
    confidence: SourceConfidence


@dataclass(frozen=True)
class ConflictReport:
    """Report documenting a conflict and its resolution."""

    key: str
    field_name: str
    sources: list[SourceValue]
    selected_value: Any
    selected_source: str
    rule_version: str
    confidence: SourceConfidence
    notes: str = ""


@dataclass(frozen=True)
class ReconciledRecord:
    """A single reconciled record after conflict resolution."""

    key: str
    fields: dict[str, Any] = field(default_factory=dict)
    conflict_reports: list[ConflictReport] = field(default_factory=list)
    source_record_ids: list[str] = field(default_factory=list)


# ── Reconciliation rules ───────────────────────────────────────────────

RULE_VERSION = "reconcile-v1"

_CONFIDENCE_RANK: dict[SourceConfidence, int] = {
    SourceConfidence.HIGH: 3,
    SourceConfidence.MEDIUM: 2,
    SourceConfidence.LOW: 1,
    SourceConfidence.DEGRADED: 0,
}


def _select_by_confidence(values: list[SourceValue]) -> SourceValue:
    """Select the value from the highest-confidence source."""
    return max(values, key=lambda sv: _CONFIDENCE_RANK.get(sv.confidence, 0))


def reconcile_field(
    key: str,
    field_name: str,
    bronze_records: list[BronzeRecord],
    *,
    field_extractor: Callable[[dict[str, Any]], Any],
    rule_version: str = RULE_VERSION,
) -> tuple[Any, ConflictReport | None]:
    """Reconcile a single field across multiple Bronze records.

    Returns (selected_value, conflict_report_or_None). If all sources
    agree, no conflict report is generated.
    """
    source_values: list[SourceValue] = []
    for rec in bronze_records:
        value = field_extractor(rec.raw_payload)
        source_values.append(
            SourceValue(
                source=rec.source,
                source_record_id=rec.source_record_id,
                source_version=rec.source_version,
                value=value,
                confidence=rec.source_confidence,
            )
        )

    if not source_values:
        raise ValueError(f"No values to reconcile for key={key}, field={field_name}")

    # Check for conflict
    unique_values = {sv.value for sv in source_values}
    if len(unique_values) == 1:
        return source_values[0].value, None

    # Conflict: select by confidence, ties broken by source name
    selected = _select_by_confidence(source_values)
    report = ConflictReport(
        key=key,
        field_name=field_name,
        sources=source_values,
        selected_value=selected.value,
        selected_source=selected.source,
        rule_version=rule_version,
        confidence=selected.confidence,
        notes=f"Conflict resolved by confidence ranking ({rule_version})",
    )
    return selected.value, report


def reconcile_records(
    records_by_key: dict[str, list[BronzeRecord]],
    *,
    field_specs: dict[str, Callable[[dict[str, Any]], Any]],
) -> dict[str, ReconciledRecord]:
    """Reconcile multiple fields for a set of records keyed by business key.

    Args:
        records_by_key: business key -> list of Bronze records from
            different sources.
        field_specs: field name -> extractor callable that reads the
            field from a raw_payload dict.

    Returns: business key -> ReconciledRecord.
    """
    result: dict[str, ReconciledRecord] = {}
    for key, records in records_by_key.items():
        fields: dict[str, Any] = {}
        conflicts: list[ConflictReport] = []
        source_ids: list[str] = []
        for rec in records:
            source_ids.append(rec.source_record_id)
        for field_name, extractor in field_specs.items():
            value, report = reconcile_field(
                key, field_name, records, field_extractor=extractor
            )
            fields[field_name] = value
            if report is not None:
                conflicts.append(report)
        result[key] = ReconciledRecord(
            key=key,
            fields=fields,
            conflict_reports=conflicts,
            source_record_ids=source_ids,
        )
    return result
