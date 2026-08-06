"""File-based data provider for local CSV/Parquet ingestion.

This adapter is used for batch ingestion from pre-downloaded data files.
It never reads environment variables at import time — all configuration
must be passed via the constructor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from seatflow.ingestion.models import (
    BronzeRecord,
    SourceConfidence,
    compute_payload_hash,
    compute_source_record_id,
)
from seatflow.ingestion.provider import (
    CapabilityReason,
    DataProvider,
    ProviderCapability,
    ProviderCapabilityEvidence,
    ProviderFieldError,
)


class FileProvider(DataProvider):
    """Provider that reads from local Parquet or CSV files."""

    def __init__(
        self,
        *,
        source_name: str = "file",
        source_version: str = "v1",
        base_dir: str | Path = ".",
        ingestion_run_id: str = "file-run",
        allow_pre_announced: bool = False,
    ) -> None:
        self._source = source_name
        self._source_version = source_version
        self._base_dir = Path(base_dir)
        self._ingestion_run_id = ingestion_run_id
        self._allow_pre_announced = allow_pre_announced
        self._capability: dict[str, str] = {}
        self._capability_evidence: dict[str, ProviderCapabilityEvidence] = {}

    def capability(self) -> dict[str, str]:
        return dict(self._capability)

    def capability_evidence(self) -> dict[str, ProviderCapabilityEvidence]:
        return dict(self._capability_evidence)

    def _read_file(self, filename: str) -> pl.DataFrame:
        path = self._base_dir / filename
        if not path.exists():
            self._capability[filename] = ProviderCapability.UNAVAILABLE
            self._capability_evidence[filename] = ProviderCapabilityEvidence(
                status=ProviderCapability.UNAVAILABLE,
                reason_code=CapabilityReason.SOURCE_UNAVAILABLE,
                details={"path": str(path)},
            )
            return pl.DataFrame()
        self._capability[filename] = ProviderCapability.AVAILABLE
        self._capability_evidence[filename] = ProviderCapabilityEvidence(
            status=ProviderCapability.AVAILABLE,
            reason_code=CapabilityReason.COMPLETE,
            field_coverage=1.0,
            details={"path": str(path)},
        )
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path)

    def _rows_to_bronze(
        self,
        df: pl.DataFrame,
        *,
        source_dataset: str,
        capability_key: str,
        event_time_col: str = "event_time",
        available_time_col: str = "available_time",
    ) -> Iterator[BronzeRecord]:
        required_columns = {
            "event_time": event_time_col,
            "available_time": available_time_col,
        }
        rows = [dict(row) for row in df.iter_rows(named=True)]
        if not rows:
            if self._capability.get(capability_key) == ProviderCapability.UNAVAILABLE:
                return
            self._capability[capability_key] = ProviderCapability.DEGRADED
            self._capability_evidence[capability_key] = ProviderCapabilityEvidence(
                status=ProviderCapability.DEGRADED,
                reason_code=CapabilityReason.EMPTY_RESPONSE,
                required_fields=tuple(required_columns),
                missing_fields=tuple(required_columns),
                field_coverage=0.0,
            )
            return
        for payload in rows:
            present = tuple(
                semantic
                for semantic, column in required_columns.items()
                if payload.get(column) not in (None, "")
            )
            missing = tuple(
                semantic for semantic in required_columns if semantic not in present
            )
            if missing:
                self._capability[capability_key] = ProviderCapability.DEGRADED
                self._capability_evidence[capability_key] = ProviderCapabilityEvidence(
                    status=ProviderCapability.DEGRADED,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    required_fields=tuple(required_columns),
                    present_fields=present,
                    missing_fields=missing,
                    field_coverage=len(present) / len(required_columns),
                )
                raise ProviderFieldError(
                    dataset=source_dataset,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    missing_fields=missing,
                )
        self._capability[capability_key] = ProviderCapability.AVAILABLE
        self._capability_evidence[capability_key] = ProviderCapabilityEvidence(
            status=ProviderCapability.AVAILABLE,
            reason_code=CapabilityReason.COMPLETE,
            required_fields=tuple(required_columns),
            present_fields=tuple(required_columns),
            field_coverage=1.0,
        )
        for payload in rows:
            event_time = self._to_aware(payload.get(event_time_col))
            available_time = self._to_aware(payload.get(available_time_col))
            record_id = compute_source_record_id(
                self._source,
                source_dataset,
                payload,
                stable_key_fields=("source_record_id", event_time_col, available_time_col),
            )
            yield BronzeRecord(
                event_time=event_time,
                available_time=available_time,
                ingested_at=datetime.now(timezone.utc),
                source=self._source,
                source_record_id=record_id,
                source_version=self._source_version,
                revision_no=int(payload.get("revision_no", 0)),
                raw_payload=payload,
                raw_payload_hash=compute_payload_hash(payload),
                ingestion_run_id=self._ingestion_run_id,
                source_confidence=SourceConfidence.MEDIUM,
            )

    @staticmethod
    def _to_aware(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return datetime.fromisoformat(str(value))

    def fetch_daily_bars(
        self, filename: str = "daily_bars.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="daily_bars",
            capability_key=filename,
        )

    def fetch_stock_master(
        self, filename: str = "stock_master.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="stock_master",
            capability_key=filename,
        )

    def fetch_trade_calendar(
        self, filename: str = "trade_calendar.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="trade_calendar",
            capability_key=filename,
        )

    def fetch_lhb_summary(
        self, filename: str = "lhb_summary.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="lhb_summary",
            capability_key=filename,
        )

    def fetch_lhb_participation(
        self, filename: str = "lhb_participation.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="lhb_participation",
            capability_key=filename,
        )

    def fetch_memberships(
        self, filename: str = "memberships.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="memberships",
            capability_key=filename,
        )

    def fetch_tradeability(
        self, filename: str = "tradeability.csv", **kwargs: Any
    ) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(
            df,
            source_dataset="tradeability",
            capability_key=filename,
        )
