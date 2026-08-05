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
)
from seatflow.ingestion.provider import DataProvider, ProviderCapability


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

    def capability(self) -> dict[str, str]:
        return dict(self._capability)

    def _read_file(self, filename: str) -> pl.DataFrame:
        path = self._base_dir / filename
        if not path.exists():
            self._capability[filename] = ProviderCapability.UNAVAILABLE
            return pl.DataFrame()
        self._capability[filename] = ProviderCapability.AVAILABLE
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path)

    def _rows_to_bronze(
        self,
        df: pl.DataFrame,
        *,
        event_time_col: str = "event_time",
        available_time_col: str = "available_time",
    ) -> Iterator[BronzeRecord]:
        for row in df.iter_rows(named=True):
            payload = dict(row)
            event_time = self._to_aware(payload.get(event_time_col))
            available_time = self._to_aware(payload.get(available_time_col))
            record_id = str(payload.get("source_record_id", f"{event_time}-{len(payload)}"))
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

    def fetch_daily_bars(self, filename: str = "daily_bars.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_stock_master(self, filename: str = "stock_master.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_trade_calendar(self, filename: str = "trade_calendar.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_lhb_summary(self, filename: str = "lhb_summary.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_lhb_participation(self, filename: str = "lhb_participation.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_memberships(self, filename: str = "memberships.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)

    def fetch_tradeability(self, filename: str = "tradeability.csv", **kwargs) -> Iterator[BronzeRecord]:
        df = self._read_file(filename)
        yield from self._rows_to_bronze(df)
