"""Tushare data adapter (uses injected transport, not live network)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from seatflow.ingestion.models import (
    BronzeRecord,
    SourceConfidence,
    compute_payload_hash,
)
from seatflow.ingestion.provider import DataProvider, ProviderCapability


class TushareProvider(DataProvider):
    """Tushare API adapter that uses an injected transport callable.

    The transport is a callable that receives an API method name and
    keyword arguments, and returns a list of dicts (raw API response).
    """

    def __init__(
        self,
        *,
        transport: Callable[..., list[dict[str, Any]]],
        source_version: str = "tushare-pro-v1",
        ingestion_run_id: str = "tushare-run",
        token: str | None = None,
    ) -> None:
        self._transport = transport
        self._source_version = source_version
        self._ingestion_run_id = ingestion_run_id
        self._token = token
        self._capability: dict[str, str] = {}

    def capability(self) -> dict[str, str]:
        return dict(self._capability)

    def _records_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_dataset: str,
        event_time_field: str = "trade_date",
        available_time_field: str = "ann_date",
    ) -> Iterator[BronzeRecord]:
        if not rows:
            self._capability[source_dataset] = ProviderCapability.DEGRADED
            return

        self._capability[source_dataset] = ProviderCapability.AVAILABLE
        for row in rows:
            payload = dict(row)
            event_time = self._parse_date(payload.get(event_time_field))
            available_time = self._parse_date(
                payload.get(available_time_field, payload.get(event_time_field))
            )
            record_id = f"{source_dataset}-{payload.get('ts_code', '')}-{payload.get(event_time_field, '')}"
            yield BronzeRecord(
                event_time=event_time,
                available_time=available_time,
                ingested_at=datetime.now(timezone.utc),
                source="tushare",
                source_record_id=record_id,
                source_version=self._source_version,
                revision_no=int(payload.get("revision_no", 0)),
                raw_payload=payload,
                raw_payload_hash=compute_payload_hash(payload),
                ingestion_run_id=self._ingestion_run_id,
                source_confidence=SourceConfidence.HIGH,
            )

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        s = str(value)
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {value!r}")

    def fetch_daily_bars(self, ts_code: str = "", **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("daily", ts_code=ts_code, **kwargs)
        yield from self._records_from_rows(rows, source_dataset="daily_bars")

    def fetch_stock_master(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_basic", **kwargs)
        yield from self._records_from_rows(
            rows, source_dataset="stock_master",
            event_time_field="list_date", available_time_field="list_date",
        )

    def fetch_trade_calendar(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("trade_cal", **kwargs)
        yield from self._records_from_rows(
            rows, source_dataset="trade_calendar",
            event_time_field="cal_date", available_time_field="cal_date",
        )

    def fetch_lhb_summary(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("top_list", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="lhb_summary")

    def fetch_lhb_participation(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("top_inst", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="lhb_participation")

    def fetch_memberships(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("index_member", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="memberships")

    def fetch_tradeability(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("suspend_d", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="tradeability")
