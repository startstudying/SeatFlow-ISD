"""AkShare data adapter (uses injected transport)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from seatflow.ingestion.models import (
    BronzeRecord,
    SourceConfidence,
    compute_payload_hash,
)
from seatflow.ingestion.provider import DataProvider, ProviderCapability


class AkshareProvider(DataProvider):
    """AkShare adapter using an injected transport callable.

    The transport receives a function name and kwargs, and returns
    a list of dicts.
    """

    def __init__(
        self,
        *,
        transport: Callable[..., list[dict[str, Any]]],
        source_version: str = "akshare-v1",
        ingestion_run_id: str = "akshare-run",
    ) -> None:
        self._transport = transport
        self._source_version = source_version
        self._ingestion_run_id = ingestion_run_id
        self._capability: dict[str, str] = {}

    def capability(self) -> dict[str, str]:
        return dict(self._capability)

    def _records_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_dataset: str,
        event_time_field: str = "日期",
        available_time_field: str | None = None,
    ) -> Iterator[BronzeRecord]:
        if not rows:
            self._capability[source_dataset] = ProviderCapability.DEGRADED
            return
        self._capability[source_dataset] = ProviderCapability.AVAILABLE
        at_field = available_time_field or event_time_field
        for row in rows:
            payload = dict(row)
            event_time = self._parse_date(payload.get(event_time_field))
            available_time = self._parse_date(payload.get(at_field, payload.get(event_time_field)))
            record_id = f"{source_dataset}-{hash(tuple(sorted(payload.items())))}"
            yield BronzeRecord(
                event_time=event_time,
                available_time=available_time,
                ingested_at=datetime.now(timezone.utc),
                source="akshare",
                source_record_id=record_id,
                source_version=self._source_version,
                raw_payload=payload,
                raw_payload_hash=compute_payload_hash(payload),
                ingestion_run_id=self._ingestion_run_id,
                source_confidence=SourceConfidence.MEDIUM,
            )

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        s = str(value)
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {value!r}")

    def fetch_daily_bars(self, symbol: str = "", **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_zh_a_hist", symbol=symbol, **kwargs)
        yield from self._records_from_rows(rows, source_dataset="daily_bars")

    def fetch_stock_master(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_info_a_code_name", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="stock_master", event_time_field="上市日期")

    def fetch_trade_calendar(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("tool_trade_date_hist_sina", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="trade_calendar", event_time_field="trade_date")

    def fetch_lhb_summary(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_lhb_detail_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="lhb_summary")

    def fetch_lhb_participation(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_lhb_jgmmtj_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="lhb_participation")

    def fetch_memberships(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_board_industry_cons_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="memberships")

    def fetch_tradeability(self, **kwargs) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_tfp_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="tradeability")
