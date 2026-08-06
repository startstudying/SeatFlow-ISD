"""AkShare adapter with explicit UTF-8 field contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping, Sequence

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
    canonicalize_provider_fields,
)

_MOJIBAKE_DATE = "\u00e6\u0097\u00a5\u00e6\u009c\u009f"
_MOJIBAKE_CODE = "\u00e4\u00bb\u00a3\u00e7\u00a0\u0081"

_BUILT_IN_ALIASES: dict[str, tuple[str, ...]] = {
    "event_time": ("日期", "交易日期", "trade_date", _MOJIBAKE_DATE),
    "stock_code": (
        "代码",
        "股票代码",
        "证券代码",
        "symbol",
        "ts_code",
        _MOJIBAKE_CODE,
    ),
    "seat_name": ("营业部名称", "机构名称", "上榜营业部"),
}


class AkshareProvider(DataProvider):
    """AkShare adapter using an injected transport callable."""

    def __init__(
        self,
        *,
        transport: Callable[..., list[dict[str, Any]]],
        source_version: str = "akshare-v1",
        ingestion_run_id: str = "akshare-run",
        field_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._transport = transport
        self._source_version = source_version
        self._ingestion_run_id = ingestion_run_id
        self._capability: dict[str, str] = {}
        self._capability_evidence: dict[str, ProviderCapabilityEvidence] = {}
        self._field_aliases = {
            key: tuple(values) for key, values in (field_aliases or {}).items()
        }

    def capability(self) -> dict[str, str]:
        return dict(self._capability)

    def capability_evidence(self) -> dict[str, ProviderCapabilityEvidence]:
        return dict(self._capability_evidence)

    def _records_from_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_dataset: str,
        event_time_field: str = "日期",
        available_time_field: str | None = None,
        stock_code_field: str | None = "代码",
        seat_name_field: str | None = None,
    ) -> Iterator[BronzeRecord]:
        canonical_fields = {"event_time": event_time_field}
        if stock_code_field is not None:
            canonical_fields["stock_code"] = stock_code_field
        if seat_name_field is not None:
            canonical_fields["seat_name"] = seat_name_field
        required = tuple(canonical_fields)

        if not rows:
            self._set_evidence(
                source_dataset,
                status=ProviderCapability.DEGRADED,
                reason_code=CapabilityReason.EMPTY_RESPONSE,
                required=required,
                missing=required,
            )
            return

        normalized_rows: list[dict[str, Any]] = []
        present_union: set[str] = set()
        for row in rows:
            try:
                payload, present, missing = canonicalize_provider_fields(
                    row,
                    canonical_fields=canonical_fields,
                    built_in_aliases=_BUILT_IN_ALIASES,
                    configured_aliases=self._field_aliases,
                )
            except ProviderFieldError as exc:
                self._set_evidence(
                    source_dataset,
                    status=ProviderCapability.DEGRADED,
                    reason_code=exc.reason_code,
                    required=required,
                    details=exc.details,
                )
                raise ProviderFieldError(
                    dataset=source_dataset,
                    reason_code=exc.reason_code,
                    details=exc.details,
                ) from exc
            present_union.update(present)
            if missing:
                self._set_evidence(
                    source_dataset,
                    status=ProviderCapability.DEGRADED,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    required=required,
                    present=present,
                    missing=missing,
                )
                raise ProviderFieldError(
                    dataset=source_dataset,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    missing_fields=missing,
                )
            normalized_rows.append(payload)

        present = tuple(field for field in required if field in present_union)
        self._set_evidence(
            source_dataset,
            status=ProviderCapability.AVAILABLE,
            reason_code=CapabilityReason.COMPLETE,
            required=required,
            present=present,
        )
        at_field = available_time_field or event_time_field
        for payload in normalized_rows:
            event_time = self._parse_date(payload[event_time_field])
            available_time = self._parse_date(payload.get(at_field, payload[event_time_field]))
            record_id = compute_source_record_id(
                "akshare",
                source_dataset,
                payload,
                stable_key_fields=tuple(canonical_fields.values()),
            )
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

    def _set_evidence(
        self,
        source_dataset: str,
        *,
        status: str,
        reason_code: str,
        required: tuple[str, ...],
        present: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._capability[source_dataset] = status
        coverage = len(present) / len(required) if required else 1.0
        self._capability_evidence[source_dataset] = ProviderCapabilityEvidence(
            status=status,
            reason_code=reason_code,
            required_fields=required,
            present_fields=present,
            missing_fields=missing,
            field_coverage=coverage,
            details=dict(details or {}),
        )

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(str(value), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {value!r}")

    def fetch_daily_bars(self, symbol: str = "", **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_zh_a_hist", symbol=symbol, **kwargs)
        yield from self._records_from_rows(rows, source_dataset="daily_bars")

    def fetch_stock_master(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_info_a_code_name", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="stock_master",
            event_time_field="上市日期",
        )

    def fetch_trade_calendar(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("tool_trade_date_hist_sina", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="trade_calendar",
            event_time_field="trade_date",
            stock_code_field=None,
        )

    def fetch_lhb_summary(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_lhb_detail_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="lhb_summary")

    def fetch_lhb_participation(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_lhb_jgmmtj_em", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="lhb_participation",
            seat_name_field="营业部名称",
        )

    def fetch_memberships(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_board_industry_cons_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="memberships")

    def fetch_tradeability(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_tfp_em", **kwargs)
        yield from self._records_from_rows(rows, source_dataset="tradeability")
