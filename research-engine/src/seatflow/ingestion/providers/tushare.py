"""Tushare adapter with explicit field and coverage contracts."""

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

_BUILT_IN_ALIASES: dict[str, tuple[str, ...]] = {
    "event_time": (
        "trade_date",
        "cal_date",
        "list_date",
        "suspend_date",
        "in_date",
    ),
    "stock_code": ("ts_code", "con_code"),
    "seat_name": ("exalter", "seat_name"),
}


class TushareProvider(DataProvider):
    """Tushare API adapter that uses an injected transport callable."""

    def __init__(
        self,
        *,
        transport: Callable[..., list[dict[str, Any]]],
        source_version: str = "tushare-pro-v1",
        ingestion_run_id: str = "tushare-run",
        token: str | None = None,
        field_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._transport = transport
        self._source_version = source_version
        self._ingestion_run_id = ingestion_run_id
        self._token = token
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
        event_time_field: str = "trade_date",
        available_time_field: str = "ann_date",
        stock_code_field: str | None = "ts_code",
        seat_name_field: str | None = None,
        proxy_only: bool = False,
        endpoint: str = "",
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
                proxy_only=proxy_only,
                details={"endpoint": endpoint} if endpoint else {},
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
                error_details = {"endpoint": endpoint, **exc.details} if endpoint else exc.details
                self._set_evidence(
                    source_dataset,
                    status=ProviderCapability.DEGRADED,
                    reason_code=exc.reason_code,
                    required=required,
                    proxy_only=proxy_only,
                    details=error_details,
                )
                raise ProviderFieldError(
                    dataset=source_dataset,
                    reason_code=exc.reason_code,
                    details=error_details,
                ) from exc
            present_union.update(present)
            if missing:
                missing_details = {"endpoint": endpoint} if endpoint else {}
                self._set_evidence(
                    source_dataset,
                    status=ProviderCapability.DEGRADED,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    required=required,
                    present=present,
                    missing=missing,
                    proxy_only=proxy_only,
                    details=missing_details,
                )
                raise ProviderFieldError(
                    dataset=source_dataset,
                    reason_code=CapabilityReason.MISSING_REQUIRED_FIELDS,
                    missing_fields=missing,
                    details=missing_details,
                )
            normalized_rows.append(payload)

        status = ProviderCapability.DEGRADED if proxy_only else ProviderCapability.AVAILABLE
        reason = CapabilityReason.PROXY_COVERAGE_ONLY if proxy_only else CapabilityReason.COMPLETE
        details: dict[str, Any] = {}
        if proxy_only:
            details = {
                "endpoint": endpoint,
                "formal_use_allowed": False,
                "limitation": "Aggregate proxy endpoint; complete real seat detail is not proven.",
            }
        present = tuple(field for field in required if field in present_union)
        self._set_evidence(
            source_dataset,
            status=status,
            reason_code=reason,
            required=required,
            present=present,
            proxy_only=proxy_only,
            details=details,
        )
        for payload in normalized_rows:
            event_time = self._parse_date(payload[event_time_field])
            available_time = self._parse_date(
                payload.get(available_time_field, payload[event_time_field])
            )
            record_id = compute_source_record_id(
                "tushare",
                source_dataset,
                payload,
                stable_key_fields=tuple(canonical_fields.values()),
            )
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
                source_confidence=(
                    SourceConfidence.DEGRADED if proxy_only else SourceConfidence.HIGH
                ),
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
        proxy_only: bool = False,
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
            proxy_only=proxy_only,
            details=dict(details or {}),
        )

    @staticmethod
    def _parse_date(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        for fmt in (
            "%Y%m%d",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(str(value), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: {value!r}")

    def fetch_daily_bars(self, ts_code: str = "", **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("daily", ts_code=ts_code, **kwargs)
        yield from self._records_from_rows(rows, source_dataset="daily_bars", endpoint="daily")

    def fetch_stock_master(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("stock_basic", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="stock_master",
            event_time_field="list_date",
            available_time_field="list_date",
            endpoint="stock_basic",
        )

    def fetch_trade_calendar(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("trade_cal", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="trade_calendar",
            event_time_field="cal_date",
            available_time_field="cal_date",
            stock_code_field=None,
            endpoint="trade_cal",
        )

    def fetch_lhb_summary(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("top_list", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="lhb_summary",
            endpoint="top_list",
        )

    def fetch_lhb_participation(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("top_inst", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="lhb_participation",
            seat_name_field="exalter",
            proxy_only=True,
            endpoint="top_inst",
        )

    def fetch_memberships(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("index_member", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="memberships",
            event_time_field="in_date",
            stock_code_field="con_code",
            endpoint="index_member",
        )

    def fetch_tradeability(self, **kwargs: Any) -> Iterator[BronzeRecord]:
        rows = self._transport("suspend_d", **kwargs)
        yield from self._records_from_rows(
            rows,
            source_dataset="tradeability",
            event_time_field="suspend_date",
            endpoint="suspend_d",
        )
