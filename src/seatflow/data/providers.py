from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from hashlib import sha256
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from seatflow.data.contracts import (
    AdjustmentFactorEvidence,
    AdjustmentFactorResponseEvidence,
    DailyBar,
    DragonTigerSummary,
    SuspensionRecord,
    SuspensionResponseEvidence,
    normalize_stock_id,
    parse_decimal,
    parse_optional_decimal,
    parse_suspension_timing,
    parse_trade_date,
    suspension_daily_query_sha256,
    suspension_row_payload_sha256,
)
from seatflow.data.records import (
    INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    NAMED_SEAT_IDENTITY_SCOPE,
    DragonTigerRecord,
    TradeSide,
)

TUSHARE_API_URL = "http://api.tushare.pro"
TUSHARE_ALLOW_INSECURE_HTTP_ENV = "TUSHARE_ALLOW_INSECURE_HTTP"
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
_TUSHARE_ANONYMOUS_INSTITUTION_LABEL = "机构专用"
_INSTITUTIONAL_FLOW_PREFIX = "institutional-flow"
_TUSHARE_SUSPEND_D_SOURCE = "tushare:suspend_d"
_TUSHARE_ADJ_FACTOR_SOURCE = "tushare:adj_factor"
_TopInstAggregateKey = tuple[date, str, str | None, str, TradeSide, str]


@dataclass
class _TopInstPayload:
    trade_date: date
    stock_id: str
    seat_name: str
    reason: str | None
    buy: Decimal
    sell: Decimal
    net_buy: Decimal
    source_row_count: int
    source_side_set: set[str]


@dataclass
class _TopInstAggregate:
    amount: Decimal
    source_row_count: int
    source_side_set: set[str]


class ProviderError(RuntimeError):
    """Base error for provider failures that never includes a credential."""


class ProviderNotConfiguredError(ProviderError):
    pass


class ProviderCapabilityError(ProviderError):
    pass


class ProviderPermissionError(ProviderCapabilityError):
    pass


class ProviderResponseError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    configured: bool
    supports_daily_bars: bool
    supports_dragon_tiger_summary: bool
    supports_seat_detail: bool
    seat_detail_scope: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "supports_daily_bars": self.supports_daily_bars,
            "supports_dragon_tiger_summary": self.supports_dragon_tiger_summary,
            "supports_seat_detail": self.supports_seat_detail,
            "seat_detail_scope": self.seat_detail_scope,
            "notes": list(self.notes),
        }


class MarketDataProvider(Protocol):
    name: str

    def status(self) -> ProviderStatus: ...

    def fetch_daily_bars(
        self, trade_date: date, stock_id: str | None = None
    ) -> tuple[DailyBar, ...]: ...

    def fetch_dragon_tiger_summaries(self, trade_date: date) -> tuple[DragonTigerSummary, ...]: ...

    def fetch_dragon_tiger_seat_events(self, trade_date: date) -> tuple[DragonTigerRecord, ...]: ...

    def fetch_suspension_evidence(self, trade_date: date) -> SuspensionResponseEvidence: ...


PostJson = Callable[[str, Mapping[str, object], float], Mapping[str, object]]


class TushareHttpProvider:
    """Tushare Pro REST adapter; the token is read only from runtime configuration."""

    name = "tushare"

    def __init__(
        self,
        token: str | None = None,
        api_url: str | None = None,
        timeout_seconds: float = 30.0,
        available_after_close: time = time(18, 0),
        allow_insecure_http: bool | None = None,
        post_json: PostJson | None = None,
    ) -> None:
        self._token = token if token is not None else os.getenv("TUSHARE_TOKEN")
        self._api_url = api_url or os.getenv("TUSHARE_API_URL") or TUSHARE_API_URL
        self._timeout_seconds = timeout_seconds
        self._available_after_close = available_after_close
        self._allow_insecure_http = (
            allow_insecure_http
            if allow_insecure_http is not None
            else os.getenv(TUSHARE_ALLOW_INSECURE_HTTP_ENV) == "1"
        )
        self._post_json = post_json or _post_json

    def status(self) -> ProviderStatus:
        has_token = bool(self._token and self._token.strip())
        uses_insecure_http = self._api_url.startswith("http://")
        configured = has_token and (not uses_insecure_http or self._allow_insecure_http)
        notes = [
            "Uses Tushare Pro REST; token is read from TUSHARE_TOKEN only.",
            "top_inst is institution-tagged flow, not a complete brokerage-seat universe or proof "
            "of a stable institution identity; anonymous rows use short-lived proxies.",
            "Remote permission is checked only during an explicit ingestion request.",
        ]
        if not has_token:
            notes.append("TUSHARE_TOKEN is not configured.")
        if uses_insecure_http and not self._allow_insecure_http:
            notes.append(
                "Official REST URL is HTTP. Set TUSHARE_ALLOW_INSECURE_HTTP=1 only after review."
            )
        return ProviderStatus(
            provider=self.name,
            configured=configured,
            supports_daily_bars=True,
            supports_dragon_tiger_summary=True,
            supports_seat_detail=True,
            seat_detail_scope="institutional_only",
            notes=tuple(notes),
        )

    def fetch_daily_bars(
        self, trade_date: date, stock_id: str | None = None
    ) -> tuple[DailyBar, ...]:
        params: dict[str, object] = {"trade_date": trade_date.strftime("%Y%m%d")}
        if stock_id:
            params["ts_code"] = normalize_stock_id(stock_id)
        rows = self._call(
            "daily",
            params,
            "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )
        available_time = self._available_time(trade_date)
        return tuple(
            self._daily_bar_from_row(row, available_time)
            for row in rows
        )

    def fetch_daily_bar_history(
        self, stock_id: str, start_date: date, end_date: date
    ) -> tuple[DailyBar, ...]:
        """Fetch one stock's inclusive OHLCV history using the documented date-range API."""
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        rows = self._call(
            "daily",
            {
                "ts_code": normalize_stock_id(stock_id),
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )
        return tuple(
            self._daily_bar_from_row(row, self._available_time(parse_trade_date(row["trade_date"])))
            for row in rows
        )

    def fetch_trading_calendar(self, start_date: date, end_date: date) -> tuple[date, ...]:
        """Return open SSE dates for a bounded research period."""
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        rows = self._call(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
                "is_open": "1",
            },
            "exchange,cal_date,is_open",
        )
        open_dates = (
            parse_trade_date(row["cal_date"]) for row in rows if str(row["is_open"]) == "1"
        )
        return tuple(sorted(open_dates))

    def fetch_index_close_history(
        self, index_code: str, start_date: date, end_date: date
    ) -> tuple[tuple[date, Decimal], ...]:
        """Fetch an index close series without forcing it through the A-share identifier rule."""
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        rows = self._call(
            "index_daily",
            {
                "ts_code": index_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "ts_code,trade_date,close",
        )
        return tuple(
            sorted(
                (parse_trade_date(row["trade_date"]), parse_decimal(row["close"], "close"))
                for row in rows
            )
        )

    def fetch_index_open_history(
        self, index_code: str, start_date: date, end_date: date
    ) -> tuple[tuple[date, Decimal], ...]:
        """Fetch index opens for V2's executable benchmark entry/exit convention."""

        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        rows = self._call(
            "index_daily",
            {
                "ts_code": index_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "ts_code,trade_date,open",
        )
        return tuple(
            sorted(
                (parse_trade_date(row["trade_date"]), parse_decimal(row["open"], "open"))
                for row in rows
            )
        )

    def fetch_adjustment_factor_evidence(
        self, stock_id: str, start_date: date, end_date: date
    ) -> AdjustmentFactorResponseEvidence:
        """Fetch actual Tushare ``adj_factor`` rows with row and query hashes."""

        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        normalized_stock_id = normalize_stock_id(stock_id)
        rows = self._call(
            "adj_factor",
            {
                "ts_code": normalized_stock_id,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "ts_code,trade_date,adj_factor",
        )
        retrieved_at = datetime.now(SHANGHAI_TIMEZONE)
        canonical_rows: list[dict[str, str]] = []
        records: list[AdjustmentFactorEvidence] = []
        for row in rows:
            row_stock_id = normalize_stock_id(str(row["ts_code"]))
            if row_stock_id != normalized_stock_id:
                raise ProviderResponseError("adj_factor row stock does not match requested stock")
            trade_date = parse_trade_date(row["trade_date"])
            factor = parse_decimal(row["adj_factor"], "adj_factor")
            canonical = {
                "stock_id": row_stock_id,
                "trade_date": trade_date.isoformat(),
                "adjustment_factor": str(factor),
            }
            serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            canonical_rows.append(canonical)
            records.append(
                AdjustmentFactorEvidence(
                    stock_id=row_stock_id,
                    trade_date=trade_date,
                    adjustment_factor=factor,
                    source=_TUSHARE_ADJ_FACTOR_SOURCE,
                    retrieved_at=retrieved_at,
                    row_payload_sha256=sha256(serialized.encode("utf-8")).hexdigest(),
                )
            )
        query_payload = {
            "stock_id": normalized_stock_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "rows": sorted(canonical_rows, key=lambda item: item["trade_date"]),
        }
        return AdjustmentFactorResponseEvidence(
            stock_id=normalized_stock_id,
            start_date=start_date,
            end_date=end_date,
            source=_TUSHARE_ADJ_FACTOR_SOURCE,
            retrieved_at=retrieved_at,
            query_sha256=sha256(
                json.dumps(query_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            records=tuple(sorted(records, key=lambda item: item.trade_date)),
        )

    def fetch_stock_status_evidence(
        self, stock_id: str, start_date: date, end_date: date
    ) -> dict[str, object]:
        """Fetch source-hashed listing and name-change evidence for V2 eligibility."""

        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        normalized_stock_id = normalize_stock_id(stock_id)
        basic_rows = self._call(
            "stock_basic", {"ts_code": normalized_stock_id}, "ts_code,name,list_date"
        )
        name_rows = self._call(
            "namechange",
            {
                "ts_code": normalized_stock_id,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            "ts_code,start_date,end_date,name",
        )
        canonical_basic = [
            {
                "stock_id": normalize_stock_id(str(row["ts_code"])),
                "name": str(row.get("name") or ""),
                "list_date": str(row.get("list_date") or ""),
            }
            for row in basic_rows
        ]
        canonical_names = [
            {
                "stock_id": normalize_stock_id(str(row["ts_code"])),
                "start_date": str(row.get("start_date") or ""),
                "end_date": str(row.get("end_date") or ""),
                "name": str(row.get("name") or ""),
            }
            for row in name_rows
        ]
        if any(row["stock_id"] != normalized_stock_id for row in canonical_basic + canonical_names):
            raise ProviderResponseError("stock status evidence returned an unexpected stock")
        payload = {
            "stock_id": normalized_stock_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stock_basic": sorted(canonical_basic, key=lambda row: row["list_date"]),
            "namechange": sorted(canonical_names, key=lambda row: row["start_date"]),
        }
        return {
            **payload,
            "source": "tushare:stock_basic+namechange",
            "query_sha256": sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    def fetch_dragon_tiger_summaries(self, trade_date: date) -> tuple[DragonTigerSummary, ...]:
        rows = self._call(
            "top_list",
            {"trade_date": trade_date.strftime("%Y%m%d")},
            "trade_date,ts_code,name,l_buy,l_sell,net_amount,reason",
        )
        available_time = self._available_time(trade_date)
        summaries: list[DragonTigerSummary] = []
        for row in rows:
            stock_id = _normalize_a_share_or_none(row["ts_code"])
            if stock_id is None:
                continue
            try:
                buy_amount = parse_optional_decimal(row.get("l_buy"), "l_buy")
                sell_amount = parse_optional_decimal(row.get("l_sell"), "l_sell")
                net_amount = parse_optional_decimal(row.get("net_amount"), "net_amount")
            except ValueError:
                continue
            if buy_amount is None or sell_amount is None or net_amount is None:
                continue
            summaries.append(
                DragonTigerSummary(
                    trade_date=parse_trade_date(row["trade_date"]),
                    stock_id=stock_id,
                    stock_name=str(row.get("name") or ""),
                    buy_amount=buy_amount,
                    sell_amount=sell_amount,
                    net_amount=net_amount,
                    reason=str(row.get("reason") or ""),
                    source_name=self.name,
                    source_version="top_list",
                    available_time=available_time,
                )
            )
        return tuple(summaries)

    def fetch_dragon_tiger_seat_events(self, trade_date: date) -> tuple[DragonTigerRecord, ...]:
        rows = self._call(
            "top_inst",
            {"trade_date": trade_date.strftime("%Y%m%d")},
            "trade_date,ts_code,exalter,side,buy,sell,net_buy,reason",
        )
        available_time = self._available_time(trade_date)
        return self._canonicalize_top_inst_rows(rows, available_time)

    def fetch_suspension_evidence(self, trade_date: date) -> SuspensionResponseEvidence:
        """Fetch immutable evidence for the ``S`` rows of Tushare's ``suspend_d`` endpoint."""

        rows = self._call(
            "suspend_d",
            {"trade_date": trade_date.strftime("%Y%m%d")},
            "ts_code,trade_date,suspend_timing,suspend_type",
        )
        retrieved_at = datetime.now(SHANGHAI_TIMEZONE)
        all_keys: set[tuple[str, date, str, str | None]] = set()
        suspended: list[SuspensionRecord] = []
        try:
            daily_query_sha256 = suspension_daily_query_sha256(trade_date, rows)
        except (TypeError, ValueError) as error:
            raise ProviderResponseError(f"Tushare suspend_d row is invalid: {error}") from error
        for row in rows:
            try:
                record = SuspensionRecord(
                    stock_id=str(row["ts_code"]),
                    trade_date=parse_trade_date(row["trade_date"]),
                    suspend_type=str(row["suspend_type"]),
                    suspend_timing=parse_suspension_timing(row["suspend_timing"]),
                    source=_TUSHARE_SUSPEND_D_SOURCE,
                    retrieved_at=retrieved_at,
                    row_payload_sha256=suspension_row_payload_sha256(row),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ProviderResponseError(f"Tushare suspend_d row is invalid: {error}") from error
            if record.trade_date != trade_date:
                raise ProviderResponseError(
                    "Tushare suspend_d row date does not match the requested trade_date"
                )
            if record.canonical_key in all_keys:
                raise ProviderResponseError(
                    f"Tushare suspend_d contains duplicate canonical row: {record.canonical_key}"
                )
            all_keys.add(record.canonical_key)
            if record.suspend_type == "S":
                suspended.append(record)
        return SuspensionResponseEvidence(
            trade_date=trade_date,
            source=_TUSHARE_SUSPEND_D_SOURCE,
            retrieved_at=retrieved_at,
            source_row_count=len(rows),
            daily_query_sha256=daily_query_sha256,
            records=tuple(suspended),
        )

    def _call(
        self, api_name: str, params: Mapping[str, object], fields: str
    ) -> tuple[dict[str, object], ...]:
        token = self._token
        if not token or not token.strip():
            raise ProviderNotConfiguredError(
                "Tushare is not configured. Set TUSHARE_TOKEN in the runtime environment."
            )
        if self._api_url.startswith("http://") and not self._allow_insecure_http:
            raise ProviderNotConfiguredError(
                "Tushare REST uses HTTP. Set TUSHARE_ALLOW_INSECURE_HTTP=1 to explicitly allow it."
            )
        payload: dict[str, object] = {
            "api_name": api_name,
            "token": token,
            "params": dict(params),
            "fields": fields,
        }
        response = self._post_json(self._api_url, payload, self._timeout_seconds)
        try:
            code = int(str(response.get("code", -1)))
        except ValueError as error:
            raise ProviderResponseError(
                f"Tushare response for {api_name} has invalid code"
            ) from error
        if code != 0:
            if code == 2002:
                raise ProviderPermissionError(f"Tushare permission denied for {api_name}")
            raise ProviderResponseError(f"Tushare request failed for {api_name} (code {code})")
        data = response.get("data")
        if not isinstance(data, Mapping):
            raise ProviderResponseError(f"Tushare response for {api_name} has no data object")
        fields_value = data.get("fields")
        items_value = data.get("items")
        if not isinstance(fields_value, list) or not isinstance(items_value, list):
            raise ProviderResponseError(f"Tushare response for {api_name} has invalid tabular data")
        fields_list = [str(field) for field in fields_value]
        rows: list[dict[str, object]] = []
        for item in items_value:
            if not isinstance(item, list) or len(item) != len(fields_list):
                raise ProviderResponseError(f"Tushare response for {api_name} has an invalid row")
            rows.append(dict(zip(fields_list, item, strict=True)))
        return tuple(rows)

    def _available_time(self, trade_date: date) -> datetime:
        return datetime.combine(trade_date, self._available_after_close, SHANGHAI_TIMEZONE)

    def _daily_bar_from_row(self, row: Mapping[str, object], available_time: datetime) -> DailyBar:
        return DailyBar(
            trade_date=parse_trade_date(row["trade_date"]),
            stock_id=str(row["ts_code"]),
            open_price=parse_decimal(row["open"], "open"),
            high_price=parse_decimal(row["high"], "high"),
            low_price=parse_decimal(row["low"], "low"),
            close_price=parse_decimal(row["close"], "close"),
            volume=parse_decimal(row["vol"], "vol"),
            amount=parse_decimal(row["amount"], "amount"),
            pre_close=parse_optional_decimal(row.get("pre_close"), "pre_close"),
            source_name=self.name,
            source_version="tushare-pro-rest",
            available_time=available_time,
            volume_unit="hand",
            amount_unit="thousand_cny",
        )

    def _canonicalize_top_inst_rows(
        self, rows: tuple[dict[str, object], ...], available_time: datetime
    ) -> tuple[DragonTigerRecord, ...]:
        """Deduplicate mirrored payloads before deriving directional institutional-flow events."""

        payloads: dict[tuple[object, ...], _TopInstPayload] = {}
        for row in rows:
            trade_date = parse_trade_date(row["trade_date"])
            stock_id = _normalize_a_share_or_none(row["ts_code"])
            if stock_id is None:
                continue
            payload_key = (
                trade_date,
                stock_id,
                _top_inst_raw_value(row.get("exalter")),
                _top_inst_raw_value(row.get("buy")),
                _top_inst_raw_value(row.get("sell")),
                _top_inst_raw_value(row.get("net_buy")),
                _top_inst_raw_value(row.get("reason")),
            )
            source_side = _top_inst_source_side(row.get("side"))
            existing_payload = payloads.get(payload_key)
            if existing_payload is not None:
                existing_payload.source_row_count += 1
                existing_payload.source_side_set.add(source_side)
                continue
            payloads[payload_key] = _TopInstPayload(
                trade_date=trade_date,
                stock_id=stock_id,
                seat_name=_top_inst_seat_name(row.get("exalter")),
                reason=_top_inst_reason(row.get("reason")),
                buy=parse_optional_decimal(row.get("buy"), "buy") or Decimal("0"),
                sell=parse_optional_decimal(row.get("sell"), "sell") or Decimal("0"),
                net_buy=parse_optional_decimal(row.get("net_buy"), "net_buy") or Decimal("0"),
                source_row_count=1,
                source_side_set={source_side},
            )

        aggregates: dict[_TopInstAggregateKey, _TopInstAggregate] = {}
        for payload in payloads.values():
            directional_amounts = _top_inst_directional_amounts(payload)
            for side, amount in directional_amounts:
                seat_id, identity_scope = _top_inst_identity(
                    payload.seat_name,
                    payload.trade_date,
                    payload.stock_id,
                    payload.reason,
                    side,
                )
                aggregate_key: _TopInstAggregateKey = (
                    payload.trade_date,
                    payload.stock_id,
                    payload.reason,
                    seat_id,
                    side,
                    identity_scope,
                )
                existing_aggregate = aggregates.get(aggregate_key)
                if existing_aggregate is None:
                    aggregates[aggregate_key] = _TopInstAggregate(
                        amount=amount,
                        source_row_count=payload.source_row_count,
                        source_side_set=set(payload.source_side_set),
                    )
                    continue
                existing_aggregate.amount += amount
                existing_aggregate.source_row_count += payload.source_row_count
                existing_aggregate.source_side_set.update(payload.source_side_set)

        events: list[DragonTigerRecord] = []
        for key in sorted(
            aggregates,
            key=lambda item: (
                str(item[0]),
                str(item[1]),
                str(item[2] or ""),
                item[4].value,
                str(item[3]),
            ),
        ):
            trade_date, stock_id, reason, seat_id, side, identity_scope = key
            aggregate = aggregates[key]
            events.append(
                DragonTigerRecord(
                    trade_date=trade_date,
                    seat_id=seat_id,
                    stock_id=stock_id,
                    side=side,
                    amount=float(aggregate.amount),
                    event_reason=reason,
                    available_time=available_time,
                    source_name=self.name,
                    source_version="top_inst",
                    source_row_count=aggregate.source_row_count,
                    source_side_set=tuple(aggregate.source_side_set),
                    identity_scope=identity_scope,
                )
            )
        return tuple(events)


def _top_inst_raw_value(value: object | None) -> str:
    """Preserve the raw JSON representation so only exact payload mirrors collapse."""

    return f"{type(value).__name__}:{value!s}"


def _top_inst_source_side(value: object | None) -> str:
    normalized = str(value or "").strip()
    return normalized or "unspecified"


def _top_inst_seat_name(value: object | None) -> str:
    normalized = str(value or "").strip()
    return normalized or _TUSHARE_ANONYMOUS_INSTITUTION_LABEL


def _top_inst_reason(value: object | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _top_inst_directional_amounts(
    payload: _TopInstPayload,
) -> tuple[tuple[TradeSide, Decimal], ...]:
    explicit = tuple(
        (side, amount)
        for side, amount in ((TradeSide.BUY, payload.buy), (TradeSide.SELL, payload.sell))
        if amount > 0
    )
    if explicit:
        return explicit
    if payload.net_buy == 0:
        return ()
    side = TradeSide.BUY if payload.net_buy > 0 else TradeSide.SELL
    return ((side, abs(payload.net_buy)),)


def _top_inst_identity(
    seat_name: str,
    trade_date: date,
    stock_id: str,
    reason: str | None,
    side: TradeSide,
) -> tuple[str, str]:
    if seat_name != _TUSHARE_ANONYMOUS_INSTITUTION_LABEL:
        return seat_name, NAMED_SEAT_IDENTITY_SCOPE
    reason_token = quote(
        f"value:{reason}" if reason is not None else "missing",
        safe="",
    )
    return (
        f"{_INSTITUTIONAL_FLOW_PREFIX}:{trade_date.isoformat()}:{stock_id}:"
        f"{reason_token}:{side.value}",
        INSTITUTIONAL_FLOW_PROXY_IDENTITY_SCOPE,
    )


class AkshareProvider:
    """Optional free provider. It is a fallback, not a stable seat-detail source."""

    name = "akshare"

    def __init__(self, available_after_close: time = time(18, 0)) -> None:
        self._available_after_close = available_after_close

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            configured=_module_available("akshare"),
            supports_daily_bars=True,
            supports_dragon_tiger_summary=True,
            supports_seat_detail=False,
            seat_detail_scope="none",
            notes=(
                "Requires the optional akshare package.",
                "Free public-web sources can change; cache snapshots and validate fields.",
                "This adapter does not claim complete seat-level Dragon-Tiger details.",
            ),
        )

    def fetch_daily_bars(
        self, trade_date: date, stock_id: str | None = None
    ) -> tuple[DailyBar, ...]:
        if not stock_id:
            raise ProviderCapabilityError("AkShare daily ingestion requires --stock-code")
        ak = self._load_akshare()
        canonical_stock_id = normalize_stock_id(stock_id)
        symbol = canonical_stock_id.split(".")[0]
        try:
            frame = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=trade_date.strftime("%Y%m%d"),
                end_date=trade_date.strftime("%Y%m%d"),
                adjust="",
            )
        except Exception as error:
            raise ProviderResponseError(
                "AkShare daily request failed; check network, proxy, or source availability"
            ) from error
        available_time = datetime.combine(
            trade_date, self._available_after_close, SHANGHAI_TIMEZONE
        )
        return tuple(
            DailyBar(
                trade_date=parse_trade_date(_row_value(row, "日期", "date")),
                stock_id=canonical_stock_id,
                open_price=parse_decimal(_row_value(row, "开盘", "open"), "open"),
                high_price=parse_decimal(_row_value(row, "最高", "high"), "high"),
                low_price=parse_decimal(_row_value(row, "最低", "low"), "low"),
                close_price=parse_decimal(_row_value(row, "收盘", "close"), "close"),
                volume=parse_decimal(_row_value(row, "成交量", "volume"), "volume"),
                amount=parse_decimal(_row_value(row, "成交额", "amount"), "amount"),
                source_name=self.name,
                source_version="akshare",
                available_time=available_time,
                volume_unit="provider_reported",
                amount_unit="provider_reported",
            )
            for row in frame.to_dict(orient="records")
        )

    def fetch_dragon_tiger_summaries(self, trade_date: date) -> tuple[DragonTigerSummary, ...]:
        ak = self._load_akshare()
        try:
            frame = ak.stock_lhb_detail_em(
                start_date=trade_date.strftime("%Y%m%d"),
                end_date=trade_date.strftime("%Y%m%d"),
            )
        except Exception as error:
            raise ProviderResponseError(
                "AkShare Dragon-Tiger request failed; check network, proxy, or source availability"
            ) from error
        available_time = datetime.combine(
            trade_date, self._available_after_close, SHANGHAI_TIMEZONE
        )
        summaries: list[DragonTigerSummary] = []
        for row in frame.to_dict(orient="records"):
            row_date = parse_trade_date(_row_value(row, "上榜日", "trade_date", "日期"))
            summaries.append(
                DragonTigerSummary(
                    trade_date=row_date,
                    stock_id=_to_ts_code(str(_row_value(row, "代码", "stock_code", "ts_code"))),
                    stock_name=str(_row_value(row, "名称", "name", default="")),
                    buy_amount=parse_decimal(
                        _row_value(row, "龙虎榜买入额", "l_buy", default=0), "l_buy"
                    ).quantize(Decimal("0.01")),
                    sell_amount=parse_decimal(
                        _row_value(row, "龙虎榜卖出额", "l_sell", default=0), "l_sell"
                    ).quantize(Decimal("0.01")),
                    net_amount=parse_decimal(
                        _row_value(row, "龙虎榜净买额", "net_amount", default=0), "net_amount"
                    ).quantize(Decimal("0.01")),
                    reason=str(_row_value(row, "上榜原因", "reason", default="")),
                    source_name=self.name,
                    source_version="akshare",
                    available_time=available_time,
                )
            )
        return tuple(summaries)

    def fetch_dragon_tiger_seat_events(self, trade_date: date) -> tuple[DragonTigerRecord, ...]:
        raise ProviderCapabilityError(
            "AkShare fallback does not provide a verified complete seat-detail adapter. "
            "Use Tushare top_inst or import a separately licensed seat-detail dataset."
        )

    def fetch_suspension_evidence(self, trade_date: date) -> SuspensionResponseEvidence:
        raise ProviderCapabilityError(
            "AkShare cannot provide auditable Tushare suspend_d S-state evidence; "
            "refusing to infer it."
        )

    @staticmethod
    def _load_akshare() -> Any:
        try:
            import akshare as ak  # type: ignore[import-untyped]
        except ModuleNotFoundError as error:
            raise ProviderNotConfiguredError(
                "AkShare is not installed. Install the optional data extra: pip install -e .[data]"
            ) from error
        return ak


class SampleProvider:
    """Deterministic, dependency-free source for end-to-end local validation."""

    name = "sample"

    def __init__(self, available_after_close: time = time(18, 0)) -> None:
        self._available_after_close = available_after_close

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.name,
            configured=True,
            supports_daily_bars=True,
            supports_dragon_tiger_summary=True,
            supports_seat_detail=True,
            seat_detail_scope="synthetic",
            notes=("Synthetic records only; never use them for research claims or trading.",),
        )

    def fetch_daily_bars(
        self, trade_date: date, stock_id: str | None = None
    ) -> tuple[DailyBar, ...]:
        code = normalize_stock_id(stock_id) if stock_id else "000001.SZ"
        available_time = datetime.combine(
            trade_date, self._available_after_close, SHANGHAI_TIMEZONE
        )
        return (
            DailyBar(
                trade_date=trade_date,
                stock_id=code,
                open_price=Decimal("10.00"),
                high_price=Decimal("10.80"),
                low_price=Decimal("9.90"),
                close_price=Decimal("10.50"),
                volume=Decimal("1200000"),
                amount=Decimal("12600000"),
                source_name=self.name,
                source_version="v1",
                available_time=available_time,
                pre_close=Decimal("9.95"),
            ),
        )

    def fetch_dragon_tiger_summaries(self, trade_date: date) -> tuple[DragonTigerSummary, ...]:
        available_time = datetime.combine(
            trade_date, self._available_after_close, SHANGHAI_TIMEZONE
        )
        return (
            DragonTigerSummary(
                trade_date=trade_date,
                stock_id="000001.SZ",
                stock_name="Sample Bank",
                buy_amount=Decimal("12000000"),
                sell_amount=Decimal("4000000"),
                net_amount=Decimal("8000000"),
                reason="sample_daily_deviation",
                source_name=self.name,
                source_version="v1",
                available_time=available_time,
            ),
        )

    def fetch_dragon_tiger_seat_events(self, trade_date: date) -> tuple[DragonTigerRecord, ...]:
        available_time = datetime.combine(
            trade_date, self._available_after_close, SHANGHAI_TIMEZONE
        )
        return (
            DragonTigerRecord(
                trade_date=trade_date,
                seat_id="sample_institution_a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=12_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="sample_daily_deviation",
                available_time=available_time,
                source_name=self.name,
                source_version="v1",
            ),
            DragonTigerRecord(
                trade_date=trade_date,
                seat_id="sample_arbitrage_b",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=4_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="sample_daily_deviation",
                available_time=available_time,
                source_name=self.name,
                source_version="v1",
            ),
        )

    def fetch_suspension_evidence(self, trade_date: date) -> SuspensionResponseEvidence:
        raise ProviderCapabilityError(
            "SampleProvider cannot provide real suspend_d S-state evidence; "
            "refusing synthetic tradability."
        )


def create_provider(name: str, available_after_close: time = time(18, 0)) -> MarketDataProvider:
    normalized = name.strip().lower()
    if normalized == "sample":
        return SampleProvider(available_after_close=available_after_close)
    if normalized == "tushare":
        return TushareHttpProvider(available_after_close=available_after_close)
    if normalized == "akshare":
        return AkshareProvider(available_after_close=available_after_close)
    raise ValueError(f"unsupported provider: {name}")


def _normalize_a_share_or_none(value: object) -> str | None:
    """Return a canonical A-share code and skip non-equity Dragon-Tiger instruments."""

    try:
        return normalize_stock_id(value)
    except ValueError:
        # Dragon-Tiger endpoints can include convertible bonds and other instruments.
        # They have a Tushare ts_code but are outside the project's A-share contract.
        return None


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> Mapping[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise ProviderResponseError(f"Tushare HTTP status {error.code}") from error
    except URLError as error:
        raise ProviderResponseError("Tushare network request failed") from error
    except TimeoutError as error:
        raise ProviderResponseError("Tushare network request timed out") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderResponseError("Tushare returned invalid JSON") from error
    if not isinstance(decoded, Mapping):
        raise ProviderResponseError("Tushare returned a non-object response")
    return decoded


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except ModuleNotFoundError:
        return False
    return True


def _row_value(
    row: Mapping[str, object], *candidates: str, default: object | None = None
) -> object:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    if default is not None:
        return default
    names = ", ".join(candidates)
    raise ProviderResponseError(f"provider row does not contain any of: {names}")


def _to_ts_code(code: str) -> str:
    return normalize_stock_id(code)
