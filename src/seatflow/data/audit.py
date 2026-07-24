from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from seatflow.data.contracts import DailyBar, DragonTigerSummary
from seatflow.data.providers import MarketDataProvider, ProviderError, ProviderStatus
from seatflow.data.records import DragonTigerRecord

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(token\s*[=:]\s*)[^\s,}\]]+"),
    re.compile(r'(?i)("token"\s*:\s*")[^"]+"'),
)
_ALLOWED_SCOPES = frozenset({"complete_seat", "institutional_only", "synthetic", "none"})


@dataclass(frozen=True)
class DataAuditPolicy:
    minimum_trade_days: int = 720
    minimum_daily_coverage: float = 0.995
    minimum_summary_coverage: float = 1.0
    minimum_seat_detail_coverage: float = 1.0
    require_nonempty_daily: bool = True
    require_nonempty_summary: bool = True


DEFAULT_AUDIT_POLICY = DataAuditPolicy()


@dataclass(frozen=True)
class EndpointAudit:
    endpoint: str
    requested_dates: int
    successful_dates: int
    row_count: int
    empty_dates: tuple[str, ...]
    failed_dates: tuple[str, ...]
    errors: tuple[str, ...]
    missing_available_time: int

    @property
    def coverage_ratio(self) -> float:
        if self.requested_dates == 0:
            return 0.0
        return self.successful_dates / self.requested_dates

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "requested_dates": self.requested_dates,
            "successful_dates": self.successful_dates,
            "coverage_ratio": self.coverage_ratio,
            "row_count": self.row_count,
            "empty_dates": list(self.empty_dates),
            "failed_dates": list(self.failed_dates),
            "errors": list(self.errors),
            "missing_available_time": self.missing_available_time,
        }


@dataclass(frozen=True)
class DataAuditReport:
    provider_status: ProviderStatus
    start_date: date
    end_date: date
    expected_trade_dates: tuple[date, ...]
    daily: EndpointAudit
    dragon_tiger_summary: EndpointAudit
    seat_detail: EndpointAudit | None
    seat_detail_scope: str
    data_scope: str
    require_seat_detail: bool
    data_readiness: bool
    full_seat_ready: bool
    blocked_reasons: tuple[str, ...]
    generated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_status": self.provider_status.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "expected_trade_dates": [item.isoformat() for item in self.expected_trade_dates],
            "calendar_verified": True,
            "daily": self.daily.to_dict(),
            "dragon_tiger_summary": self.dragon_tiger_summary.to_dict(),
            "seat_detail": self.seat_detail.to_dict() if self.seat_detail else None,
            "seat_detail_scope": self.seat_detail_scope,
            "data_scope": self.data_scope,
            "require_seat_detail": self.require_seat_detail,
            "data_readiness": self.data_readiness,
            "full_seat_ready": self.full_seat_ready,
            "leakage_assumptions": {
                "timezone": "Asia/Shanghai",
                "available_time_required": True,
                "source_publication_time": "must be independently verified before training",
            },
            "blocked_reasons": list(self.blocked_reasons),
            "generated_at": self.generated_at.isoformat(),
        }


def audit_market_data(
    provider: MarketDataProvider,
    expected_trade_dates: Sequence[date],
    *,
    require_seat_detail: bool,
    policy: DataAuditPolicy = DEFAULT_AUDIT_POLICY,
) -> DataAuditReport:
    """Exercise live provider endpoints and return an evidence-only readiness decision."""

    trade_dates = _validate_trade_dates(expected_trade_dates)
    provider_status = provider.status()
    daily = _audit_endpoint(trade_dates, provider.fetch_daily_bars, "daily")
    summaries = _audit_endpoint(
        trade_dates, provider.fetch_dragon_tiger_summaries, "dragon_tiger_summary"
    )
    seat_detail = (
        _audit_endpoint(trade_dates, provider.fetch_dragon_tiger_seat_events, "seat_detail")
        if require_seat_detail
        else None
    )

    scope = _observed_scope(provider_status, seat_detail)
    blocked_reasons = _blocked_reasons(
        provider_status,
        daily,
        summaries,
        seat_detail,
        scope,
        require_seat_detail,
        policy,
        len(trade_dates),
    )
    data_readiness = not blocked_reasons
    data_scope = _data_scope(scope, data_readiness)
    return DataAuditReport(
        provider_status=provider_status,
        start_date=trade_dates[0],
        end_date=trade_dates[-1],
        expected_trade_dates=trade_dates,
        daily=daily,
        dragon_tiger_summary=summaries,
        seat_detail=seat_detail,
        seat_detail_scope=scope,
        data_scope=data_scope,
        require_seat_detail=require_seat_detail,
        data_readiness=data_readiness,
        full_seat_ready=data_readiness and scope == "complete_seat",
        blocked_reasons=tuple(blocked_reasons),
        generated_at=datetime.now().astimezone(),
    )


def write_audit_report(path: str | Path, report: DataAuditReport) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    with NamedTemporaryFile("wb", delete=False, dir=output_path.parent, prefix=".audit.") as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    temporary_path.replace(output_path)
    return output_path


def load_audit_policy(path: str | Path) -> DataAuditPolicy:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("data-audit config must contain a [coverage] table")
    try:
        policy = DataAuditPolicy(
            minimum_trade_days=int(coverage["minimum_trade_days"]),
            minimum_daily_coverage=float(coverage["minimum_daily_coverage"]),
            minimum_summary_coverage=float(coverage["minimum_summary_coverage"]),
            minimum_seat_detail_coverage=float(coverage["minimum_seat_detail_coverage"]),
            require_nonempty_daily=bool(coverage["require_nonempty_daily"]),
            require_nonempty_summary=bool(coverage["require_nonempty_summary"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid [coverage] values in data-audit config") from error
    if policy.minimum_trade_days < 1:
        raise ValueError("minimum_trade_days must be positive")
    for value in (
        policy.minimum_daily_coverage,
        policy.minimum_summary_coverage,
        policy.minimum_seat_detail_coverage,
    ):
        if not 0 < value <= 1:
            raise ValueError("coverage thresholds must be in (0, 1]")
    return policy


def _validate_trade_dates(values: Sequence[date]) -> tuple[date, ...]:
    if not values:
        raise ValueError("expected_trade_dates must not be empty")
    normalized = tuple(sorted(set(values)))
    if len(normalized) != len(values):
        raise ValueError("expected_trade_dates must not contain duplicates")
    return normalized


def _audit_endpoint(
    trade_dates: Iterable[date],
    fetcher: object,
    endpoint: str,
) -> EndpointAudit:
    successful_dates = 0
    row_count = 0
    missing_available_time = 0
    empty_dates: list[str] = []
    failed_dates: list[str] = []
    errors: list[str] = []
    dates = tuple(trade_dates)

    for trade_date in dates:
        try:
            rows = _call_fetcher(fetcher, trade_date)
        except ProviderError as error:
            failed_dates.append(trade_date.isoformat())
            errors.append(_safe_error(error))
            continue
        except Exception as error:  # Defensive boundary for third-party provider adapters.
            failed_dates.append(trade_date.isoformat())
            errors.append(f"unexpected {type(error).__name__}: {_safe_error(error)}")
            continue

        successful_dates += 1
        row_count += len(rows)
        if not rows:
            empty_dates.append(trade_date.isoformat())
        missing_available_time += sum(
            1 for row in rows if getattr(row, "available_time", None) is None
        )

    return EndpointAudit(
        endpoint=endpoint,
        requested_dates=len(dates),
        successful_dates=successful_dates,
        row_count=row_count,
        empty_dates=tuple(empty_dates),
        failed_dates=tuple(failed_dates),
        errors=tuple(errors),
        missing_available_time=missing_available_time,
    )


def _call_fetcher(
    fetcher: object, trade_date: date
) -> tuple[DailyBar | DragonTigerSummary | DragonTigerRecord, ...]:
    if not callable(fetcher):
        raise TypeError("provider endpoint is not callable")
    rows = fetcher(trade_date)
    return tuple(rows)


def _observed_scope(status: ProviderStatus, seat_detail: EndpointAudit | None) -> str:
    if seat_detail is None or seat_detail.successful_dates == 0:
        return "none"
    scope = status.seat_detail_scope.strip().lower()
    if scope in _ALLOWED_SCOPES:
        return scope
    return "none"


def _data_scope(seat_detail_scope: str, data_readiness: bool) -> str:
    if seat_detail_scope == "synthetic":
        return "synthetic"
    if data_readiness and seat_detail_scope in {"complete_seat", "institutional_only"}:
        return "real"
    return "none"


def _blocked_reasons(
    status: ProviderStatus,
    daily: EndpointAudit,
    summaries: EndpointAudit,
    seat_detail: EndpointAudit | None,
    scope: str,
    require_seat_detail: bool,
    policy: DataAuditPolicy,
    trade_day_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not status.configured:
        reasons.append("provider is not configured for a live audit")
    if trade_day_count < policy.minimum_trade_days:
        reasons.append(
            "expected trade dates "
            f"({trade_day_count}) are below minimum_trade_days={policy.minimum_trade_days}"
        )
    if daily.coverage_ratio < policy.minimum_daily_coverage:
        reasons.append("daily endpoint coverage is below the configured threshold")
    if policy.require_nonempty_daily and daily.empty_dates:
        reasons.append("daily returned empty data for expected trade dates")
    if daily.missing_available_time:
        reasons.append("daily contains rows without available_time")
    if summaries.coverage_ratio < policy.minimum_summary_coverage:
        reasons.append("dragon_tiger_summary endpoint coverage is below the configured threshold")
    if policy.require_nonempty_summary and summaries.empty_dates:
        reasons.append("dragon_tiger_summary returned empty data for expected trade dates")
    if summaries.missing_available_time:
        reasons.append("dragon_tiger_summary contains rows without available_time")
    if require_seat_detail:
        if seat_detail is None or seat_detail.coverage_ratio < policy.minimum_seat_detail_coverage:
            reasons.append("seat-detail endpoint coverage is below the configured threshold")
        elif seat_detail.missing_available_time:
            reasons.append("seat-detail records contain missing available_time")
        if scope in {"none", "synthetic"}:
            reasons.append(f"seat_detail_scope={scope} cannot support a real SeatHGT claim")
    if scope == "synthetic":
        reasons.append("synthetic provider data cannot establish real-data readiness")
    return reasons


def _safe_error(error: BaseException) -> str:
    message = str(error)
    for pattern in _TOKEN_PATTERNS:
        message = pattern.sub(r"\1<redacted>", message)
    return message[:240]
