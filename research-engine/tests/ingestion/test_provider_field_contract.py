"""Fail-closed tests for provider field maps and capability evidence."""

from __future__ import annotations

from typing import Any

import pytest

from seatflow.ingestion.models import SourceConfidence
from seatflow.ingestion.provider import ProviderCapability
from seatflow.ingestion.providers.akshare import AkshareProvider
from seatflow.ingestion.providers.file_provider import FileProvider
from seatflow.ingestion.providers.tushare import TushareProvider


def _transport(rows: list[dict[str, Any]]):
    def transport(method: str, **kwargs: Any) -> list[dict[str, Any]]:
        return rows

    return transport


def test_akshare_accepts_explicit_utf8_mojibake_aliases() -> None:
    """Removing the explicit mojibake alias map must reject recoverable provider rows."""
    row = {
        "\u00e6\u0097\u00a5\u00e6\u009c\u009f": "2026-08-05",  # 日期 decoded as latin-1
        "\u00e4\u00bb\u00a3\u00e7\u00a0\u0081": "000001",  # 代码 decoded as latin-1
        "收盘": 10.5,
    }
    provider = AkshareProvider(transport=_transport([row]))

    records = list(provider.fetch_daily_bars())

    assert records[0].event_time.isoformat() == "2026-08-05T00:00:00+00:00"
    assert records[0].raw_payload["日期"] == "2026-08-05"
    evidence = provider.capability_evidence()["daily_bars"]
    assert evidence.status == ProviderCapability.AVAILABLE
    assert evidence.field_coverage == 1.0


def test_configurable_aliases_are_canonicalized_before_parsing() -> None:
    """Ignoring caller-provided aliases must make this otherwise valid row fail."""
    row = {"交易日期": "2026-08-05", "证券代码": "000001", "收盘": 10.5}
    provider = AkshareProvider(
        transport=_transport([row]),
        field_aliases={
            "event_time": ("交易日期",),
            "stock_code": ("证券代码",),
        },
    )

    record = next(iter(provider.fetch_daily_bars()))

    assert record.raw_payload["日期"] == "2026-08-05"
    assert record.raw_payload["代码"] == "000001"


def test_missing_required_provider_field_fails_closed_with_coverage() -> None:
    """Falling back to str(None) must no longer produce a Bronze record."""
    provider = AkshareProvider(transport=_transport([{"代码": "000001", "收盘": 10.5}]))

    with pytest.raises(ValueError, match="MISSING_REQUIRED_FIELDS") as exc_info:
        list(provider.fetch_daily_bars())

    assert getattr(exc_info.value, "reason_code") == "MISSING_REQUIRED_FIELDS"
    assert provider.capability()["daily_bars"] == ProviderCapability.DEGRADED
    evidence = provider.capability_evidence()["daily_bars"]
    assert evidence.missing_fields == ("event_time",)
    assert evidence.field_coverage == 0.5


def test_tushare_top_inst_is_machine_readable_proxy_only_degradation() -> None:
    """Treating top_inst as complete real seat detail must fail this contract."""
    row = {
        "ts_code": "000001.SZ",
        "trade_date": "20260805",
        "exalter": "机构专用",
        "buy": 1_000_000.0,
    }
    provider = TushareProvider(transport=_transport([row]))

    records = list(provider.fetch_lhb_participation())

    assert len(records) == 1
    assert records[0].source_confidence == SourceConfidence.DEGRADED
    assert provider.capability()["lhb_participation"] == ProviderCapability.DEGRADED
    evidence = provider.capability_evidence()["lhb_participation"]
    assert evidence.reason_code == "PROXY_COVERAGE_ONLY"
    assert evidence.proxy_only is True
    assert evidence.details["endpoint"] == "top_inst"
    assert evidence.details["formal_use_allowed"] is False


def test_tushare_missing_date_does_not_silently_parse_none() -> None:
    """Deleting required-field validation must reintroduce the None parsing bug."""
    provider = TushareProvider(transport=_transport([{"ts_code": "000001.SZ"}]))

    with pytest.raises(ValueError, match="MISSING_REQUIRED_FIELDS"):
        list(provider.fetch_daily_bars())

    evidence = provider.capability_evidence()["daily_bars"]
    assert evidence.status == ProviderCapability.DEGRADED
    assert evidence.missing_fields == ("event_time",)


def test_file_provider_missing_time_column_fails_closed_with_coverage(tmp_path) -> None:
    """A present file must not stay AVAILABLE when required PIT columns are absent."""
    path = tmp_path / "daily.csv"
    path.write_text("event_time,close\n2026-08-05T00:00:00+00:00,10.5\n", encoding="utf-8")
    provider = FileProvider(base_dir=tmp_path)

    with pytest.raises(ValueError, match="MISSING_REQUIRED_FIELDS"):
        list(provider.fetch_daily_bars("daily.csv"))

    assert provider.capability()["daily.csv"] == ProviderCapability.DEGRADED
    evidence = provider.capability_evidence()["daily.csv"]
    assert evidence.missing_fields == ("available_time",)
    assert evidence.field_coverage == 0.5
