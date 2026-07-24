"""Build a source-hashed V2 label candidate and fail closed on missing market status."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from seatflow.backtest.trading_rules import price_limit_pct
from seatflow.data.contracts import DailyBar, parse_trade_date
from seatflow.data.providers import ProviderError, TushareHttpProvider
from seatflow.labels import (
    ExecutablePriceEvidence,
    MarketDispositionEvidence,
    executable_open_evidence_from_daily_bar,
    generate_executable_excess_return_labels_v2,
)
from seatflow.runtime_env import load_workspace_env

OUTPUT = Path("artifacts/diagnostics/R0_V2_CANDIDATE_DATASET/result.json")
STOCK_ID = "000001.SZ"
BENCHMARK_ID = "000300.SH"
START = date(2024, 1, 2)
END = date(2024, 1, 16)


def main() -> None:
    load_workspace_env()
    provider = TushareHttpProvider()
    if not provider.status().configured:
        raise RuntimeError("R0 V2 candidate requires configured Tushare read-only access")
    calendar = provider.fetch_trading_calendar(START, END)
    bars = provider.fetch_daily_bar_history(STOCK_ID, START, END)
    factors = provider.fetch_adjustment_factor_evidence(STOCK_ID, START, END)
    index_opens = provider.fetch_index_open_history(BENCHMARK_ID, START, END)
    if not bars or bars[0].available_time is None:
        raise RuntimeError("R0 V2 candidate requires timestamped daily bars")
    suspension_probe: dict[str, object]
    try:
        suspension = provider.fetch_suspension_evidence(calendar[1])
        suspension_probe = {
            "query_sha256": suspension.daily_query_sha256,
            "record_count": len(suspension.records),
            "stock_ids": [record.stock_id for record in suspension.records],
        }
    except ProviderError as error:
        suspension_probe = {"error": f"{type(error).__name__}: {error}"}
    status_probe = provider.fetch_stock_status_evidence(STOCK_ID, START, END)
    market_status_hash = _market_status_hash(suspension_probe, status_probe)

    factors_by_date = {record.trade_date: record for record in factors.records}
    stock_prices = {
        (bar.stock_id, bar.trade_date): executable_open_evidence_from_daily_bar(
            bar, factors_by_date[bar.trade_date]
        )
        for bar in bars
        if bar.trade_date in factors_by_date
    }
    benchmark_prices = {
        (BENCHMARK_ID, trade_date): _benchmark_open_evidence(trade_date, open_price)
        for trade_date, open_price in index_opens
    }
    disposition = _market_disposition(calendar, bars, suspension_probe, status_probe)
    dispositions = (
        {(STOCK_ID, calendar[1]): disposition} if disposition is not None else {}
    )
    labels = generate_executable_excess_return_labels_v2(
        as_of_date=calendar[0],
        cutoff_time=datetime.combine(
            calendar[0], datetime.max.time(), tzinfo=bars[0].available_time.tzinfo
        ),
        target_stock_ids=(STOCK_ID,),
        horizons=(1, 3, 5),
        trading_calendar=calendar,
        benchmark_by_stock={STOCK_ID: BENCHMARK_ID},
        stock_open_prices=stock_prices,
        benchmark_open_prices=benchmark_prices,
        dispositions=dispositions,
        market_status_hashes={STOCK_ID: market_status_hash},
    )
    payload = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "label_schema_version": "executable_excess_return_v2",
        "training_eligible_sample_count": sum(
            label.disposition == "eligible" for label in labels
        ),
        "candidate_sample_count": len(labels),
        "calendar_hash": sha256(
            "|".join(item.isoformat() for item in calendar).encode()
        ).hexdigest(),
        "adjustment_query_sha256": factors.query_sha256,
        "suspension_probe": suspension_probe,
        "status_probe": status_probe,
        "market_status_sha256": market_status_hash,
        "labels": [label.to_dict() for label in labels],
        "reason": None if disposition is None else disposition.disposition,
    }
    payload["sha256"] = sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {"result": str(OUTPUT), "eligible": payload["training_eligible_sample_count"]}
        )
    )


def _benchmark_open_evidence(trade_date: date, open_price: Decimal) -> ExecutablePriceEvidence:
    value = str(open_price)
    return ExecutablePriceEvidence(
        trade_date=trade_date,
        price=open_price,
        adjustment_factor=Decimal("1"),
        available_time=datetime.combine(trade_date, datetime.max.time()).astimezone(),
        source_hash=sha256(
            json.dumps(
                {"benchmark_id": BENCHMARK_ID, "trade_date": trade_date.isoformat(), "open": value},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    )


def _market_disposition(
    calendar: tuple[date, ...],
    bars: tuple[DailyBar, ...],
    suspension_probe: dict[str, object],
    status_probe: dict[str, object],
) -> MarketDispositionEvidence | None:
    evidence_hash = _market_status_hash(suspension_probe, status_probe)
    if "error" in suspension_probe:
        return MarketDispositionEvidence("missing_market_status", evidence_hash)
    suspended_stock_ids = suspension_probe.get("stock_ids")
    if isinstance(suspended_stock_ids, list) and STOCK_ID in suspended_stock_ids:
        return MarketDispositionEvidence("suspended", evidence_hash)
    raw_names = status_probe.get("namechange")
    names = raw_names if isinstance(raw_names, list) else []
    if any("ST" in str(row.get("name", "")).upper() for row in names if isinstance(row, dict)):
        return MarketDispositionEvidence("st_restricted", evidence_hash)
    raw_basic = status_probe.get("stock_basic")
    basic = raw_basic if isinstance(raw_basic, list) else []
    first_basic = basic[0] if basic and isinstance(basic[0], dict) else None
    if first_basic is None or not str(first_basic.get("list_date", "")):
        return MarketDispositionEvidence("missing_market_status", evidence_hash)
    listing_date = parse_trade_date(str(first_basic["list_date"]))
    if (calendar[1] - listing_date).days < 60:
        return MarketDispositionEvidence("new_stock_restricted", evidence_hash)
    bar_by_date = {bar.trade_date: bar for bar in bars}
    for trade_date in calendar[1:7]:
        bar = bar_by_date.get(trade_date)
        if bar is None or bar.pre_close is None:
            return MarketDispositionEvidence("missing_market_status", evidence_hash)
        limit = price_limit_pct(bar.stock_id, Decimal("0.10"))
        upper_limit = bar.pre_close * (Decimal("1") + limit)
        lower_limit = bar.pre_close * (Decimal("1") - limit)
        if bar.open_price >= upper_limit or bar.open_price <= lower_limit:
            return MarketDispositionEvidence("price_limit_blocked", evidence_hash)
    return None


def _market_status_hash(
    suspension_probe: dict[str, object], status_probe: dict[str, object]
) -> str:
    """Bind every eligibility source, including a clear suspension result, to labels."""

    return sha256(
        json.dumps(
            {"suspension": suspension_probe, "status": status_probe},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


if __name__ == "__main__":
    main()
