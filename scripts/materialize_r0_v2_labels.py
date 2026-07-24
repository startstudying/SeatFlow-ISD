"""Materialize source-hashed executable-label V2 records for a historical event date.

The output is deliberately fail-closed: it records non-tradable labels but never
silently substitutes raw prices, missing status, or incomplete adjustment data.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from seatflow.backtest.trading_rules import price_limit_pct
from seatflow.data.contracts import AdjustmentFactorEvidence, DailyBar, parse_trade_date
from seatflow.data.providers import SHANGHAI_TIMEZONE, ProviderError, TushareHttpProvider
from seatflow.labels import (
    ExecutablePriceEvidence,
    MarketDispositionEvidence,
    StockLabelV2,
    executable_open_evidence_from_daily_bar,
    generate_executable_excess_return_labels_v2,
)
from seatflow.runtime_env import load_workspace_env

BENCHMARK_ID = "000300.SH"
HORIZONS = (1, 3, 5)


def main() -> None:
    args = _arguments()
    event_path = Path(args.event_snapshot)
    targets = _event_targets(event_path)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if not targets:
        raise RuntimeError("no target stocks were found in the event snapshot")

    load_workspace_env()
    provider = TushareHttpProvider()
    if not provider.status().configured:
        raise RuntimeError("R0 V2 materialization requires configured Tushare read-only access")
    calendar = provider.fetch_trading_calendar(args.as_of, args.as_of + timedelta(days=21))
    if len(calendar) < 7 or calendar[0] != args.as_of:
        raise RuntimeError("materialization requires as-of plus six following trading days")
    calendar = calendar[:7]
    observed_calendar = tuple(
        trade_date
        for trade_date in calendar
        if args.observed_label_end is None or trade_date <= args.observed_label_end
    )
    benchmark_prices = _benchmark_prices(provider, observed_calendar)
    suspension_probes = _suspension_probes(provider, observed_calendar[1:])

    labels: list[StockLabelV2] = []
    target_results: list[dict[str, object]] = []
    for stock_id in targets:
        generated, result = _materialize_target(
            provider,
            stock_id,
            calendar,
            observed_calendar,
            benchmark_prices,
            suspension_probes,
            args.observed_label_end,
        )
        labels.extend(generated)
        target_results.append(result)
    payload: dict[str, object] = {
        "status": "evaluated_fail_closed",
        "model_valid": False,
        "label_schema_version": "executable_excess_return_v2",
        "as_of_date": args.as_of.isoformat(),
        "benchmark_id": BENCHMARK_ID,
        "horizons": list(HORIZONS),
        "source_event_snapshot": str(event_path),
        "source_event_snapshot_sha256": _file_sha256(event_path),
        "calendar": [item.isoformat() for item in calendar],
        "calendar_hash": _hash_values(item.isoformat() for item in calendar),
        "observed_label_end": (
            None if args.observed_label_end is None else args.observed_label_end.isoformat()
        ),
        "target_count": len(targets),
        "candidate_sample_count": len(labels),
        "training_eligible_sample_count": sum(
            label.disposition == "eligible" for label in labels
        ),
        "labels": [label.to_dict() for label in labels],
        "target_results": target_results,
    }
    payload["sha256"] = sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "targets": len(targets),
                "eligible_labels": payload["training_eligible_sample_count"],
            }
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--event-snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--observed-label-end",
        type=date.fromisoformat,
        help="latest date whose price and market-status evidence may enter training labels",
    )
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error("--max-targets must be positive")
    return args


def _event_targets(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("event snapshot must contain a JSON array")
    targets = {
        str(row["stock_id"])
        for row in raw
        if isinstance(row, dict) and isinstance(row.get("stock_id"), str)
    }
    return tuple(sorted(targets))


def _benchmark_prices(
    provider: TushareHttpProvider, calendar: tuple[date, ...]
) -> dict[tuple[str, date], ExecutablePriceEvidence]:
    if len(calendar) < 2:
        return {}
    rows = provider.fetch_index_open_history(BENCHMARK_ID, calendar[1], calendar[-1])
    return {
        (BENCHMARK_ID, trade_date): ExecutablePriceEvidence(
            trade_date=trade_date,
            price=open_price,
            adjustment_factor=Decimal("1"),
            available_time=datetime.combine(
                trade_date, datetime.max.time(), SHANGHAI_TIMEZONE
            ),
            source_hash=_hash_json(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "trade_date": trade_date.isoformat(),
                    "open": str(open_price),
                }
            ),
        )
        for trade_date, open_price in rows
    }


def _suspension_probes(
    provider: TushareHttpProvider, dates: tuple[date, ...]
) -> dict[date, dict[str, object]]:
    probes: dict[date, dict[str, object]] = {}
    for trade_date in dates:
        try:
            response = provider.fetch_suspension_evidence(trade_date)
            probes[trade_date] = {
                "query_sha256": response.daily_query_sha256,
                "stock_ids": sorted(record.stock_id for record in response.records),
            }
        except ProviderError as error:
            probes[trade_date] = {"error": f"{type(error).__name__}: {error}"}
    return probes


def _materialize_target(
    provider: TushareHttpProvider,
    stock_id: str,
    calendar: tuple[date, ...],
    observed_calendar: tuple[date, ...],
    benchmark_prices: dict[tuple[str, date], ExecutablePriceEvidence],
    suspension_probes: dict[date, dict[str, object]],
    observed_label_end: date | None,
) -> tuple[tuple[StockLabelV2, ...], dict[str, object]]:
    start = calendar[0]
    end = observed_calendar[-1] if observed_calendar else start
    errors: list[str] = []
    try:
        bars = provider.fetch_daily_bar_history(stock_id, start, end)
    except ProviderError as error:
        bars = ()
        errors.append(f"daily:{type(error).__name__}")
    try:
        factors = provider.fetch_adjustment_factor_evidence(stock_id, start, end)
    except ProviderError as error:
        factors = None
        errors.append(f"adj_factor:{type(error).__name__}")
    try:
        status_probe = provider.fetch_stock_status_evidence(stock_id, start, end)
    except ProviderError as error:
        status_probe = {"error": f"{type(error).__name__}: {error}"}
        errors.append(f"status:{type(error).__name__}")
    market_hash = _hash_json(
        {
            "suspension": {
                trade_date.isoformat(): probe
                for trade_date, probe in sorted(suspension_probes.items())
            },
            "status": status_probe,
        }
    )
    disposition = (
        None
        if len(observed_calendar) < 2
        else _market_disposition(
            stock_id, observed_calendar, bars, suspension_probes, status_probe, market_hash
        )
    )
    factors_by_date = {} if factors is None else {item.trade_date: item for item in factors.records}
    stock_prices = {
        (stock_id, bar.trade_date): _stock_price_evidence(bar, factors_by_date.get(bar.trade_date))
        for bar in bars
    }
    cutoff = datetime.combine(calendar[0], datetime.max.time(), SHANGHAI_TIMEZONE)
    dispositions = {} if disposition is None else {
        (stock_id, trade_date): disposition for trade_date in observed_calendar[1:]
    }
    if observed_label_end is not None:
        unobserved = MarketDispositionEvidence(
            "unobserved_horizon",
            _hash_json(
                {
                    "as_of_date": start.isoformat(),
                    "observed_label_end": observed_label_end.isoformat(),
                    "stock_id": stock_id,
                }
            ),
        )
        dispositions.update(
            {
                (stock_id, trade_date): unobserved
                for trade_date in calendar[1:]
                if trade_date > observed_label_end
            }
        )
    generated = generate_executable_excess_return_labels_v2(
        as_of_date=calendar[0],
        cutoff_time=cutoff,
        target_stock_ids=(stock_id,),
        horizons=HORIZONS,
        trading_calendar=calendar,
        benchmark_by_stock={stock_id: BENCHMARK_ID},
        stock_open_prices=stock_prices,
        benchmark_open_prices=benchmark_prices,
        dispositions=dispositions,
        market_status_hashes={stock_id: market_hash},
    )
    return generated, {
        "stock_id": stock_id,
        "status_hash": market_hash,
        "error_codes": errors,
        "disposition": None if disposition is None else disposition.disposition,
        "label_count": len(generated),
        "eligible_label_count": sum(label.disposition == "eligible" for label in generated),
    }


def _stock_price_evidence(
    bar: DailyBar, adjustment: AdjustmentFactorEvidence | None
) -> ExecutablePriceEvidence:
    if adjustment is not None:
        return executable_open_evidence_from_daily_bar(bar, adjustment)
    if bar.available_time is None:
        raise ValueError("materialized daily bar lacks available_time")
    return ExecutablePriceEvidence(
        trade_date=bar.trade_date,
        price=bar.open_price,
        adjustment_factor=None,
        available_time=bar.available_time,
        source_hash=_hash_json({"daily_bar": bar.to_dict(), "adjustment_factor": None}),
    )


def _market_disposition(
    stock_id: str,
    calendar: tuple[date, ...],
    bars: tuple[DailyBar, ...],
    suspension_probes: dict[date, dict[str, object]],
    status_probe: dict[str, object],
    evidence_hash: str,
) -> MarketDispositionEvidence | None:
    if any("error" in probe for probe in suspension_probes.values()) or "error" in status_probe:
        return MarketDispositionEvidence("missing_market_status", evidence_hash)
    if any(
        stock_id in stock_ids
        for probe in suspension_probes.values()
        for stock_ids in [probe.get("stock_ids")]
        if isinstance(stock_ids, list)
    ):
        return MarketDispositionEvidence("suspended", evidence_hash)
    names = status_probe.get("namechange", [])
    if isinstance(names, list) and any(
        "ST" in str(row.get("name", "")).upper() for row in names if isinstance(row, dict)
    ):
        return MarketDispositionEvidence("st_restricted", evidence_hash)
    basic = status_probe.get("stock_basic", [])
    first = (
        basic[0]
        if isinstance(basic, list) and basic and isinstance(basic[0], dict)
        else None
    )
    if first is None or not str(first.get("list_date", "")):
        return MarketDispositionEvidence("missing_market_status", evidence_hash)
    if (calendar[1] - parse_trade_date(str(first["list_date"]))).days < 60:
        return MarketDispositionEvidence("new_stock_restricted", evidence_hash)
    by_date = {bar.trade_date: bar for bar in bars}
    for trade_date in calendar[1:]:
        bar = by_date.get(trade_date)
        if bar is None or bar.pre_close is None:
            return MarketDispositionEvidence("missing_market_status", evidence_hash)
        limit = price_limit_pct(stock_id, Decimal("0.10"))
        upper = bar.pre_close * (Decimal("1") + limit)
        lower = bar.pre_close * (Decimal("1") - limit)
        if bar.open_price >= upper or bar.open_price <= lower:
            return MarketDispositionEvidence("price_limit_blocked", evidence_hash)
    return None


def _hash_json(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_values(values: Iterable[str]) -> str:
    return sha256("|".join(values).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
