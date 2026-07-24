from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar, cast

from seatflow.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestReport,
    FailClosedBacktestError,
    OosFold,
    PredictionSignal,
)
from seatflow.data.contracts import (
    DailyBar,
    DragonTigerSummary,
    SuspensionRecord,
    SuspensionResponseEvidence,
)
from seatflow.data.providers import AkshareProvider, ProviderError, TushareHttpProvider
from seatflow.evaluation.blind import freeze_evaluation_manifest
from seatflow.evaluation.protocol import freeze_protocol, load_experiment_protocol
from seatflow.research import (
    ResearchSample,
    build_research_samples,
    evaluate_predictions,
    predict_proba,
    samples_to_matrix,
    time_split_samples,
    train_l2_logistic,
)
from seatflow.tracking.lineage import LineageRecord, now_iso, record_training_run
from seatflow.training.preflight import TimeSplit

T = TypeVar("T")


@dataclass(frozen=True)
class HistoricalResearchConfig:
    start_date: date
    end_date: date
    data_root: Path
    output_root: Path
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    top_n: int = 5
    minimum_score: Decimal | None = None
    akshare_sample_count: int = 3
    require_akshare_confirmation: bool = False
    protocol_config: Path = Path("configs/experiments/research_protocol_v2.toml")

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.akshare_sample_count < 0:
            raise ValueError("akshare_sample_count must be non-negative")


@dataclass(frozen=True)
class HistoricalResearchResult:
    effective_end_date: date
    calendar_days: int
    event_count: int
    stock_count: int
    bar_count: int
    sample_count: int
    development_metrics: dict[str, object]
    contaminated_oos_metrics: dict[str, object]
    backtest_metrics: dict[str, object]
    akshare_confirmation: dict[str, object]
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_end_date": self.effective_end_date.isoformat(),
            "calendar_days": self.calendar_days,
            "event_count": self.event_count,
            "stock_count": self.stock_count,
            "bar_count": self.bar_count,
            "sample_count": self.sample_count,
            "development_metrics": self.development_metrics,
            "contaminated_oos_metrics": self.contaminated_oos_metrics,
            "backtest_metrics": self.backtest_metrics,
            "akshare_confirmation": self.akshare_confirmation,
            "report_path": str(self.report_path),
        }


def run_historical_research(
    config: HistoricalResearchConfig,
    tushare: TushareHttpProvider,
    akshare: AkshareProvider | None = None,
) -> HistoricalResearchResult:
    """Collect, train, and backtest a fixed, event-driven research baseline.

    The run uses Tushare for the primary historical source. AkShare only validates
    a small end-date sample; an unavailable public source is recorded explicitly.
    """

    protocol = load_experiment_protocol(config.protocol_config)
    if protocol.data_scope != "real":
        raise ValueError("historical research protocol must use data_scope=real")
    if protocol.seat_detail_scope != "institutional_proxy":
        raise ValueError("historical research requires the institutional_proxy protocol")
    future_blind_start = protocol.window("future_blind").start
    if config.end_date >= future_blind_start:
        raise ValueError(
            "historical research refuses a range that reaches future blind labels; "
            "use an independent evaluator after the blind freeze"
        )
    calendar = _load_or_fetch_calendar(
        config.data_root, tushare, config.start_date, config.end_date
    )
    if len(calendar) < 100:
        raise ValueError("historical research requires at least 100 open trading days")
    effective_end_date = calendar[-1]
    frozen_protocol = freeze_protocol(config.output_root / "protocol", protocol)
    events = _load_or_fetch_events(config.data_root, tushare, calendar)
    stock_ids = tuple(sorted({event.stock_id for event in events}))
    if not stock_ids:
        raise ValueError("Tushare returned no Dragon-Tiger events in the requested period")
    bars = _load_or_fetch_histories(
        config.data_root, tushare, stock_ids, config.start_date, effective_end_date
    )
    benchmark_closes = _load_or_fetch_benchmark_closes(
        config.data_root,
        tushare,
        "000300.SH",
        config.start_date,
        effective_end_date,
    )
    akshare_confirmation = _confirm_akshare(
        akshare,
        bars,
        stock_ids,
        effective_end_date,
        config.akshare_sample_count,
    )
    if config.require_akshare_confirmation and not bool(akshare_confirmation["confirmed"]):
        raise ValueError("AkShare confirmation was required but did not complete successfully")

    samples = build_research_samples(events, bars)
    if not samples:
        raise ValueError("no trainable samples remain after history and label requirements")
    _write_dataset_files(config.output_root, samples)

    contaminated_start = protocol.window("contaminated_oos").start
    if effective_end_date < contaminated_start:
        raise ValueError("requested period does not contain the 2026 contaminated OOS window")
    pre_contaminated = tuple(sample for sample in samples if sample.as_of_date < contaminated_start)
    contaminated_end = protocol.window("contaminated_oos").end
    contaminated_oos = tuple(
        sample
        for sample in samples
        if contaminated_start <= sample.as_of_date <= contaminated_end
    )
    if not pre_contaminated or not contaminated_oos:
        raise ValueError("both pre-contaminated and contaminated OOS samples are required")

    development_split = time_split_samples(
        pre_contaminated,
        train_end=date(2024, 6, 30),
        valid_end=date(2024, 12, 31),
        purge_days=protocol.purge_days,
        embargo_days=protocol.embargo_days,
    )
    if not development_split.train or not development_split.valid or not development_split.test:
        raise ValueError("2023-2025 split produced an empty development partition")
    development = _run_development_seeds(
        config, development_split.train, development_split.valid, development_split.test
    )

    frozen_model = train_l2_logistic(samples_to_matrix(pre_contaminated), seed=config.seeds[0])
    contaminated_dataset = samples_to_matrix(contaminated_oos)
    contaminated_probabilities = predict_proba(frozen_model, contaminated_dataset.x)
    contaminated_oos_metrics = evaluate_predictions(
        contaminated_probabilities, contaminated_dataset.y, contaminated_dataset.label_returns
    ).to_dict()
    _write_json(
        config.output_root / "models" / "frozen_precontaminated_model.json", frozen_model.to_dict()
    )

    evaluation_paths = _write_evaluation_files(
        config.output_root, contaminated_oos, "contaminated_oos"
    )
    evaluation_manifest = freeze_evaluation_manifest(
        config.output_root / "contaminated_oos" / "evaluation_manifest.json",
        feature_paths=[evaluation_paths[0]],
        label_paths=[evaluation_paths[1]],
        split_status="contaminated_oos",
        metadata={
            "data_scope": "real",
            "seat_detail_scope": "institutional_proxy",
            "label_status": "observed",
            "blind_status": "contaminated_oos",
            "model_family": "event_logistic_baseline_v1",
            "contaminated_oos_start": contaminated_start.isoformat(),
            "contaminated_oos_end": effective_end_date.isoformat(),
            "training_end": "2025-12-31",
            "model_seed": config.seeds[0],
        },
    )
    try:
        backtest_report = _backtest_contaminated_oos(
            config,
            contaminated_oos,
            contaminated_probabilities,
            bars,
            benchmark_closes,
            calendar,
            evaluation_manifest.sha256,
            tushare.fetch_suspension_evidence,
        )
        threshold_sensitivity = _backtest_threshold_sensitivity(
            config,
            contaminated_oos,
            contaminated_probabilities,
            bars,
            benchmark_closes,
            calendar,
            evaluation_manifest.sha256,
            tushare.fetch_suspension_evidence,
        )
        backtest_payload: dict[str, object] = backtest_report.to_dict()
        backtest_payload["threshold_sensitivity"] = threshold_sensitivity
        backtest_metrics: dict[str, object] = backtest_report.metrics
    except FailClosedBacktestError as error:
        # Preserve the failed execution evidence instead of emitting an optimistic report.
        negative_path = config.output_root / "backtest" / "contaminated_oos_negative_result.json"
        if error.report is None:
            backtest_payload = {
                "run_id": "tushare-2023-2026-logistic",
                "mode": "real",
                "claims": {
                    "real_backtest_valid": False,
                    "negative_result": True,
                },
                "status": "fail_closed",
                "failure_reason": str(error),
                "diagnostics": error.diagnostics,
                "evaluation_manifest_sha256": evaluation_manifest.sha256,
                "final_blind_manifest": None,
            }
        else:
            backtest_payload = error.report.to_dict()
            backtest_payload["status"] = "fail_closed"
            backtest_payload["failure_reason"] = str(error)
            backtest_payload["diagnostics"] = error.diagnostics
            claims = dict(cast(dict[str, object], backtest_payload.get("claims", {})))
            claims["real_backtest_valid"] = False
            claims["negative_result"] = True
            backtest_payload["claims"] = claims
        _write_json(negative_path, backtest_payload)
        backtest_payload["negative_result_path"] = str(negative_path)
        backtest_metrics = {
            "negative_result": True,
            "failure_reason": str(error),
            "negative_result_path": str(negative_path),
        }
    _write_json(
        config.output_root / "backtest" / "contaminated_oos_report.json", backtest_payload
    )

    run_hashes = _run_hashes(config, frozen_protocol.sha256, evaluation_manifest.sha256)
    lineage = record_training_run(
        LineageRecord(
            run_id="tushare-2023-2026-event-logistic-contaminated-oos",
            run_type="research_baseline",
            status="contaminated_oos_evaluated",
            model_valid=False,
            provider="tushare",
            seed=config.seeds[0],
            split=TimeSplit(
                train_start=protocol.window("train").start,
                train_end=protocol.window("train").end,
                valid_start=protocol.window("valid").start,
                valid_end=protocol.window("valid").end,
                test_start=protocol.window("dev_oos").start,
                test_end=protocol.window("dev_oos").end,
            ),
            hashes=run_hashes,
            metrics=contaminated_oos_metrics,
            tags=(
                "data_scope:real",
                "seat_detail_scope:institutional_proxy",
                "blind_status:contaminated_oos",
                "model_family:event_logistic_baseline_v1",
            ),
            created_at=now_iso(),
            data_scope="real",
            seat_detail_scope="institutional_proxy",
            label_status="observed",
            blind_status="contaminated_oos",
            model_family="event_logistic_baseline_v1",
        ),
        config.output_root,
    )
    report_path = config.output_root / "research_report.json"
    report = {
        "schema_version": 2,
        "data_scope": "real",
        "data_provider": "tushare",
        "seat_detail_scope": "institutional_proxy",
        "label_status": "observed",
        "blind_status": "contaminated_oos",
        "model_family": "event_logistic_baseline_v1",
        "protocol": frozen_protocol.to_dict(),
        "release_claims": {
            "model_valid": False,
            "competition_ready": False,
            "complete_seat_data": False,
            "akshare_cross_source_confirmed": akshare_confirmation["confirmed"],
            "real_backtest_valid": False,
            "result_interpretation": "negative_result_contaminated_oos_not_final_blind",
        },
        "period": {"start": config.start_date.isoformat(), "end": effective_end_date.isoformat()},
        "counts": {
            "calendar_days": len(calendar),
            "event_count": len(events),
            "stock_count": len(stock_ids),
            "bar_count": len(bars),
            "sample_count": len(samples),
            "pre_contaminated_sample_count": len(pre_contaminated),
            "contaminated_oos_sample_count": len(contaminated_oos),
        },
        "development_metrics": development,
        "contaminated_oos_metrics": contaminated_oos_metrics,
        "backtest": backtest_payload,
        "evaluation_manifest": evaluation_manifest.to_dict(),
        "run_hashes": run_hashes,
        "lineage": lineage.to_dict(),
        "akshare_confirmation": akshare_confirmation,
        "risk_flags": [
            "tushare_top_inst_is_institutional_only",
            "baseline_is_not_complete_seat_dtseathgt",
            "2026_evaluation_is_contaminated_oos_not_final_blind",
            "evaluation_manifest_is_not_a_final_blind_manifest",
            "akshare_is_a_public_web_source_and_may_be_unavailable",
            "no_release_or_competition_validity_claim_is_made",
        ],
    }
    _write_json(report_path, report)
    return HistoricalResearchResult(
        effective_end_date=effective_end_date,
        calendar_days=len(calendar),
        event_count=len(events),
        stock_count=len(stock_ids),
        bar_count=len(bars),
        sample_count=len(samples),
        development_metrics=development,
        contaminated_oos_metrics=contaminated_oos_metrics,
        backtest_metrics=backtest_metrics,
        akshare_confirmation=akshare_confirmation,
        report_path=report_path,
    )


def _load_or_fetch_calendar(
    root: Path, provider: TushareHttpProvider, start_date: date, end_date: date
) -> tuple[date, ...]:
    path = root / "raw" / "tushare" / "trading_calendar.json"
    if path.exists():
        values = _string_list(_read_json(path).get("dates"), "cached calendar dates")
        return tuple(date.fromisoformat(value) for value in values)
    dates = provider.fetch_trading_calendar(start_date, end_date)
    _write_json(path, {"dates": [item.isoformat() for item in dates]})
    return dates


def _load_or_fetch_events(
    root: Path, provider: TushareHttpProvider, calendar: tuple[date, ...]
) -> tuple[DragonTigerSummary, ...]:
    events: list[DragonTigerSummary] = []
    for trade_date in calendar:
        path = root / "raw" / "tushare" / "top_list" / f"date={trade_date.isoformat()}.json"
        if path.exists():
            payload = _read_json(path)
            cached_rows = _object_list(payload.get("rows"), "cached events")
            events.extend(_summary_from_dict(row) for row in cached_rows)
            continue
        rows = _fetch_summaries(provider, trade_date)
        _write_json(path, {"rows": [row.to_dict() for row in rows]})
        events.extend(rows)
    return tuple(events)


def _load_or_fetch_histories(
    root: Path,
    provider: TushareHttpProvider,
    stock_ids: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> tuple[DailyBar, ...]:
    bars: list[DailyBar] = []
    for stock_id in stock_ids:
        bars.extend(_load_or_fetch_history(root, provider, stock_id, start_date, end_date))
    return tuple(bars)


def _load_or_fetch_history(
    root: Path,
    provider: TushareHttpProvider,
    stock_id: str,
    start_date: date,
    end_date: date,
) -> tuple[DailyBar, ...]:
    path = (
        root
        / "raw"
        / "tushare"
        / "daily"
        / f"stock={stock_id}.start={start_date.isoformat()}.end={end_date.isoformat()}.json.gz"
    )
    if path.exists():
        payload = _read_gzip_json(path)
        cached_rows = _object_list(payload.get("rows"), "cached daily bars")
        return tuple(_bar_from_dict(row) for row in cached_rows)
    rows = _fetch_daily_history(provider, stock_id, start_date, end_date)
    _write_gzip_json(path, {"rows": [row.to_dict() for row in rows]})
    return rows


def _load_or_fetch_benchmark_closes(
    root: Path,
    provider: TushareHttpProvider,
    index_code: str,
    start_date: date,
    end_date: date,
) -> tuple[tuple[date, Decimal], ...]:
    path = (
        root
        / "raw"
        / "tushare"
        / "index_daily"
        / f"index={index_code}.start={start_date.isoformat()}.end={end_date.isoformat()}.json"
    )
    if path.exists():
        rows = _read_json(path)["rows"]
        if not isinstance(rows, list):
            raise ValueError("cached benchmark rows are invalid")
        return tuple(_benchmark_close_from_dict(row) for row in rows)
    rows = _retry(lambda: provider.fetch_index_close_history(index_code, start_date, end_date))
    _write_json(
        path,
        {
            "rows": [
                {"trade_date": trade_date.isoformat(), "close": str(close)}
                for trade_date, close in rows
            ]
        },
    )
    return rows


def _confirm_akshare(
    provider: AkshareProvider | None,
    bars: tuple[DailyBar, ...],
    stock_ids: tuple[str, ...],
    effective_end_date: date,
    sample_count: int,
) -> dict[str, object]:
    if provider is None or sample_count == 0:
        return {"attempted": False, "confirmed": False, "reason": "not_requested"}
    bar_map = {(bar.stock_id, bar.trade_date): bar for bar in bars}
    checked: list[dict[str, object]] = []
    failures: list[str] = []
    for stock_id in stock_ids[:sample_count]:
        try:
            rows = provider.fetch_daily_bars(effective_end_date, stock_id)
        except ProviderError as error:
            failures.append(str(error))
            continue
        source_bar = bar_map.get((stock_id, effective_end_date))
        akshare_bar = next((row for row in rows if row.trade_date == effective_end_date), None)
        if source_bar is None or akshare_bar is None:
            failures.append(f"missing comparable bar for {stock_id}")
            continue
        checked.append(
            {
                "stock_id": stock_id,
                "tushare_close": str(source_bar.close_price),
                "akshare_close": str(akshare_bar.close_price),
                "close_matches": source_bar.close_price == akshare_bar.close_price,
            }
        )
    confirmed = bool(checked) and not failures and all(item["close_matches"] for item in checked)
    return {
        "attempted": True,
        "confirmed": confirmed,
        "checked": checked,
        "failures": failures,
    }


def _run_development_seeds(
    config: HistoricalResearchConfig,
    train: tuple[ResearchSample, ...],
    valid: tuple[ResearchSample, ...],
    test: tuple[ResearchSample, ...],
) -> dict[str, object]:
    train_dataset = samples_to_matrix(train)
    valid_dataset = samples_to_matrix(valid)
    test_dataset = samples_to_matrix(test)
    per_seed: list[dict[str, object]] = []
    for seed in config.seeds:
        model = train_l2_logistic(train_dataset, seed=seed)
        valid_metrics = evaluate_predictions(
            predict_proba(model, valid_dataset.x), valid_dataset.y, valid_dataset.label_returns
        ).to_dict()
        test_metrics = evaluate_predictions(
            predict_proba(model, test_dataset.x), test_dataset.y, test_dataset.label_returns
        ).to_dict()
        _write_json(
            config.output_root / "models" / f"development_seed={seed}.json", model.to_dict()
        )
        per_seed.append({"seed": seed, "validation": valid_metrics, "test_2025": test_metrics})
    return {"seeds": per_seed, "seed_count": len(per_seed)}


def _write_dataset_files(root: Path, samples: tuple[ResearchSample, ...]) -> None:
    _write_jsonl(root / "datasets" / "all_samples.jsonl", (sample.to_dict() for sample in samples))
    _write_json(
        root / "datasets" / "manifest.json",
        {
            "feature_names": list(samples[0].feature_names),
            "sample_count": len(samples),
            "start": min(sample.as_of_date for sample in samples).isoformat(),
            "end": max(sample.as_of_date for sample in samples).isoformat(),
            "sha256": _sha256_file(root / "datasets" / "all_samples.jsonl"),
        },
    )


def _write_evaluation_files(
    root: Path, samples: tuple[ResearchSample, ...], split_status: str
) -> tuple[Path, Path]:
    feature_path = root / split_status / "features.jsonl"
    label_path = root / split_status / "labels.jsonl"
    _write_jsonl(
        feature_path,
        (
            {
                "sample_id": sample.sample_id,
                "as_of_date": sample.as_of_date.isoformat(),
                "stock_id": sample.stock_id,
                "feature_names": list(sample.feature_names),
                "features": list(sample.features),
            }
            for sample in samples
        ),
    )
    _write_jsonl(
        label_path,
        (
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "label_return": sample.label_return,
                "label_start_date": sample.label_start_date.isoformat(),
                "label_end_date": sample.label_end_date.isoformat(),
            }
            for sample in samples
        ),
    )
    return feature_path, label_path


def _backtest_contaminated_oos(
    config: HistoricalResearchConfig,
    contaminated_oos: tuple[ResearchSample, ...],
    probabilities: tuple[float, ...],
    bars: tuple[DailyBar, ...],
    benchmark_closes: tuple[tuple[date, Decimal], ...],
    calendar: tuple[date, ...],
    evaluation_manifest_sha256: str,
    fetch_suspension_evidence: Callable[[date], SuspensionResponseEvidence],
) -> BacktestReport:
    signals = tuple(
        PredictionSignal(
            as_of_date=sample.as_of_date,
            stock_id=sample.stock_id,
            score=Decimal(str(probability)),
            available_time=datetime.combine(sample.as_of_date, datetime.min.time()).replace(
                hour=18
            ),
        )
        for sample, probability in zip(contaminated_oos, probabilities, strict=True)
    )
    contaminated_start = min(sample.as_of_date for sample in contaminated_oos)
    calendar_slice = tuple(item for item in calendar if item >= contaminated_start)
    event_universe: dict[date, tuple[str, ...]] = {}
    for sample in contaminated_oos:
        event_universe.setdefault(sample.as_of_date, ())
        event_universe[sample.as_of_date] = tuple(
            sorted(set(event_universe[sample.as_of_date]) | {sample.stock_id})
        )
    normalized_bars = tuple(
        replace(bar, volume=bar.volume * Decimal("100"), volume_unit="shares")
        for bar in bars
        if bar.trade_date >= contaminated_start
    )
    confirmed_suspensions = _require_execution_bar_completeness(
        output_root=config.output_root,
        signals=signals,
        bars=normalized_bars,
        trading_calendar=calendar_slice,
        fetch_suspension_evidence=fetch_suspension_evidence,
    )
    benchmark_returns = _benchmark_returns(benchmark_closes, contaminated_start)
    return BacktestEngine(
        BacktestConfig(
            run_id="tushare-2023-2026-logistic",
            mode="real",
            top_n=config.top_n,
            minimum_score=config.minimum_score,
            exclude_as_of_limit_up_signals=True,
            exclude_as_of_zero_volume_signals=True,
            min_as_of_close_price=Decimal("5"),
            excluded_stock_exchanges=("BJ",),
        )
    ).run(
        signals=signals,
        bars=normalized_bars,
        trading_calendar=calendar_slice,
        universe_by_date=event_universe,
        benchmark_returns=benchmark_returns,
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        evaluation_manifest_verified=True,
        final_blind_manifest_sha256=None,
        final_blind_manifest_verified=False,
        oos_folds=(OosFold("contaminated-oos-2026", contaminated_start, calendar_slice[-1]),),
        confirmed_suspensions=confirmed_suspensions,
    )


def _backtest_threshold_sensitivity(
    config: HistoricalResearchConfig,
    contaminated_oos: tuple[ResearchSample, ...],
    probabilities: tuple[float, ...],
    bars: tuple[DailyBar, ...],
    benchmark_closes: tuple[tuple[date, Decimal], ...],
    calendar: tuple[date, ...],
    evaluation_manifest_sha256: str,
    fetch_suspension_evidence: Callable[[date], SuspensionResponseEvidence],
) -> dict[str, object]:
    contaminated_start = min(sample.as_of_date for sample in contaminated_oos)
    calendar_slice = tuple(item for item in calendar if item >= contaminated_start)
    event_universe: dict[date, tuple[str, ...]] = {}
    for sample in contaminated_oos:
        event_universe.setdefault(sample.as_of_date, ())
        event_universe[sample.as_of_date] = tuple(
            sorted(set(event_universe[sample.as_of_date]) | {sample.stock_id})
        )
    normalized_bars = tuple(
        replace(bar, volume=bar.volume * Decimal("100"), volume_unit="shares")
        for bar in bars
        if bar.trade_date >= contaminated_start
    )
    confirmed_suspensions = _require_execution_bar_completeness(
        output_root=config.output_root,
        signals=tuple(
            PredictionSignal(
                as_of_date=sample.as_of_date,
                stock_id=sample.stock_id,
                score=Decimal(str(probability)),
                available_time=datetime.combine(sample.as_of_date, datetime.min.time()).replace(
                    hour=18
                ),
            )
            for sample, probability in zip(contaminated_oos, probabilities, strict=True)
        ),
        bars=normalized_bars,
        trading_calendar=calendar_slice,
        fetch_suspension_evidence=fetch_suspension_evidence,
    )
    benchmark_returns = _benchmark_returns(benchmark_closes, contaminated_start)
    thresholds = (
        (Decimal("0.50"), "threshold_0.50"),
        (Decimal("0.55"), "threshold_0.55"),
        (Decimal("0.60"), "threshold_0.60"),
        (Decimal("2"), "no_trade_policy"),
    )
    rows: list[dict[str, object]] = []
    signals = tuple(
        PredictionSignal(
            as_of_date=sample.as_of_date,
            stock_id=sample.stock_id,
            score=Decimal(str(probability)),
            available_time=datetime.combine(sample.as_of_date, datetime.min.time()).replace(
                hour=18
            ),
        )
        for sample, probability in zip(contaminated_oos, probabilities, strict=True)
    )
    for threshold, label in thresholds:
        common_run = {
            "signals": signals,
            "bars": normalized_bars,
            "trading_calendar": calendar_slice,
            "universe_by_date": event_universe,
            "benchmark_returns": benchmark_returns,
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
            "evaluation_manifest_verified": True,
            "final_blind_manifest_sha256": None,
            "final_blind_manifest_verified": False,
            "oos_folds": (
                OosFold("contaminated-oos-2026", contaminated_start, calendar_slice[-1]),
            ),
            "confirmed_suspensions": confirmed_suspensions,
        }
        gross_report = _run_expected_no_trade_sensitivity(
            BacktestConfig(
                run_id=f"tushare-2023-2026-logistic-sensitivity-{label}",
                mode="real",
                top_n=config.top_n,
                minimum_score=threshold,
                commission_bps=Decimal("0"),
                stamp_tax_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
                exclude_as_of_limit_up_signals=True,
                exclude_as_of_zero_volume_signals=True,
                min_as_of_close_price=Decimal("5"),
                excluded_stock_exchanges=("BJ",),
            ),
            common_run,
        )
        net_report = _run_expected_no_trade_sensitivity(
            BacktestConfig(
                run_id=f"tushare-2023-2026-logistic-sensitivity-net-{label}",
                mode="real",
                top_n=config.top_n,
                minimum_score=threshold,
                exclude_as_of_limit_up_signals=True,
                exclude_as_of_zero_volume_signals=True,
                min_as_of_close_price=Decimal("5"),
                excluded_stock_exchanges=("BJ",),
            ),
            common_run,
        )
        selection = cast(dict[str, object], net_report.audit["selection"])
        execution = cast(dict[str, object], net_report.audit["execution"])
        rows.append(
            {
                "policy": label,
                "minimum_score": str(threshold),
                "gross_return": gross_report.metrics["total_return"],
                "net_return": net_report.metrics["total_return"],
                "turnover": execution["turnover"],
                "fill_quality": execution["fill_rate"],
                "fill_rate": execution["fill_rate"],
                "no_trade_days": selection["abstention_day_count"],
                "selected_signal_count": selection["selected_signal_count"],
                "abstention_day_count": selection["abstention_day_count"],
                "exposure_day_count": selection["exposure_day_count"],
                "diagnostic_only": True,
            }
        )
    return {
        "thresholds": rows,
        "note": "diagnostic_only_no_promotion_threshold_selected_from_contaminated_oos",
    }


def _run_expected_no_trade_sensitivity(
    config: BacktestConfig,
    run_kwargs: dict[str, object],
) -> BacktestReport:
    try:
        return BacktestEngine(config).run(**run_kwargs)  # type: ignore[arg-type]
    except FailClosedBacktestError as error:
        if (
            error.report is not None
            and "no executable orders" in str(error)
            and config.minimum_score is not None
        ):
            return error.report
        raise


def _require_execution_bar_completeness(
    *,
    output_root: Path,
    signals: tuple[PredictionSignal, ...],
    bars: tuple[DailyBar, ...],
    trading_calendar: tuple[date, ...],
    fetch_suspension_evidence: Callable[[date], SuspensionResponseEvidence],
) -> dict[date, tuple[str, ...]]:
    """Classify every missing execution bar with current Tushare ``suspend_d`` evidence."""

    calendar = tuple(sorted(set(trading_calendar)))
    next_trading_days = {
        as_of_date: calendar[index + 1] for index, as_of_date in enumerate(calendar[:-1])
    }
    bar_keys = {(bar.stock_id, bar.trade_date) for bar in bars}
    latest_bar_by_stock: dict[str, date] = {}
    for bar in bars:
        latest_bar_by_stock[bar.stock_id] = max(
            latest_bar_by_stock.get(bar.stock_id, bar.trade_date), bar.trade_date
        )

    missing: list[dict[str, str]] = []
    for signal in sorted(signals, key=lambda item: (item.as_of_date, item.stock_id)):
        execution_date = next_trading_days.get(signal.as_of_date)
        if execution_date is None or (signal.stock_id, execution_date) in bar_keys:
            continue
        gap_type = (
            "cache_gap"
            if latest_bar_by_stock.get(signal.stock_id, date.min) > execution_date
            else "tail_truncation"
        )
        missing.append(
            {
                "as_of_date": signal.as_of_date.isoformat(),
                "execution_date": execution_date.isoformat(),
                "gap_type": gap_type,
                "stock_id": signal.stock_id,
            }
        )

    detail_path = output_root / "backtest" / "execution_tradability.jsonl"
    evidence_by_date: dict[date, SuspensionResponseEvidence] = {}
    query_errors: dict[date, str] = {}
    query_rows: list[dict[str, object]] = []
    for execution_date in calendar:
        try:
            evidence = fetch_suspension_evidence(execution_date)
        except ProviderError as error:
            query_errors[execution_date] = str(error)
            query_rows.append(
                {
                    "daily_query_sha256": None,
                    "error": str(error),
                    "record_count": None,
                    "retrieved_at": None,
                    "source": None,
                    "source_row_count": None,
                    "status": "query_failed",
                    "trade_date": execution_date.isoformat(),
                }
            )
            continue
        if evidence.trade_date != execution_date:
            query_errors[execution_date] = (
                "suspension evidence trade date does not match execution date"
            )
            query_rows.append(
                {
                    "daily_query_sha256": evidence.daily_query_sha256,
                    "error": query_errors[execution_date],
                    "record_count": len(evidence.records),
                    "retrieved_at": evidence.retrieved_at.isoformat(),
                    "source": evidence.source,
                    "source_row_count": evidence.source_row_count,
                    "status": "invalid_response_date",
                    "trade_date": execution_date.isoformat(),
                }
            )
            continue
        evidence_by_date[execution_date] = evidence
        query_rows.append(
            {
                "daily_query_sha256": evidence.daily_query_sha256,
                "error": None,
                "record_count": len(evidence.records),
                "retrieved_at": evidence.retrieved_at.isoformat(),
                "source": evidence.source,
                "source_row_count": evidence.source_row_count,
                "status": "queried",
                "trade_date": execution_date.isoformat(),
            }
        )
    _write_jsonl(output_root / "backtest" / "suspension_query_evidence.jsonl", query_rows)

    confirmed: dict[date, set[str]] = {}
    for execution_date, evidence in evidence_by_date.items():
        confirmed[execution_date] = {record.stock_id for record in evidence.records}
    if not missing:
        _write_jsonl(detail_path, ())
        return {
            trade_date: tuple(sorted(stock_ids))
            for trade_date, stock_ids in confirmed.items()
            if stock_ids
        }

    tradability_rows: list[dict[str, object]] = []
    for item in missing:
        execution_date = date.fromisoformat(item["execution_date"])
        execution_evidence = evidence_by_date.get(execution_date)
        matching_records: tuple[SuspensionRecord, ...] = ()
        if execution_evidence is not None:
            matching_records = tuple(
                sorted(
                    (
                        record
                        for record in execution_evidence.records
                        if record.stock_id == item["stock_id"]
                    ),
                    key=lambda record: (
                        record.suspend_timing,
                        record.row_payload_sha256,
                    ),
                )
            )
        record = matching_records[0] if matching_records else None
        if record is not None:
            status = "confirmed_suspension_no_daily_bar"
        else:
            status = "unclassified_data_gap"
        tradability_rows.append(
            {
                "as_of_date": item["as_of_date"],
                "execution_date": item["execution_date"],
                "gap_type": item["gap_type"],
                "retrieved_at": (
                    record.retrieved_at.isoformat()
                    if record is not None
                    else (
                        execution_evidence.retrieved_at.isoformat()
                        if execution_evidence is not None
                        else None
                    )
                ),
                "source": (
                    record.source
                    if record is not None
                    else execution_evidence.source if execution_evidence is not None else None
                ),
                "status": status,
                "stock_id": item["stock_id"],
                "suspend_timing": record.suspend_timing if record is not None else None,
                "suspend_type": record.suspend_type if record is not None else None,
                "suspension_daily_query_sha256": (
                    execution_evidence.daily_query_sha256
                    if execution_evidence is not None
                    else None
                ),
                "suspension_query_error": query_errors.get(execution_date),
                "suspension_row_payload_sha256": (
                    record.row_payload_sha256 if record is not None else None
                ),
            }
        )

    _write_jsonl(detail_path, tradability_rows)
    unclassified = [row for row in tradability_rows if row["status"] == "unclassified_data_gap"]
    if not unclassified:
        return {
            trade_date: tuple(sorted(stock_ids))
            for trade_date, stock_ids in confirmed.items()
            if stock_ids
        }
    raise FailClosedBacktestError(
        "real backtest execution preflight found "
        f"{len(unclassified)} unclassified missing execution bar(s); details: {detail_path}"
    )


def _benchmark_returns(
    closes: tuple[tuple[date, Decimal], ...], start_date: date
) -> dict[date, Decimal]:
    ordered = sorted((item for item in closes if item[0] >= start_date), key=lambda item: item[0])
    returns: dict[date, Decimal] = {}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous[1] > 0:
            returns[current[0]] = current[1] / previous[1] - Decimal("1")
    if not returns:
        raise ValueError("benchmark bars do not cover the evaluation period")
    return returns


def _run_hashes(
    config: HistoricalResearchConfig, protocol_sha256: str, evaluation_manifest_sha256: str
) -> dict[str, str]:
    dataset_path = config.output_root / "datasets" / "all_samples.jsonl"
    checkpoint_path = config.output_root / "models" / "frozen_precontaminated_model.json"
    split_payload = {
        "contaminated_oos_start": "2026-01-01",
        "train_end": "2024-06-30",
        "valid_end": "2024-12-31",
    }
    return {
        "data": _sha256_file(dataset_path),
        "config": _sha256_file(config.protocol_config),
        "code": _sha256_file(Path(__file__)),
        "seed": sha256(json.dumps(list(config.seeds)).encode("utf-8")).hexdigest(),
        "split": sha256(json.dumps(split_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        "checkpoint": _sha256_file(checkpoint_path),
        "protocol": protocol_sha256,
        "evaluation_manifest": evaluation_manifest_sha256,
        "final_blind_manifest": "",
        "blind_manifest": "",
    }


def _summary_from_dict(raw: object) -> DragonTigerSummary:
    if not isinstance(raw, dict):
        raise ValueError("cached event row is invalid")
    available_time = raw.get("available_time")
    return DragonTigerSummary(
        trade_date=date.fromisoformat(str(raw["trade_date"])),
        stock_id=str(raw["stock_id"]),
        stock_name=str(raw["stock_name"]),
        buy_amount=Decimal(str(raw["buy_amount"])),
        sell_amount=Decimal(str(raw["sell_amount"])),
        net_amount=Decimal(str(raw["net_amount"])),
        reason=str(raw["reason"]),
        source_name=str(raw["source_name"]),
        source_version=str(raw["source_version"]) if raw.get("source_version") else None,
        available_time=datetime.fromisoformat(str(available_time)) if available_time else None,
    )


def _benchmark_close_from_dict(raw: object) -> tuple[date, Decimal]:
    if not isinstance(raw, dict):
        raise ValueError("cached benchmark row is invalid")
    return date.fromisoformat(str(raw["trade_date"])), Decimal(str(raw["close"]))


def _bar_from_dict(raw: object) -> DailyBar:
    if not isinstance(raw, dict):
        raise ValueError("cached daily row is invalid")
    available_time = raw.get("available_time")
    pre_close = raw.get("pre_close")
    return DailyBar(
        trade_date=date.fromisoformat(str(raw["trade_date"])),
        stock_id=str(raw["stock_id"]),
        open_price=Decimal(str(raw["open_price"])),
        high_price=Decimal(str(raw["high_price"])),
        low_price=Decimal(str(raw["low_price"])),
        close_price=Decimal(str(raw["close_price"])),
        volume=Decimal(str(raw["volume"])),
        amount=Decimal(str(raw["amount"])),
        source_name=str(raw["source_name"]),
        source_version=str(raw["source_version"]) if raw.get("source_version") else None,
        available_time=datetime.fromisoformat(str(available_time)) if available_time else None,
        pre_close=Decimal(str(pre_close)) if pre_close is not None else None,
        volume_unit=str(raw.get("volume_unit") or "hand"),
        amount_unit=str(raw.get("amount_unit") or "thousand_cny"),
    )


def _retry(action: Callable[[], T], attempts: int = 6) -> T:
    last_error: ProviderError | None = None
    for attempt in range(attempts):
        try:
            return action()
        except ProviderError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    if last_error is None:
        raise RuntimeError("retry action did not execute")
    raise last_error


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _read_gzip_json(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid gzip JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(path, content)


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write(path, content)


def _write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            compressed.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fetch_summaries(
    provider: TushareHttpProvider, trade_date: date
) -> tuple[DragonTigerSummary, ...]:
    return _retry(lambda: provider.fetch_dragon_tiger_summaries(trade_date))


def _fetch_daily_history(
    provider: TushareHttpProvider, stock_id: str, start_date: date, end_date: date
) -> tuple[DailyBar, ...]:
    return _retry(lambda: provider.fetch_daily_bar_history(stock_id, start_date, end_date))


def _object_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return value


def _string_list(value: object, name: str) -> list[str]:
    return [str(item) for item in _object_list(value, name)]
