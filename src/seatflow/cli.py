from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

from seatflow.config import ProjectConfig, load_config
from seatflow.data.audit import audit_market_data, load_audit_policy, write_audit_report
from seatflow.data.providers import (
    AkshareProvider,
    MarketDataProvider,
    ProviderError,
    TushareHttpProvider,
    create_provider,
)
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import TensorizedWindowSample, tensorize_window_samples
from seatflow.datasets.snapshots import (
    MaturedLabelDiskCache,
    build_real_window_samples,
    load_verified_historical_snapshots,
)
from seatflow.evaluation.blind import freeze_blind_manifest
from seatflow.evaluation.dt_rolling import (
    DTRollingFold,
    run_dt_seathgt_folds_evaluation,
    save_dt_rolling_failure,
    save_dt_rolling_result,
)
from seatflow.evaluation.gates import evaluate_promotion_decision, save_promotion_decision
from seatflow.evaluation.protocol import (
    ExperimentProtocol,
    freeze_protocol,
    load_experiment_protocol,
)
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import TENSORIZATION_VERSION, derive_frozen_graph_schema
from seatflow.models.contracts import ModelInput
from seatflow.models.seat_hgt import DTSeatHGTConfig
from seatflow.paper_trading import (
    JsonPaperBroker,
    PaperBrokerError,
    PaperTradingService,
    load_paper_trading_config,
    load_quote_batch,
    load_signal_batch,
)
from seatflow.pipeline import (
    DataIngestionService,
    write_historical_backfill_report,
    write_historical_backfill_summary,
)
from seatflow.research_cache_repair import (
    repair_research_cache,
    write_research_cache_repair_report,
)
from seatflow.research_workflow import HistoricalResearchConfig, run_historical_research
from seatflow.runtime_env import load_workspace_env
from seatflow.services.predictor import HeuristicSmokePredictor
from seatflow.status import (
    load_phase_status,
    phase_status_from_promotion,
    write_phase_status,
)
from seatflow.tracking.lineage import export_run_package
from seatflow.training.dtseathgt import DTSeatHGTTrainingConfig
from seatflow.training.preflight import TimeSplit, validate_time_split
from seatflow.training.runner import run_engineering_smoke, run_pyg_shape_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seatflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="load and validate TOML config")
    _add_config_argument(validate)

    smoke = subparsers.add_parser("smoke", help="run the dependency-free synthetic flow")
    _add_config_argument(smoke)

    preflight = subparsers.add_parser(
        "preflight", help="validate time splits and local data-pipeline configuration"
    )
    _add_config_argument(preflight)
    _add_provider_argument(preflight)

    provider_status = subparsers.add_parser(
        "provider-status", help="show local provider configuration without network calls"
    )
    _add_config_argument(provider_status)
    _add_provider_argument(provider_status)

    ingest = subparsers.add_parser(
        "ingest", help="collect one market day, validate it, and write versioned snapshots"
    )
    _add_config_argument(ingest)
    _add_provider_argument(ingest)
    ingest.add_argument("--trade-date", type=_parse_date, required=True)
    ingest.add_argument("--stock-code", help="required by the AkShare daily-bar adapter")
    ingest.add_argument("--data-root", type=Path, help="defaults to the parent of data.raw_dir")
    seat_detail = ingest.add_mutually_exclusive_group()
    seat_detail.add_argument(
        "--skip-seat-detail",
        action="store_true",
        help="do not request institutional or seat-detail records",
    )
    seat_detail.add_argument(
        "--require-seat-detail",
        action="store_true",
        help="fail if complete seat-detail data cannot be collected",
    )

    historical_backfill = subparsers.add_parser(
        "backfill-history",
        help="ingest a verified historical trading calendar with resumable daily reports",
    )
    _add_config_argument(historical_backfill)
    _add_provider_argument(historical_backfill)
    historical_backfill.add_argument(
        "--trading-calendar-file",
        type=Path,
        required=True,
        help="JSON array or object containing historical trading dates",
    )
    historical_backfill.add_argument("--start", type=_parse_date)
    historical_backfill.add_argument("--end", type=_parse_date)
    historical_backfill.add_argument(
        "--stock-code", help="optional stock code forwarded to daily-bar ingestion"
    )
    historical_backfill.add_argument(
        "--data-root", type=Path, help="defaults to the parent of data.raw_dir"
    )
    historical_backfill.add_argument(
        "--output",
        type=Path,
        required=True,
        help="full JSON backfill report path",
    )
    historical_backfill.add_argument(
        "--summary",
        type=Path,
        help="optional compact JSON summary path",
    )
    backfill_seat_detail = historical_backfill.add_mutually_exclusive_group()
    backfill_seat_detail.add_argument(
        "--skip-seat-detail",
        action="store_true",
        help="do not request institutional or seat-detail records",
    )
    backfill_seat_detail.add_argument(
        "--require-seat-detail",
        action="store_true",
        help="fail a day if complete seat-detail data cannot be collected",
    )
    historical_backfill.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue collecting later days after a failed day, then return a failed report",
    )
    historical_backfill.add_argument(
        "--no-skip-verified",
        action="store_true",
        help="force re-ingestion even when all required snapshots verify by manifest hash",
    )

    audit = subparsers.add_parser(
        "audit-data", help="exercise provider endpoints against a verified trading calendar"
    )
    _add_config_argument(audit)
    _add_provider_argument(audit)
    audit.add_argument("--start", type=_parse_date, required=True)
    audit.add_argument("--end", type=_parse_date, required=True)
    audit.add_argument(
        "--trading-calendar-file",
        type=Path,
        required=True,
        help="JSON array or newline-delimited verified trading dates",
    )
    audit.add_argument("--audit-config", type=Path, default=Path("configs/data_audit.toml"))
    audit.add_argument("--output", type=Path)
    audit_seat_detail = audit.add_mutually_exclusive_group()
    audit_seat_detail.add_argument(
        "--require-seat-detail",
        dest="require_audit_seat_detail",
        action="store_true",
        default=True,
        help="require a live seat-detail endpoint audit (default)",
    )
    audit_seat_detail.add_argument(
        "--allow-summary-only",
        dest="require_audit_seat_detail",
        action="store_false",
        help="audit daily/top-list only; cannot establish SeatHGT readiness",
    )

    train_smoke = subparsers.add_parser(
        "train-smoke", help="run an explicitly non-valid engineering training smoke"
    )
    _add_config_argument(train_smoke)
    train_smoke.add_argument(
        "--model",
        choices=("baseline", "dtseathgt-shape"),
        default="baseline",
        help="baseline is dependency-free; dtseathgt-shape requires optional torch/PyG",
    )
    train_smoke.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    train_smoke.add_argument("--run-id")

    research_run = subparsers.add_parser(
        "research-run",
        help="collect Tushare data, train a fixed baseline, and evaluate contaminated 2026 OOS",
    )
    _add_config_argument(research_run)
    research_run.add_argument("--start", type=_parse_date, default=date(2023, 1, 1))
    research_run.add_argument("--end", type=_parse_date, default=date.today())
    research_run.add_argument(
        "--data-root", type=Path, default=Path("data/research/tushare_2023_2026")
    )
    research_run.add_argument(
        "--output-root", type=Path, default=Path("artifacts/research/tushare_2023_2026")
    )
    research_run.add_argument("--seed", type=int, action="append")
    research_run.add_argument("--top-n", type=int, default=5)
    research_run.add_argument(
        "--minimum-score",
        type=Decimal,
        default=None,
        help=(
            "optional contaminated-OOS backtest abstention threshold; "
            "omitted preserves legacy top-n"
        ),
    )
    research_run.add_argument("--akshare-sample-count", type=int, default=3)
    research_run.add_argument(
        "--research-protocol-config",
        type=Path,
        default=Path("configs/experiments/research_protocol_v2.toml"),
        help="frozen protocol source; the default labels viewed 2026 data as contaminated OOS",
    )
    research_run.add_argument(
        "--require-akshare-confirmation",
        action="store_true",
        help="fail rather than continue when AkShare cannot confirm the sampled Tushare closes",
    )

    dt_rolling = subparsers.add_parser(
        "dt-rolling-run",
        help="run resumable 5-seed/3-fold DT-SeatHGT evaluation on verified snapshots",
    )
    _add_config_argument(dt_rolling)
    dt_rolling.add_argument("--data-root", type=Path, required=True)
    dt_rolling.add_argument("--output-root", type=Path, required=True)
    dt_rolling.add_argument("--result-file", type=Path, required=True)
    dt_rolling.add_argument("--as-of-start", type=_parse_date, required=True)
    dt_rolling.add_argument("--as-of-end", type=_parse_date, required=True)
    dt_rolling.add_argument("--observed-label-end", type=_parse_date, required=True)
    dt_rolling.add_argument("--seed", type=int, action="append")
    dt_rolling.add_argument(
        "--diagnostic-e1",
        action="store_true",
        help="run only the D1 diagnostic: one seed, one fold, and 3-10 epochs",
    )
    dt_rolling.add_argument(
        "--diagnostic-e2",
        action="store_true",
        help="run only the D2 numerical-stability diagnostic: one seed, one fold, and 3-10 epochs",
    )
    dt_rolling.add_argument(
        "--diagnostic-e3",
        action="store_true",
        help="run only the D3 return-calibration diagnostic: one seed, one fold, and 3-10 epochs",
    )
    dt_rolling.add_argument("--epochs", type=int, default=3)
    dt_rolling.add_argument("--learning-rate", type=float, default=0.001)
    dt_rolling.add_argument("--gradient-clip-norm", type=float, default=1.0)
    dt_rolling.add_argument("--selection-horizon", type=int, default=1)
    dt_rolling.add_argument("--min-train-dates", type=int, default=60)
    dt_rolling.add_argument("--top-k", type=int, default=5)
    dt_rolling.add_argument("--cooccurrence-cap", type=int, default=20)
    dt_rolling.add_argument(
        "--chunk-dates",
        type=int,
        default=30,
        help="number of as-of dates materialized per bounded graph-construction chunk",
    )
    dt_rolling.add_argument(
        "--research-protocol-config",
        type=Path,
        default=Path("configs/experiments/research_protocol_v2.toml"),
    )

    repair_research_cache = subparsers.add_parser(
        "repair-research-cache",
        help="refresh full-range research daily caches and fail if execution bars remain missing",
    )
    _add_config_argument(repair_research_cache)
    _add_provider_argument(repair_research_cache)
    repair_research_cache.add_argument("--data-root", type=Path, required=True)
    repair_research_cache.add_argument("--start", type=_parse_date, required=True)
    repair_research_cache.add_argument("--end", type=_parse_date, required=True)
    repair_research_cache.add_argument("--missing-execution-jsonl", type=Path, required=True)
    repair_research_cache.add_argument("--output", type=Path, required=True)

    export_run = subparsers.add_parser(
        "export-run-package",
        help="export a redacted reproducibility package from append-only lineage",
    )
    _add_config_argument(export_run)
    export_run.add_argument("--lineage-path", type=Path, required=True)
    export_run.add_argument("--run-id", required=True)
    export_run.add_argument("--output-root", type=Path, default=Path("artifacts/run_packages"))
    export_run.add_argument("--include-artifact", type=Path, action="append", default=[])

    phase_status = subparsers.add_parser(
        "phase-status", help="show the explicit fail-closed project completion state"
    )
    _add_config_argument(phase_status)
    phase_status.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/phase-state.json")
    )

    evaluate_promotion = subparsers.add_parser(
        "evaluate-promotion",
        help="evaluate audited rolling metrics and persist a fail-closed promotion decision",
    )
    _add_config_argument(evaluate_promotion)
    evaluate_promotion.add_argument("--metrics-file", type=Path, required=True)
    evaluate_promotion.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/promotion-decision.json")
    )
    evaluate_promotion.add_argument(
        "--status-file", type=Path, default=Path("artifacts/status/phase-state.json")
    )

    freeze = subparsers.add_parser(
        "freeze-protocol", help="freeze an experiment protocol before comparable training"
    )
    _add_config_argument(freeze)
    freeze.add_argument(
        "--protocol-config", type=Path, default=Path("configs/experiments/protocol.toml")
    )
    freeze.add_argument("--output-root", type=Path, default=Path("artifacts/protocol"))

    freeze_blind = subparsers.add_parser(
        "freeze-blind-manifest", help="freeze blind feature and label artifact hashes"
    )
    _add_config_argument(freeze_blind)
    freeze_blind.add_argument("--feature-file", type=Path, action="append", required=True)
    freeze_blind.add_argument("--label-file", type=Path, action="append", required=True)
    freeze_blind.add_argument("--output", type=Path, required=True)

    artifacts = subparsers.add_parser(
        "build-artifacts", help="build basic features and an as-of graph from ingested snapshots"
    )
    _add_config_argument(artifacts)
    _add_provider_argument(artifacts)
    artifacts.add_argument("--trade-date", type=_parse_date, required=True)
    artifacts.add_argument("--data-root", type=Path, help="defaults to the parent of data.raw_dir")

    promote_graph = subparsers.add_parser(
        "promote-graph",
        help="attach verified real-data audit evidence to an existing graph snapshot",
    )
    _add_config_argument(promote_graph)
    _add_provider_argument(promote_graph)
    promote_graph.add_argument("--trade-date", type=_parse_date, required=True)
    promote_graph.add_argument(
        "--data-root", type=Path, help="defaults to the parent of data.raw_dir"
    )
    promote_graph.add_argument("--audit-report", type=Path, required=True)
    promote_graph.add_argument("--require-complete-seat", action="store_true")

    paper_init = subparsers.add_parser(
        "paper-init", help="initialize a persistent local paper-trading account"
    )
    _add_config_argument(paper_init)
    _add_paper_arguments(paper_init)

    paper_status = subparsers.add_parser(
        "paper-status", help="show cash and positions in the local paper account"
    )
    _add_config_argument(paper_status)
    _add_paper_arguments(paper_status)
    paper_status.add_argument(
        "--quotes-file", type=Path, help="optional quote snapshot used to mark positions"
    )

    paper_run = subparsers.add_parser(
        "paper-run", help="execute one idempotent automatic paper-trading rebalance"
    )
    _add_config_argument(paper_run)
    _add_paper_arguments(paper_run)
    paper_run.add_argument("--signals-file", type=Path, required=True)
    paper_run.add_argument("--quotes-file", type=Path, required=True)
    paper_run.add_argument(
        "--cycle-id",
        help="idempotency key; defaults to signal date plus execution date",
    )

    serve = subparsers.add_parser("serve-api", help="serve read-only persisted snapshots over HTTP")
    _add_config_argument(serve)
    _add_provider_argument(serve)
    serve.add_argument("--data-root", type=Path, help="defaults to the parent of data.raw_dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))


def _add_provider_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("sample", "tushare", "akshare"))


def _add_paper_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--paper-config",
        type=Path,
        default=Path("configs/paper_trading/local.toml"),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("artifacts/paper_trading/account.json"),
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def _synthetic_records(as_of_date: date) -> tuple[DragonTigerRecord, ...]:
    return (
        DragonTigerRecord(
            trade_date=as_of_date,
            seat_id="seat_institution_a",
            stock_id="600000.SH",
            side=TradeSide.BUY,
            amount=12_000_000,
            industry="banking",
            concepts=("large_cap",),
            event_reason="daily_deviation",
        ),
        DragonTigerRecord(
            trade_date=as_of_date - timedelta(days=2),
            seat_id="seat_momentum_b",
            stock_id="600000.SH",
            side=TradeSide.BUY,
            amount=7_000_000,
            industry="banking",
            concepts=("large_cap",),
            event_reason="consecutive_listing",
        ),
        DragonTigerRecord(
            trade_date=as_of_date - timedelta(days=1),
            seat_id="seat_arbitrage_c",
            stock_id="600000.SH",
            side=TradeSide.SELL,
            amount=4_000_000,
            industry="banking",
            concepts=("large_cap",),
            event_reason="daily_deviation",
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_workspace_env(Path.cwd() / ".env")
    config = load_config(args.config)

    try:
        if args.command == "validate-config":
            _print_json(asdict(config))
            return
        if args.command == "phase-status":
            _print_json(load_phase_status(args.status_file).to_dict())
            return
        if args.command == "evaluate-promotion":
            _run_promotion_evaluation(args)
            return
        if args.command == "smoke":
            _run_smoke(config)
            return
        if args.command == "train-smoke":
            _run_training_smoke(args)
            return
        if args.command == "research-run":
            _run_historical_research(args, config)
            return
        if args.command == "dt-rolling-run":
            _run_dt_rolling(args, config)
            return
        if args.command == "export-run-package":
            _print_json(
                export_run_package(
                    args.lineage_path,
                    args.run_id,
                    args.output_root,
                    include_artifacts=tuple(args.include_artifact),
                ).to_dict()
            )
            return
        if args.command == "freeze-protocol":
            _print_json(
                freeze_protocol(
                    args.output_root, load_experiment_protocol(args.protocol_config)
                ).to_dict()
            )
            return
        if args.command == "freeze-blind-manifest":
            _print_json(
                freeze_blind_manifest(
                    args.output,
                    feature_paths=args.feature_file,
                    label_paths=args.label_file,
                ).to_dict()
            )
            return
        if args.command == "paper-init":
            paper_config = load_paper_trading_config(args.paper_config)
            broker = JsonPaperBroker(args.state_file, paper_config)
            _print_json(
                {
                    "state_file": str(args.state_file),
                    "config": paper_config.to_dict(),
                    "account": broker.initialize().to_dict(),
                }
            )
            return
        if args.command == "paper-status":
            paper_config = load_paper_trading_config(args.paper_config)
            broker = JsonPaperBroker(args.state_file, paper_config)
            if args.quotes_file is None:
                trade_date = date.today()
                quotes = {}
            else:
                quote_batch = load_quote_batch(args.quotes_file)
                trade_date = quote_batch.trade_date
                quotes = {quote.stock_id: quote for quote in quote_batch.quotes}
            _print_json(broker.snapshot(trade_date, quotes).to_dict())
            return
        if args.command == "paper-run":
            paper_config = load_paper_trading_config(args.paper_config)
            signal_batch = load_signal_batch(args.signals_file)
            quote_batch = load_quote_batch(args.quotes_file)
            cycle_id = args.cycle_id or (
                f"{signal_batch.as_of_date.isoformat()}-to-{quote_batch.trade_date.isoformat()}"
            )
            broker = JsonPaperBroker(args.state_file, paper_config)
            report = PaperTradingService(broker, paper_config).run(
                cycle_id=cycle_id,
                signal_batch=signal_batch,
                quote_batch=quote_batch,
            )
            _print_json(report.to_dict())
            return

        provider_name = args.provider or config.data.provider
        provider = create_provider(provider_name, config.data.available_after_close)
        if args.command == "provider-status":
            _print_json(provider.status().to_dict())
            return
        if args.command == "preflight":
            _run_preflight(config, provider.status().to_dict())
            return

        if args.command == "audit-data":
            _run_data_audit(args, provider)
            return
        if args.command == "repair-research-cache":
            _run_research_cache_repair(args, provider)
            return

        if args.command == "serve-api":
            _run_api_server(args, config)
            return

        data_root = args.data_root or config.data.raw_dir.parent
        service = DataIngestionService(provider, data_root)
        if args.command == "ingest":
            result = service.ingest_market_day(
                args.trade_date,
                stock_id=args.stock_code,
                include_seat_detail=not args.skip_seat_detail,
                require_seat_detail=args.require_seat_detail,
            )
            _print_json(result.to_dict())
            return
        if args.command == "backfill-history":
            _run_historical_backfill(args, service)
            return
        if args.command == "build-artifacts":
            cutoff_time = datetime.combine(
                args.trade_date, config.data.available_after_close, ZoneInfo("Asia/Shanghai")
            )
            feature_artifacts = service.build_features()
            graph_artifact = service.build_graph_snapshot(
                as_of_date=args.trade_date,
                decay_lambda=config.graph.decay_lambda,
                max_edge_age_days=config.graph.max_edge_age_days,
                cutoff_time=cutoff_time,
            )
            _print_json(
                {
                    "provider": provider.name,
                    "feature_artifacts": [artifact.to_dict() for artifact in feature_artifacts],
                    "graph_artifact": graph_artifact.to_dict() if graph_artifact else None,
                }
            )
            return
        if args.command == "promote-graph":
            artifact = service.promote_graph_snapshot(
                as_of_date=args.trade_date,
                audit_report_path=args.audit_report,
                require_complete_seat=args.require_complete_seat,
            )
            _print_json(artifact.to_dict())
            return
    except (OSError, PaperBrokerError, ProviderError, ValueError) as error:
        parser.error(str(error))

    parser.error(f"unsupported command: {args.command}")


def _run_smoke(config: ProjectConfig) -> None:
    as_of_date = date(2026, 1, 15)
    builder = DynamicGraphBuilder(
        decay_lambda=config.graph.decay_lambda,
        max_edge_age_days=config.graph.max_edge_age_days,
    )
    graph = builder.build(_synthetic_records(as_of_date), as_of_date)
    predictor = HeuristicSmokePredictor(config.graph.prediction_horizons)
    prediction = predictor.predict(ModelInput(graph=graph, target_stock_id="600000.SH"))
    _print_json(
        {
            "project": config.name,
            "as_of_date": graph.as_of_date.isoformat(),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "prediction": asdict(prediction),
        }
    )


def _run_preflight(config: ProjectConfig, provider_status: dict[str, object]) -> None:
    split = TimeSplit(
        train_start=config.training.train_start,
        train_end=config.training.train_end,
        valid_start=config.training.valid_start,
        valid_end=config.training.valid_end,
        test_start=config.training.test_start,
        test_end=config.training.test_end,
    )
    split_errors = validate_time_split(split)
    _print_json(
        {
            "passed": not split_errors,
            "time_split": split.to_dict(),
            "time_split_errors": list(split_errors),
            "data_provider": provider_status,
            "warnings": [
                "available_after_close is configurable; verify source publication time.",
                "A complete SeatHGT graph requires seat details, not only top_list summaries.",
            ],
        }
    )


def _run_api_server(args: argparse.Namespace, config: ProjectConfig) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise ProviderError(
            "API dependencies are not installed. Run: python -m pip install -e .[api]"
        ) from error

    from seatflow.api import create_app

    data_root = args.data_root or config.data.raw_dir.parent
    uvicorn.run(
        create_app(data_root, args.provider or config.data.provider), host=args.host, port=args.port
    )


def _run_training_smoke(args: argparse.Namespace) -> None:
    if args.model == "baseline":
        result = run_engineering_smoke(args.config, args.output_root, args.run_id)
    else:
        result = run_pyg_shape_smoke(args.config, args.output_root, args.run_id)
    _print_json(result.to_dict())


def _run_historical_research(args: argparse.Namespace, config: ProjectConfig) -> None:
    seeds = tuple(args.seed) if args.seed else (2026, 2027, 2028)
    result = run_historical_research(
        HistoricalResearchConfig(
            start_date=args.start,
            end_date=args.end,
            data_root=args.data_root,
            output_root=args.output_root,
            seeds=seeds,
            top_n=args.top_n,
            minimum_score=args.minimum_score,
            akshare_sample_count=args.akshare_sample_count,
            require_akshare_confirmation=args.require_akshare_confirmation,
            protocol_config=args.research_protocol_config,
        ),
        TushareHttpProvider(available_after_close=config.data.available_after_close),
        AkshareProvider(available_after_close=config.data.available_after_close),
    )
    _print_json(result.to_dict())


def _run_dt_rolling(args: argparse.Namespace, config: ProjectConfig) -> None:
    protocol = load_experiment_protocol(args.research_protocol_config)
    split_status = _dt_split_status(args.as_of_start, args.as_of_end, protocol)
    audit: dict[str, object] = {
        "verified": False,
        "data_scope": "real",
        "seat_detail_scope": "institutional_proxy",
        "split_status": split_status,
        "as_of_start": args.as_of_start.isoformat(),
        "as_of_end": args.as_of_end.isoformat(),
        "observed_label_end": args.observed_label_end.isoformat(),
    }
    try:
        if args.as_of_end < args.as_of_start:
            raise ValueError("as_of_end must not be before as_of_start")
        future_blind_start = protocol.window("future_blind").start
        if args.as_of_end >= future_blind_start:
            raise ValueError("DT rolling run refuses future-blind feature dates")
        if args.observed_label_end >= future_blind_start:
            raise ValueError("DT rolling run refuses unreleased future-blind labels")
        if args.observed_label_end < args.as_of_end:
            raise ValueError("observed_label_end must not be before as_of_end")
        if args.epochs <= 0 or args.min_train_dates <= 0:
            raise ValueError("epochs and min_train_dates must be positive")
        if args.cooccurrence_cap < 2:
            raise ValueError("cooccurrence_cap must be at least two")

        seeds = tuple(args.seed) if args.seed else protocol.seeds
        diagnostic_flags = (args.diagnostic_e1, args.diagnostic_e2, args.diagnostic_e3)
        if sum(diagnostic_flags) > 1:
            raise ValueError("only one diagnostic gate may be selected")
        diagnostic_gate = (
            "E1"
            if args.diagnostic_e1
            else "E2"
            if args.diagnostic_e2
            else "E3"
            if args.diagnostic_e3
            else None
        )
        if diagnostic_gate:
            if len(seeds) != 1 or len(set(seeds)) != 1:
                raise ValueError(f"{diagnostic_gate} diagnostic requires exactly one unique seed")
            if not 3 <= args.epochs <= 10:
                raise ValueError(f"{diagnostic_gate} diagnostic requires 3-10 epochs")
            fold_count = 1
        else:
            if len(seeds) != 5 or len(set(seeds)) != 5:
                raise ValueError("DT rolling run requires exactly 5 unique seeds")
            fold_count = 3
        full_calendar = _snapshot_calendar(args.data_root, "tushare")
        selected_dates = tuple(
            item for item in full_calendar if args.as_of_start <= item <= args.as_of_end
        )
        if not selected_dates:
            raise ValueError("no verified snapshot dates are inside the requested as-of range")
        try:
            observed_index = full_calendar.index(args.observed_label_end)
        except ValueError as error:
            raise ValueError("observed_label_end is absent from snapshot calendar") from error
        chunk_root = args.output_root / "tensorized_chunks"
        chunk_root.mkdir(parents=True, exist_ok=True)
        chunk_files: list[Path] = []
        chunk_sample_count = 0
        frozen_schema = None
        matured_label_cache = MaturedLabelDiskCache(
            args.output_root / "matured_labels.sqlite3"
        )
        manifest_hashes: dict[str, str] = {}
        target_dates_seen: set[date] = set()
        loaded_dates_seen: set[date] = set()
        if args.chunk_dates <= 0:
            raise ValueError("chunk_dates must be positive")
        max_horizon = max(config.graph.prediction_horizons)
        for chunk_start in range(0, len(selected_dates), args.chunk_dates):
            chunk_dates = selected_dates[chunk_start : chunk_start + args.chunk_dates]
            first_chunk_index = full_calendar.index(chunk_dates[0])
            last_chunk_index = full_calendar.index(chunk_dates[-1])
            load_start_index = max(0, first_chunk_index - config.data.lookback_days - 1)
            load_end_index = min(observed_index, last_chunk_index + max_horizon)
            loaded_calendar = full_calendar[load_start_index : load_end_index + 1]
            snapshots = load_verified_historical_snapshots(
                args.data_root,
                provider="tushare",
                trading_calendar=loaded_calendar,
            )
            manifest_hashes.update(snapshots.manifest_hashes)
            loaded_dates_seen.update(loaded_calendar)
            selected_set = set(chunk_dates)
            targets_by_date: dict[date, set[str]] = {}
            for event in snapshots.seat_events:
                if event.trade_date in selected_set:
                    targets_by_date.setdefault(event.trade_date, set()).add(event.stock_id)
            chunk_targets = {
                trade_date: tuple(sorted(stock_ids))
                for trade_date, stock_ids in sorted(targets_by_date.items())
            }
            target_dates_seen.update(chunk_targets)
            if not chunk_targets:
                del snapshots
                continue
            windows = build_real_window_samples(
                snapshots,
                target_stock_ids_by_date=chunk_targets,
                cutoff_for_date=lambda trade_date: datetime.combine(
                    trade_date,
                    config.data.available_after_close,
                    ZoneInfo("Asia/Shanghai"),
                ),
                observed_label_end=min(args.observed_label_end, loaded_calendar[-1]),
                decay_lambda=config.graph.decay_lambda,
                max_edge_age_days=config.graph.max_edge_age_days,
                cooccurrence_max_records_per_stock=args.cooccurrence_cap,
                matured_label_cache=matured_label_cache,
            )
            if frozen_schema is None:
                frozen_schema = derive_frozen_graph_schema(
                    sample.graph for sample in windows
                )
            tensorized_chunk = tensorize_window_samples(
                windows, frozen_schema=frozen_schema
            )
            if tensorized_chunk:
                chunk_path = chunk_root / f"chunk-{chunk_start:06d}.pkl"
                with chunk_path.open("wb") as stream:
                    pickle.dump(
                        tensorized_chunk,
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                chunk_files.append(chunk_path)
                chunk_sample_count += len(tensorized_chunk)
            del tensorized_chunk
            del windows
            del snapshots
        matured_label_cache.close()
        folds = _build_spilled_dt_folds(
            chunk_files,
            selected_dates,
            selection_horizon=args.selection_horizon,
            min_train_dates=args.min_train_dates,
            purge_days=protocol.purge_days,
            embargo_days=protocol.embargo_days,
            n_folds=fold_count,
        )
        evaluation = run_dt_seathgt_folds_evaluation(
            folds,
            DTSeatHGTTrainingConfig(
                model=DTSeatHGTConfig(
                    hidden_dim=config.model.hidden_dim,
                    num_heads=config.model.num_heads,
                    num_layers=config.model.num_layers,
                    dropout=config.model.dropout,
                    horizons=config.graph.prediction_horizons,
                ),
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                seeds=seeds,
                gradient_clip_norm=args.gradient_clip_norm,
            ),
            args.output_root,
            selection_horizon=args.selection_horizon,
            seeds=seeds,
            top_k=args.top_k,
            diagnostic_e1=bool(diagnostic_gate),
        )
        audit.update(
            {
                "verified": True,
                "snapshot_manifest_hash": sha256(
                    json.dumps(
                        dict(sorted(manifest_hashes.items())),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "snapshot_manifest_count": len(manifest_hashes),
                "loaded_calendar_days": len(loaded_dates_seen),
                "requested_as_of_days": len(selected_dates),
                "event_as_of_days": len(target_dates_seen),
                "sample_count": chunk_sample_count,
                "seed_count": len(seeds),
                "fold_count": len(evaluation.folds),
                "diagnostic_gate": diagnostic_gate,
                "diagnostic_e1": bool(args.diagnostic_e1),
                "diagnostic_e2": bool(args.diagnostic_e2),
                "diagnostic_e3": bool(args.diagnostic_e3),
                "cooccurrence_max_records_per_stock": args.cooccurrence_cap,
                "chunk_dates": args.chunk_dates,
                "tensorized_chunk_count": len(chunk_files),
                "tensorized_chunk_root": str(chunk_root),
            }
        )
        result_path = save_dt_rolling_result(
            args.result_file,
            evaluation,
            audit=audit,
            split_status=split_status,
        )
        _print_json(
            {
                "result_path": str(result_path),
                "checkpoint_root": str(args.output_root),
                "sample_count": chunk_sample_count,
                "seed_count": len(seeds),
                "fold_count": len(evaluation.folds),
                "ablation_count": len(evaluation.ablations),
                "model_valid": False,
                "negative_result": True,
            }
        )
    except Exception as error:
        save_dt_rolling_failure(
            args.result_file,
            error=error,
            audit=audit,
            split_status=split_status,
        )
        if isinstance(error, (OSError, ValueError)):
            raise
        raise ValueError(f"DT rolling run failed: {error}") from error


def _snapshot_calendar(data_root: Path, provider: str) -> tuple[date, ...]:
    dataset_root = data_root / "processed" / provider / "daily_bar"
    if not dataset_root.exists():
        raise ValueError(f"daily-bar snapshot directory does not exist: {dataset_root}")
    dates = tuple(
        sorted(
            date.fromisoformat(path.name.removeprefix("date="))
            for path in dataset_root.glob("date=*")
            if path.is_dir()
        )
    )
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("snapshot calendar is empty or contains duplicate dates")
    return dates


def _build_spilled_dt_folds(
    chunk_files: Sequence[Path],
    selected_dates: Sequence[date],
    *,
    selection_horizon: int,
    min_train_dates: int,
    purge_days: int,
    embargo_days: int,
    n_folds: int = 3,
) -> tuple[DTRollingFold, ...]:
    """Load one rolling fold at a time from tensorized chunk spill files."""

    if not chunk_files:
        raise ValueError("DT rolling spill produced no tensorized chunks")
    if min_train_dates <= 0:
        raise ValueError("min_train_dates must be positive")
    if purge_days < 0 or embargo_days < 0:
        raise ValueError("purge_days and embargo_days must be non-negative")
    dates = tuple(sorted(set(selected_dates)))
    evaluation_dates = dates[min_train_dates:]
    if n_folds <= 0:
        raise ValueError("rolling spill fold count must be positive")
    test_date_chunks = _contiguous_date_chunks(evaluation_dates, n_folds)
    folds: list[DTRollingFold] = []
    for index, test_dates in enumerate(test_date_chunks):
        test_start = test_dates[0]
        test_end = test_dates[-1]
        train_as_of_cutoff = test_start - timedelta(days=embargo_days)
        train_label_cutoff = test_start - timedelta(days=purge_days)
        train: list[TensorizedWindowSample] = []
        test: list[TensorizedWindowSample] = []
        for chunk_path in chunk_files:
            with chunk_path.open("rb") as stream:
                chunk = pickle.load(stream)
            if not isinstance(chunk, tuple):
                raise ValueError(f"tensorized spill chunk is malformed: {chunk_path}")
            for sample in chunk:
                if not isinstance(sample, TensorizedWindowSample):
                    raise ValueError(f"tensorized spill contains an invalid sample: {chunk_path}")
                _require_current_tensor_schema(sample, chunk_path)
                as_of_date = sample.as_of_date
                label_end = sample.label_end_dates.get(selection_horizon)
                if as_of_date is None or label_end is None:
                    raise ValueError(
                        f"tensorized sample metadata is incomplete: {sample.sample_id}"
                    )
                if as_of_date < train_as_of_cutoff and label_end < train_label_cutoff:
                    train.append(sample)
                if test_start <= as_of_date <= test_end:
                    test.append(sample)
            del chunk
        if not train:
            raise ValueError(f"rolling spill fold {index} has empty train partition")
        if not test:
            raise ValueError(f"rolling spill fold {index} has empty test partition")
        train.sort(key=lambda item: (_sample_as_of_date(item), item.sample_id))
        test.sort(key=lambda item: (_sample_as_of_date(item), item.sample_id))
        folds.append(
            DTRollingFold(
                fold_id=f"fold_{index}",
                train_start=_sample_as_of_date(train[0]),
                train_end=max(_sample_as_of_date(item) for item in train),
                test_start=test_start,
                test_end=test_end,
                train=tuple(train),
                test=tuple(test),
            )
        )
        del train
        del test
    return tuple(folds)


def _require_current_tensor_schema(sample: TensorizedWindowSample, chunk_path: Path) -> None:
    metadata = sample.graph.metadata
    if metadata.get("tensor_schema_version") != TENSORIZATION_VERSION:
        raise ValueError(
            "tensorized spill uses an obsolete or unversioned schema and must be regenerated: "
            f"{chunk_path}"
        )
    schema_hash = metadata.get("edge_feature_schema_hash")
    if not isinstance(schema_hash, str) or len(schema_hash) != 64:
        raise ValueError(f"tensorized spill is missing its schema fingerprint: {chunk_path}")


def _contiguous_date_chunks(
    values: Sequence[date], count: int
) -> tuple[tuple[date, ...], ...]:
    ordered = tuple(sorted(values))
    if len(ordered) < count:
        raise ValueError("cannot create non-empty rolling folds from spill dates")
    chunks: list[tuple[date, ...]] = []
    start = 0
    for index in range(count):
        remaining_values = len(ordered) - start
        remaining_chunks = count - index
        size = (remaining_values + remaining_chunks - 1) // remaining_chunks
        chunk = ordered[start : start + size]
        if not chunk:
            raise ValueError("rolling spill fold date chunk is empty")
        chunks.append(chunk)
        start += size
    return tuple(chunks)


def _sample_as_of_date(sample: TensorizedWindowSample) -> date:
    if sample.as_of_date is None:
        raise ValueError(f"tensorized sample is missing as_of_date: {sample.sample_id}")
    return sample.as_of_date


def _dt_split_status(start: date, end: date, protocol: ExperimentProtocol) -> str:
    contaminated = protocol.window("contaminated_oos")
    if contaminated.start <= start and end <= contaminated.end:
        return "contaminated_oos"
    return "development_rolling"


def _run_research_cache_repair(
    args: argparse.Namespace, provider: MarketDataProvider
) -> None:
    if not isinstance(provider, TushareHttpProvider):
        raise ProviderError("repair-research-cache requires the Tushare provider")
    report = repair_research_cache(
        provider=provider,
        data_root=args.data_root,
        start_date=args.start,
        end_date=args.end,
        missing_execution_jsonl=args.missing_execution_jsonl,
    )
    report_path = write_research_cache_repair_report(args.output, report)
    _print_json({"report_path": str(report_path), **report.to_dict()})
    if not report.passed:
        raise SystemExit(1)


def _run_promotion_evaluation(args: argparse.Namespace) -> None:
    try:
        raw_metrics = json.loads(args.metrics_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError("metrics file must contain a JSON object") from error
    if not isinstance(raw_metrics, dict):
        raise ValueError("metrics file must contain a JSON object")

    decision = evaluate_promotion_decision(raw_metrics)
    decision_path = save_promotion_decision(args.output, decision)
    prior_status = load_phase_status(args.status_file)
    evidence_hashes = dict(prior_status.evidence_hashes)
    evidence_hashes["promotion_decision"] = sha256(decision_path.read_bytes()).hexdigest()
    status = phase_status_from_promotion(
        engineering_complete=prior_status.engineering_complete,
        data_ready=prior_status.data_ready,
        scope=prior_status.scope,
        data_scope=prior_status.data_scope,
        evidence_hashes=evidence_hashes,
        promotion_decision=decision,
    )
    status_path = write_phase_status(args.status_file, status)
    _print_json(
        {
            "decision_path": str(decision_path),
            "status_path": str(status_path),
            "decision": decision.to_dict(),
            "phase_status": status.to_dict(),
        }
    )


def _run_data_audit(args: argparse.Namespace, provider: MarketDataProvider) -> None:
    if args.end < args.start:
        raise ProviderError("audit end date must not be before start date")
    trade_dates = _load_verified_trade_dates(args.trading_calendar_file, args.start, args.end)
    policy = load_audit_policy(args.audit_config)
    report = audit_market_data(
        provider,
        trade_dates,
        require_seat_detail=args.require_audit_seat_detail,
        policy=policy,
    )
    output = args.output or Path(
        f"artifacts/data_audit/{provider.name}-{args.start.isoformat()}-{args.end.isoformat()}.json"
    )
    write_audit_report(output, report)
    _print_json({"report_path": str(output), **report.to_dict()})


def _run_historical_backfill(args: argparse.Namespace, service: DataIngestionService) -> None:
    if args.start is not None and args.end is not None and args.end < args.start:
        raise ProviderError("historical backfill end date must not be before start date")
    trade_dates = _load_historical_trade_dates(args.trading_calendar_file, args.start, args.end)
    result = service.backfill_history(
        trade_dates,
        stock_id=args.stock_code,
        include_seat_detail=not args.skip_seat_detail,
        require_seat_detail=args.require_seat_detail,
        skip_verified=not args.no_skip_verified,
        continue_on_error=args.continue_on_error,
    )
    report_path = write_historical_backfill_report(args.output, result)
    summary_path = None
    if args.summary is not None:
        summary_path = write_historical_backfill_summary(args.summary, result)
    _print_json(
        {
            "report_path": str(report_path),
            "summary_path": str(summary_path) if summary_path is not None else None,
            **result.to_dict(),
        }
    )
    if not result.passed:
        raise SystemExit(1)


def _load_verified_trade_dates(path: Path, start: date, end: date) -> tuple[date, ...]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ProviderError("trading calendar file is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = [line.strip() for line in content.splitlines() if line.strip()]
    if not isinstance(parsed, list):
        raise ProviderError("trading calendar file must be a JSON array or one date per line")
    try:
        dates = tuple(_parse_date(str(value)) for value in parsed)
    except argparse.ArgumentTypeError as error:
        raise ProviderError("trading calendar contains an invalid date") from error
    filtered = tuple(item for item in dates if start <= item <= end)
    if not filtered:
        raise ProviderError("trading calendar has no dates inside the requested audit window")
    if len(set(filtered)) != len(filtered):
        raise ProviderError("trading calendar contains duplicate dates in the audit window")
    return filtered


def _load_historical_trade_dates(
    path: Path, start: date | None = None, end: date | None = None
) -> tuple[date, ...]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ProviderError("trading calendar file is empty")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ProviderError("historical trading calendar must be JSON") from error
    values: object
    if isinstance(parsed, dict):
        values = parsed.get("trade_dates", parsed.get("dates"))
    else:
        values = parsed
    if not isinstance(values, list):
        raise ProviderError(
            "historical trading calendar must be a JSON array or object with trade_dates"
        )
    try:
        dates = tuple(_parse_date(str(value)) for value in values)
    except argparse.ArgumentTypeError as error:
        raise ProviderError("historical trading calendar contains an invalid date") from error
    if start is not None:
        dates = tuple(item for item in dates if item >= start)
    if end is not None:
        dates = tuple(item for item in dates if item <= end)
    if not dates:
        raise ProviderError("historical trading calendar has no dates inside the requested window")
    if len(set(dates)) != len(dates):
        raise ProviderError("historical trading calendar contains duplicate dates")
    return dates


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
