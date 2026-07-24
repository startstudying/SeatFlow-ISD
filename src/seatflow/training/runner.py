from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

from seatflow.config import load_config
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import PyGUnavailableError, tensorize_hetero_graph
from seatflow.models.baselines import EngineeringSmokeBaseline, prediction_to_dict
from seatflow.models.seat_hgt import (
    DTSeatHGTConfig,
    PyGModelUnavailableError,
    build_dt_seat_hgt,
    dt_seat_hgt_checkpoint_payload,
)
from seatflow.tracking.lineage import (
    LineageRecord,
    create_blind_manifest,
    now_iso,
    record_training_run,
)
from seatflow.training.checkpointing import hash_file, hash_json, write_json_checkpoint
from seatflow.training.preflight import TimeSplit, validate_time_split


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    status: str
    model_valid: bool
    lineage_path: Path
    checkpoint_path: Path
    hashes: dict[str, str]
    metrics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "model_valid": self.model_valid,
            "lineage_path": str(self.lineage_path),
            "checkpoint_path": str(self.checkpoint_path),
            "hashes": dict(self.hashes),
            "metrics": dict(self.metrics),
        }


def run_engineering_smoke(
    config_path: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
) -> TrainingRunResult:
    config = load_config(config_path)
    if config.data.provider != "sample":
        raise ValueError(
            "engineering_smoke runner is fail-closed and only accepts provider='sample'"
        )
    split = TimeSplit(
        train_start=config.training.train_start,
        train_end=config.training.train_end,
        valid_start=config.training.valid_start,
        valid_end=config.training.valid_end,
        test_start=config.training.test_start,
        test_end=config.training.test_end,
    )
    split_errors = validate_time_split(split)
    if split_errors:
        raise ValueError(f"invalid time split: {'; '.join(split_errors)}")

    effective_run_id = run_id or f"engineering_smoke-{uuid4().hex}"
    root = Path(output_root)
    graph = DynamicGraphBuilder(
        decay_lambda=config.graph.decay_lambda,
        max_edge_age_days=config.graph.max_edge_age_days,
    ).build(_sample_records(date(2026, 1, 15)), date(2026, 1, 15))
    baseline = EngineeringSmokeBaseline(config.graph.prediction_horizons)
    prediction = baseline.predict(graph, "000001.SZ")

    data_payload = {
        "as_of_date": graph.as_of_date.isoformat(),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "target_stock_id": prediction.target_stock_id,
    }
    blind_manifest = create_blind_manifest(
        root / "blind_manifests",
        effective_run_id,
        (
            {
                "example_id": "engineering-smoke-000001.SZ-2026-01-15",
                "as_of_date": graph.as_of_date.isoformat(),
                "target_stock_id": "000001.SZ",
            },
        ),
    )
    metrics: dict[str, object] = {
        "sample_count": 1,
        "rise_probability": prediction.rise_probability,
        "engineering_smoke": True,
        "model_valid": False,
    }
    checkpoint = write_json_checkpoint(
        root / "checkpoints",
        effective_run_id,
        {
            "baseline": baseline.checkpoint_payload(),
            "prediction": prediction_to_dict(prediction),
        },
        metadata={"run_type": "engineering_smoke", "model_valid": False},
    )
    hashes = {
        "data": hash_json(data_payload),
        "config": hash_file(config_path),
        "code": _code_hash(),
        "seed": hash_json({"seed": config.seed}),
        "split": hash_json(split.to_dict()),
        "blind_manifest": blind_manifest.sha256,
        "checkpoint": checkpoint.sha256,
    }
    record = LineageRecord(
        run_id=effective_run_id,
        run_type="engineering_smoke",
        status="completed",
        model_valid=False,
        provider=config.data.provider,
        seed=config.seed,
        split=split,
        hashes=hashes,
        metrics=metrics,
        tags=("engineering_smoke", "sample_only", "not_model_valid"),
        created_at=now_iso(),
    )
    tracking = record_training_run(record, root)
    return TrainingRunResult(
        run_id=effective_run_id,
        status="completed",
        model_valid=False,
        lineage_path=tracking.lineage_path,
        checkpoint_path=checkpoint.checkpoint_path,
        hashes=hashes,
        metrics=metrics,
    )


def run_pyg_shape_smoke(
    config_path: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
) -> TrainingRunResult:
    config = load_config(config_path)
    if config.data.provider != "sample":
        raise ValueError("pyg_shape_smoke is fail-closed and only accepts provider='sample'")
    split = TimeSplit(
        train_start=config.training.train_start,
        train_end=config.training.train_end,
        valid_start=config.training.valid_start,
        valid_end=config.training.valid_end,
        test_start=config.training.test_start,
        test_end=config.training.test_end,
    )
    split_errors = validate_time_split(split)
    if split_errors:
        raise ValueError(f"invalid time split: {'; '.join(split_errors)}")

    effective_run_id = run_id or f"pyg_shape_smoke-{uuid4().hex}"
    root = Path(output_root)
    graph = DynamicGraphBuilder(
        decay_lambda=config.graph.decay_lambda,
        max_edge_age_days=config.graph.max_edge_age_days,
    ).build(_sample_records(date(2026, 1, 15)), date(2026, 1, 15))
    try:
        tensorized = tensorize_hetero_graph(graph)
    except PyGUnavailableError as error:
        raise PyGModelUnavailableError(str(error)) from error

    model_config = DTSeatHGTConfig(
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        horizons=config.graph.prediction_horizons,
    )
    model = build_dt_seat_hgt(model_config, tensorized)
    outputs = model(tensorized.data)
    direction_logits = outputs["direction_logits"]
    return_predictions = outputs["return_predictions"]
    if not isinstance(direction_logits, dict) or not isinstance(return_predictions, dict):
        raise ValueError("DTSeatHGT forward must return multi-task head dictionaries")

    output_shapes = {
        "direction_logits": {
            str(horizon): list(direction_logits[horizon].shape) for horizon in model_config.horizons
        },
        "return_predictions": {
            str(horizon): list(return_predictions[horizon].shape)
            for horizon in model_config.horizons
        },
    }
    blind_manifest = create_blind_manifest(
        root / "blind_manifests",
        effective_run_id,
        (
            {
                "example_id": "pyg-shape-smoke-000001.SZ-2026-01-15",
                "as_of_date": graph.as_of_date.isoformat(),
                "target_stock_id": "000001.SZ",
            },
        ),
    )
    checkpoint_payload = dt_seat_hgt_checkpoint_payload(model_config)
    checkpoint = write_json_checkpoint(
        root / "checkpoints",
        effective_run_id,
        {"model": checkpoint_payload, "output_shapes": output_shapes},
        metadata={"run_type": "pyg_shape_smoke", "model_valid": False},
    )
    metrics: dict[str, object] = {
        "sample_count": 1,
        "engineering_smoke": True,
        "model_valid": False,
        "output_shapes": output_shapes,
    }
    hashes = {
        "data": hash_json(tensorized.metadata),
        "config": hash_file(config_path),
        "code": _code_hash(),
        "seed": hash_json({"seed": config.seed}),
        "split": hash_json(split.to_dict()),
        "blind_manifest": blind_manifest.sha256,
        "checkpoint": checkpoint.sha256,
        "model_architecture": hash_json(checkpoint_payload),
    }
    record = LineageRecord(
        run_id=effective_run_id,
        run_type="pyg_shape_smoke",
        status="completed",
        model_valid=False,
        provider=config.data.provider,
        seed=config.seed,
        split=split,
        hashes=hashes,
        metrics=metrics,
        tags=("pyg_shape_smoke", "sample_only", "not_model_valid"),
        created_at=now_iso(),
    )
    tracking = record_training_run(record, root)
    return TrainingRunResult(
        run_id=effective_run_id,
        status="completed",
        model_valid=False,
        lineage_path=tracking.lineage_path,
        checkpoint_path=checkpoint.checkpoint_path,
        hashes=hashes,
        metrics=metrics,
    )


def _sample_records(as_of_date: date) -> tuple[DragonTigerRecord, ...]:
    return (
        DragonTigerRecord(
            trade_date=as_of_date,
            seat_id="sample_institution_a",
            stock_id="000001.SZ",
            side=TradeSide.BUY,
            amount=12_000_000,
            industry="banking",
            concepts=("large_cap",),
            event_reason="sample_daily_deviation",
        ),
        DragonTigerRecord(
            trade_date=as_of_date - timedelta(days=1),
            seat_id="sample_arbitrage_b",
            stock_id="000001.SZ",
            side=TradeSide.SELL,
            amount=4_000_000,
            industry="banking",
            concepts=("large_cap",),
            event_reason="sample_daily_deviation",
        ),
    )


def _code_hash() -> str:
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("checkpointing.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("losses.py"),
        Path(__file__).parents[1] / "models" / "baselines.py",
        Path(__file__).parents[1] / "models" / "seat_hgt.py",
        Path(__file__).parents[1] / "tracking" / "lineage.py",
    )
    return hash_json({str(path): hash_file(path) for path in code_paths})
