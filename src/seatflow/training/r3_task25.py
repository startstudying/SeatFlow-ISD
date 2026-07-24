"""Controlled data preparation and CUDA preflight for R3 Task 25."""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
import subprocess
import tomllib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from seatflow.graph.tensorize import TensorizedGraph
from seatflow.models.seat_rank_lite import (
    SeatRankLiteConfig,
    build_seat_rank_lite,
    validate_parameter_count,
)
from seatflow.runtime.device import move_graph_to_device, release_cuda_objects, report_cuda_memory
from seatflow.training.amp import build_amp_controller
from seatflow.training.r2_runner import (
    R2GraphBatch,
    R2TrainingConfig,
    train_seat_rank_lite_seed,
)
from seatflow.training.ranking_losses import compute_ranking_loss

DATASET_PATH = Path("artifacts/diagnostics/R0_V2_TENSOR_DATASET_856D/v2-tensor-dataset.pkl")
PREFLIGHT_DATES = {
    "p50": "2023-12-14",
    "p90": "2025-12-30",
    "maximum_legal": "2024-02-07",
}
HARD_EDGE_LIMIT = 100_000
MEMORY_LIMIT_BYTES = int(7.2 * 1024**3)


@dataclass(frozen=True)
class GraphCapacityProfile:
    label: str
    prediction_date: str
    candidate_count: int
    stock_node_count: int
    node_count: int
    edge_count: int
    parameter_count: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    duration_seconds: float


@dataclass(frozen=True)
class R3InnerFold:
    fold_id: int
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    test_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_dataset(path: str | Path = DATASET_PATH) -> tuple[Any, ...]:
    value = pickle.loads(Path(path).read_bytes())
    if not isinstance(value, tuple):
        raise ValueError("R3 Task25 dataset must be a tuple")
    return value


def group_samples_by_date(samples: Sequence[Any]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for sample in samples:
        if sample.as_of_date is None:
            raise ValueError("R3 sample is missing as_of_date")
        grouped.setdefault(sample.as_of_date.isoformat(), []).append(sample)
    return {day: tuple(rows) for day, rows in sorted(grouped.items())}


def legalize_r3_graph(graph: TensorizedGraph) -> TensorizedGraph:
    """Apply the frozen R2 relation policy to the legacy point-in-time tensor."""

    data = graph.data.clone()
    cooccurrence = ("seat", "cooccurs_with", "seat")
    if cooccurrence in data.edge_types:
        del data[cooccurrence]
    original_edge_types = tuple(data.edge_types)
    for source, relation, target in original_edge_types:
        reverse = (target, f"rev_{relation}", source)
        if reverse in data.edge_types:
            continue
        store = data[(source, relation, target)]
        data[reverse].edge_index = store.edge_index.flip(0).clone()
        data[reverse].edge_attr = store.edge_attr.clone()
    edge_count = sum(int(data[edge].edge_index.size(1)) for edge in data.edge_types)
    if edge_count > HARD_EDGE_LIMIT:
        raise ValueError(f"legalized graph exceeds hard edge limit: {edge_count}")
    metadata = dict(graph.metadata)
    metadata["r3_relation_policy"] = {
        "seat_cooccurrence": False,
        "reverse_relations": True,
        "legacy_missing_optional_relations": ["stock_correlation", "industry_stock"],
    }
    return TensorizedGraph(data=data, metadata=metadata)


def run_l2_cuda_memory_preflight(
    samples: Sequence[Any],
    *,
    model_config: SeatRankLiteConfig | None = None,
) -> tuple[GraphCapacityProfile, ...]:
    """Run one full CUDA optimizer step for each frozen graph capacity point."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("R3 Task25 production preflight requires CUDA")
    model_config = SeatRankLiteConfig() if model_config is None else model_config
    by_date = group_samples_by_date(samples)
    profiles: list[GraphCapacityProfile] = []
    with TemporaryDirectory():
        for label, prediction_date in PREFLIGHT_DATES.items():
            day_samples = by_date[prediction_date]
            graph = legalize_r3_graph(day_samples[0].graph)
            device = torch.device("cuda")
            torch.cuda.reset_peak_memory_stats(device)
            model = build_seat_rank_lite(model_config, graph).to(device)
            parameter_count = int(validate_parameter_count(model)["parameter_count"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0003,
                weight_decay=0.0001,
                foreach=False,
            )
            amp = build_amp_controller(device, mixed_precision=True, preferred_dtype="fp16")
            data = move_graph_to_device(graph.data.clone(), device)
            target_indices = torch.tensor(
                [int(sample.target_node_index) for sample in day_samples],
                dtype=torch.long,
                device=device,
            )
            count = len(day_samples)
            standardized = torch.linspace(-1.0, 1.0, count, device=device)
            percentiles = torch.linspace(0.0, 1.0, count, device=device)
            directions = (standardized > 0).float()
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                raw_outputs = model(data)
                outputs = {
                    name: raw_outputs[name].index_select(0, target_indices)
                    for name in (
                        "rank_score",
                        "residual_return",
                        "direction_logit",
                        "uncertainty",
                    )
                }
                loss, _ = compute_ranking_loss(
                    outputs,
                    standardized,
                    directions,
                    (prediction_date,) * count,
                    objective="L2",
                    ranking_targets=percentiles,
                    max_pairs_per_date=1024,
                )
            amp.backward(loss)
            amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(optimizer)
            torch.cuda.synchronize(device)
            duration = perf_counter() - started
            memory = report_cuda_memory(device)
            if memory.peak_reserved_bytes > MEMORY_LIMIT_BYTES:
                raise MemoryError(f"{label} exceeds 7.2 GiB: {memory.peak_reserved_bytes} bytes")
            profiles.append(
                GraphCapacityProfile(
                    label=label,
                    prediction_date=prediction_date,
                    candidate_count=count,
                    stock_node_count=int(data["stock"].num_nodes),
                    node_count=sum(int(data[node].num_nodes) for node in data.node_types),
                    edge_count=sum(int(data[edge].edge_index.size(1)) for edge in data.edge_types),
                    parameter_count=parameter_count,
                    peak_allocated_bytes=memory.peak_allocated_bytes,
                    peak_reserved_bytes=memory.peak_reserved_bytes,
                    duration_seconds=duration,
                )
            )
            del data, model, optimizer, amp, raw_outputs, outputs, loss
            release_cuda_objects(device=device)
            gc.collect()
    return tuple(profiles)


def write_preflight_report(
    profiles: Sequence[GraphCapacityProfile],
    output: str | Path,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "r3_task25_cuda_preflight.v1",
        "outcome_metrics_accessed": False,
        "quantile_scope": "legalized_graph_edge_count",
        "quantile_rule": "nearest index round((N-1)*q)",
        "relation_policy": {
            "seat_cooccurrence": False,
            "reverse_relations": True,
            "legacy_missing_optional_relations": ["stock_correlation", "industry_stock"],
        },
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "passed": all(row.peak_reserved_bytes <= MEMORY_LIMIT_BYTES for row in profiles),
        "profiles": [asdict(row) for row in profiles],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_inner_folds(
    samples: Sequence[Any],
    *,
    minimum_train_dates: int = 320,
    purge_days: int = 6,
    embargo_days: int = 6,
) -> tuple[R3InnerFold, R3InnerFold]:
    """Freeze two expanding folds without inspecting return values."""

    by_date = group_samples_by_date(samples)
    dates = tuple(by_date)
    if len(dates) < minimum_train_dates + 2:
        raise ValueError("insufficient dates for R3 inner folds")
    remaining = dates[minimum_train_dates:]
    split = (len(remaining) + 1) // 2
    test_chunks = (remaining[:split], remaining[split:])
    folds: list[R3InnerFold] = []
    for fold_id, test_dates in enumerate(test_chunks):
        if not test_dates:
            raise ValueError("R3 inner test fold must not be empty")
        test_start = date.fromisoformat(test_dates[0])
        as_of_cutoff = test_start - timedelta(days=embargo_days)
        label_cutoff = test_start - timedelta(days=purge_days)
        train_rows = [
            sample
            for day in dates
            for sample in by_date[day]
            if sample.as_of_date < as_of_cutoff and sample.label_end_dates[1] < label_cutoff
        ]
        test_rows = [sample for day in test_dates for sample in by_date[day]]
        train_dates = tuple(sorted({sample.as_of_date.isoformat() for sample in train_rows}))
        folds.append(
            R3InnerFold(
                fold_id=fold_id,
                train_dates=train_dates,
                test_dates=tuple(test_dates),
                train_sample_ids=tuple(sample.sample_id for sample in train_rows),
                test_sample_ids=tuple(sample.sample_id for sample in test_rows),
            )
        )
    return folds[0], folds[1]


def create_immutable_run_manifest(
    samples: Sequence[Any],
    *,
    config_path: str | Path,
    output_root: str | Path,
) -> Path:
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_path = output_root / "folds.json"
    manifest_path = output_root / "run_manifest.json"
    if fold_path.exists() or manifest_path.exists():
        raise FileExistsError("R3 Task25 immutable manifests already exist")
    folds = build_inner_folds(samples)
    _write_json_exclusive(
        fold_path,
        {
            "schema_version": "r3_task25_folds.v1",
            "outcome_metrics_accessed": False,
            "folds": [fold.to_dict() for fold in folds],
        },
    )
    config_hash = _sha256(config_path)
    fold_hash = _sha256(fold_path)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_order: list[dict[str, object]] = []
    for objective in ("L0", "L1", "L2"):
        for fold in folds:
            for seed in (2026, 2027, 2028):
                run_id = f"{objective}-f{fold.fold_id}-s{seed}"
                relative = Path("runs") / objective / f"fold_{fold.fold_id}" / f"seed_{seed}"
                run_order.append(
                    {
                        "run_id": run_id,
                        "objective": objective,
                        "fold": fold.fold_id,
                        "seed": seed,
                        "config_sha256": config_hash,
                        "fold_manifest_sha256": fold_hash,
                        "output_directory": relative.as_posix(),
                        "planned_outputs": [
                            "model_state.pt",
                            "predictions.json",
                            "metadata.json",
                            "daily_predictions.jsonl",
                            "top5_holdings.jsonl",
                            "training_history.json",
                            "resources.json",
                            "run_complete.json",
                        ],
                    }
                )
    payload = {
        "schema_version": "r3_task25_run_manifest.v1",
        "immutable": True,
        "evaluation_locked_until_complete": True,
        "code_commit": code_commit,
        "config": {"path": config_path.as_posix(), "sha256": config_hash},
        "dataset": {
            "path": DATASET_PATH.as_posix(),
            "sha256": "c6fe493b6169e63cb0a8b8ef29d073b4142c97a7d97f1d120cc5282a66544fc9",
        },
        "fold_manifest": {"path": fold_path.as_posix(), "sha256": fold_hash},
        "expected_run_count": 18,
        "run_order": run_order,
    }
    _write_json_exclusive(manifest_path, payload)
    return manifest_path


def execute_all_planned_runs(
    samples: Sequence[Any],
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    output_root: str | Path,
    shard_count: int = 1,
    shard_index: int = 0,
) -> None:
    """Execute the immutable 18-run order without computing outcome metrics."""

    if shard_count < 1:
        raise ValueError("Task25 shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("Task25 shard_index must be within shard_count")
    config = _load_toml(config_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_root = Path(output_root)
    if _sha256(config_path) != manifest["config"]["sha256"]:
        raise ValueError("frozen Task25 config hash mismatch")
    if len(manifest["run_order"]) != 18:
        raise ValueError("Task25 run manifest must contain exactly 18 runs")
    folds_payload = json.loads(Path(manifest["fold_manifest"]["path"]).read_text(encoding="utf-8"))
    folds = {int(row["fold_id"]): row for row in folds_payload["folds"]}
    by_date = group_samples_by_date(samples)
    batches = {day: _ranking_batch(rows) for day, rows in by_date.items()}
    model_config = SeatRankLiteConfig(
        **{key: value for key, value in config["model"].items() if key != "maximum_parameters"}
    )
    training = config["training"]
    selected_runs = _select_manifest_runs(
        manifest["run_order"],
        shard_count=shard_count,
        shard_index=shard_index,
    )
    for planned in selected_runs:
        run_directory = output_root / planned["output_directory"]
        complete_path = run_directory / "run_complete.json"
        if complete_path.exists():
            _verify_run_complete(complete_path, manifest_path)
            continue
        objective = str(planned["objective"])
        fold_id = int(planned["fold"])
        seed = int(planned["seed"])
        fold = folds[fold_id]
        train_batches = tuple(batches[day] for day in fold["train_dates"])
        test_batches = tuple(batches[day] for day in fold["test_dates"])
        started = perf_counter()
        result = train_seat_rank_lite_seed(
            train_batches,
            model_config,
            R2TrainingConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                mixed_precision=bool(config["device"]["mixed_precision"]),
                preferred_dtype=str(config["device"]["preferred_dtype"]),
                device=str(config["device"]["device"]),
                hard_memory_bytes=MEMORY_LIMIT_BYTES,
                max_pairs_per_date=int(training["max_pairs_per_date"]),
                objective=objective,
            ),
            output_root / "runs" / objective,
            fold_id=fold_id,
            seed=seed,
            validation_batches=(),
            prediction_batches=test_batches,
        )
        duration = perf_counter() - started
        if result.epochs_completed != int(training["epochs"]):
            raise ValueError("Task25 run did not complete every frozen epoch")
        _finalize_non_evaluated_run(
            run_directory,
            result,
            duration_seconds=duration,
            run_id=str(planned["run_id"]),
            objective=objective,
            manifest_path=Path(manifest_path),
        )


def _select_manifest_runs(
    run_order: Sequence[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> Sequence[dict[str, Any]]:
    """Select a deterministic execution shard without changing manifest order."""

    return run_order[shard_index::shard_count]


def _ranking_batch(day_samples: Sequence[Any]) -> R2GraphBatch:
    import torch

    if not day_samples:
        raise ValueError("R3 date batch must not be empty")
    residuals = [float(sample.returns[1]) for sample in day_samples]
    mean = sum(residuals) / len(residuals)
    centered = [value - mean for value in residuals]
    lower = _quantile(centered, 0.01)
    upper = _quantile(centered, 0.99)
    winsorized = [min(max(value, lower), upper) for value in centered]
    winsor_mean = sum(winsorized) / len(winsorized)
    variance = sum((value - winsor_mean) ** 2 for value in winsorized) / len(winsorized)
    deviation = variance**0.5
    standardized = [
        0.0 if deviation == 0 else (value - winsor_mean) / deviation for value in winsorized
    ]
    ranks = _average_ranks(winsorized)
    denominator = max(len(ranks) - 1, 1)
    percentiles = [0.5 if len(ranks) == 1 else rank / denominator for rank in ranks]
    stock_ids = tuple(str(sample.target_stock_id) for sample in day_samples)
    if len(stock_ids) != len(set(stock_ids)):
        raise ValueError("R3 date batch contains duplicate target stocks")
    return R2GraphBatch(
        graph=legalize_r3_graph(day_samples[0].graph),
        residual_targets=torch.tensor(centered, dtype=torch.float32),
        direction_targets=torch.tensor(
            [float(value > 0) for value in centered], dtype=torch.float32
        ),
        prediction_date=day_samples[0].as_of_date.isoformat(),
        standardized_residual_targets=torch.tensor(standardized, dtype=torch.float32),
        ranking_targets=torch.tensor(percentiles, dtype=torch.float32),
        target_node_indices=torch.tensor(
            [int(sample.target_node_index) for sample in day_samples],
            dtype=torch.long,
        ),
        target_stock_ids=stock_ids,
    )


def _finalize_non_evaluated_run(
    run_directory: Path,
    result: Any,
    *,
    duration_seconds: float,
    run_id: str,
    objective: str,
    manifest_path: Path,
) -> None:
    predictions_path = run_directory / "predictions.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    holdings: list[dict[str, object]] = []
    for prediction_date, snapshot in sorted(predictions.items()):
        stock_ids = snapshot["stock_ids"]
        scores = snapshot["rank_score"]
        residuals = snapshot["residual_return"]
        logits = snapshot["direction_logit"]
        uncertainties = snapshot["uncertainty"]
        day_rows = [
            {
                "prediction_date": prediction_date,
                "stock_id": stock_id,
                "score": float(score),
                "predicted_residual": float(predicted_residual),
                "direction_logit": float(logit),
                "uncertainty": float(uncertainty),
            }
            for stock_id, score, predicted_residual, logit, uncertainty in zip(
                stock_ids, scores, residuals, logits, uncertainties, strict=True
            )
        ]
        rows.extend(day_rows)
        selected = sorted(
            day_rows,
            key=lambda row: (-float(str(row["score"])), str(row["stock_id"])),
        )[:5]
        weight = 1.0 / len(selected)
        holdings.extend(
            {
                "prediction_date": prediction_date,
                "rank": rank,
                "stock_id": row["stock_id"],
                "weight": weight,
            }
            for rank, row in enumerate(selected, start=1)
        )
    _write_jsonl_exclusive(run_directory / "daily_predictions.jsonl", rows)
    _write_jsonl_exclusive(run_directory / "top5_holdings.jsonl", holdings)
    _write_json_exclusive(
        run_directory / "training_history.json",
        {"run_id": run_id, "objective": objective, "steps": list(result.step_metrics)},
    )
    _write_json_exclusive(
        run_directory / "resources.json",
        {
            "run_id": run_id,
            "device": result.device,
            "parameter_count": result.parameter_count,
            "epochs_completed": result.epochs_completed,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
            "allocated_after_release_bytes": result.allocated_after_release_bytes,
            "duration_seconds": duration_seconds,
        },
    )
    artifact_names = [
        "model_state.pt",
        "predictions.json",
        "metadata.json",
        "daily_predictions.jsonl",
        "top5_holdings.jsonl",
        "training_history.json",
        "resources.json",
    ]
    _write_json_exclusive(
        run_directory / "run_complete.json",
        {
            "schema_version": "r3_task25_run_complete.v1",
            "run_id": run_id,
            "planned_manifest_sha256": _sha256(manifest_path),
            "evaluation_metrics_computed": False,
            "artifacts": [
                {
                    "path": name,
                    "bytes": (run_directory / name).stat().st_size,
                    "sha256": _sha256(run_directory / name),
                }
                for name in artifact_names
            ],
        },
    )


def _verify_run_complete(path: Path, manifest_path: str | Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["planned_manifest_sha256"] != _sha256(manifest_path):
        raise ValueError(f"run completion manifest mismatch: {path}")
    for item in payload["artifacts"]:
        artifact = path.parent / item["path"]
        if not artifact.exists() or _sha256(artifact) != item["sha256"]:
            raise ValueError(f"run artifact hash mismatch: {artifact}")


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def _write_jsonl_exclusive(
    path: Path,
    rows: Sequence[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _average_ranks(values: Sequence[float]) -> list[float]:
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    ranks = [0.0] * len(values)
    cursor = 0
    for value in sorted(grouped):
        indexes = grouped[value]
        average = (cursor + cursor + len(indexes) - 1) / 2.0
        for index in indexes:
            ranks[index] = average
        cursor += len(indexes)
    return ranks
