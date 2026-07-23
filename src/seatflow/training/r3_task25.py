"""Controlled data preparation and CUDA preflight for R3 Task 25."""

from __future__ import annotations

import gc
import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
from seatflow.training.ranking_losses import compute_ranking_loss

DATASET_PATH = Path(
    "artifacts/diagnostics/R0_V2_TENSOR_DATASET_856D/v2-tensor-dataset.pkl"
)
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
                raise MemoryError(
                    f"{label} exceeds 7.2 GiB: {memory.peak_reserved_bytes} bytes"
                )
            profiles.append(
                GraphCapacityProfile(
                    label=label,
                    prediction_date=prediction_date,
                    candidate_count=count,
                    stock_node_count=int(data["stock"].num_nodes),
                    node_count=sum(int(data[node].num_nodes) for node in data.node_types),
                    edge_count=sum(
                        int(data[edge].edge_index.size(1)) for edge in data.edge_types
                    ),
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

