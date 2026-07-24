"""Per-seed artifact and resource lifecycle for serial R2 training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seatflow.runtime.device import CudaMemoryReport, release_cuda_objects


def seed_artifact_directory(
    output_root: str | Path,
    *,
    fold_id: int,
    seed: int,
) -> Path:
    directory = Path(output_root) / f"fold_{fold_id}" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cpu_state_dict(model: Any) -> dict[str, Any]:
    """Clone a model state so no saved tensor retains CUDA storage."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_seed_artifacts(
    output_root: str | Path,
    *,
    fold_id: int,
    seed: int,
    model: Any,
    predictions: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    allow_overwrite: bool = False,
) -> Path:
    import torch

    directory = seed_artifact_directory(output_root, fold_id=fold_id, seed=seed)
    checkpoint_path = directory / "model_state.pt"
    targets = (
        checkpoint_path,
        directory / "predictions.json",
        directory / "metadata.json",
    )
    if not allow_overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"seed artifacts already exist for fold={fold_id}, seed={seed}"
        )
    torch.save(cpu_state_dict(model), checkpoint_path)
    _write_json(directory / "predictions.json", dict(predictions))
    _write_json(
        directory / "metadata.json",
        {"fold_id": fold_id, "seed": seed, **dict(metadata or {})},
    )
    return checkpoint_path


def release_seed_resources(
    *,
    model: Any,
    optimizer: Any,
    graph: Any | None = None,
    device: Any | None = None,
) -> CudaMemoryReport:
    """Drop one seed's heavy objects before the next seed is constructed."""

    return release_cuda_objects(model, optimizer, graph, device=device)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
