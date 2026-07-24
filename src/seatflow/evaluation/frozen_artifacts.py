"""Read-only integrity boundary for the frozen R3 Task 25 study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


class FrozenArtifactError(ValueError):
    """Raised when a Task 25 artifact is missing, changed, or outside the freeze."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frozen_manifest(
    repository_root: str | Path,
    *,
    source_commit: str = "1ac4097",
) -> dict[str, object]:
    """Validate Task 25's completion chain and return a complete freeze manifest."""
    repository = Path(repository_root).resolve()
    task_root = repository / "artifacts" / "r3_task25"
    plan_path = task_root / "run_manifest.json"
    completion_path = task_root / "completion_manifest.json"
    plan = _read_mapping(plan_path)
    completion = _read_mapping(completion_path)
    runs = _mapping_sequence(plan.get("run_order"), "run_order")
    completed = _mapping_sequence(completion.get("runs"), "completion runs")
    if len(runs) != 18 or len(completed) != 18:
        raise FrozenArtifactError("Task 25 freeze requires exactly 18 completed runs")
    if completion.get("all_runs_complete") is not True:
        raise FrozenArtifactError("Task 25 completion manifest is not complete")
    if completion.get("planned_manifest_sha256") != sha256_file(plan_path):
        raise FrozenArtifactError("completion manifest does not bind the run manifest")

    completed_by_id = {str(row["run_id"]): row for row in completed}
    frozen_runs: list[dict[str, object]] = []
    file_count = 0
    checkpoint_count = 0
    for run in runs:
        run_id = str(run["run_id"])
        output_directory = str(run["output_directory"])
        run_root = _contained(task_root, task_root / output_directory)
        complete_path = run_root / "run_complete.json"
        complete = _read_mapping(complete_path)
        completion_row = completed_by_id.get(run_id)
        if completion_row is None:
            raise FrozenArtifactError(f"run missing from completion manifest: {run_id}")
        if complete.get("run_id") != run_id:
            raise FrozenArtifactError(f"run identity mismatch: {run_id}")
        if complete.get("planned_manifest_sha256") != sha256_file(plan_path):
            raise FrozenArtifactError(f"run manifest binding mismatch: {run_id}")
        if completion_row.get("run_complete_sha256") != sha256_file(complete_path):
            raise FrozenArtifactError(f"run completion hash mismatch: {run_id}")

        artifacts: list[dict[str, object]] = []
        for item in _mapping_sequence(complete.get("artifacts"), f"{run_id} artifacts"):
            artifact_path = _contained(run_root, run_root / str(item["path"]))
            actual_size = artifact_path.stat().st_size
            actual_hash = sha256_file(artifact_path)
            if item.get("bytes") != actual_size or item.get("sha256") != actual_hash:
                raise FrozenArtifactError(f"run artifact integrity mismatch: {artifact_path}")
            relative = artifact_path.relative_to(repository).as_posix()
            artifacts.append(
                {"path": relative, "bytes": actual_size, "sha256": actual_hash}
            )
            file_count += 1
            checkpoint_count += artifact_path.name == "model_state.pt"
        artifacts.append(
            {
                "path": complete_path.relative_to(repository).as_posix(),
                "bytes": complete_path.stat().st_size,
                "sha256": sha256_file(complete_path),
            }
        )
        file_count += 1
        frozen_runs.append(
            {
                "run_id": run_id,
                "objective": run["objective"],
                "fold": run["fold"],
                "seed": run["seed"],
                "artifacts": artifacts,
            }
        )
    if file_count != 144 or checkpoint_count != 18:
        raise FrozenArtifactError(
            f"expected 144 run files and 18 checkpoints, got {file_count} and "
            f"{checkpoint_count}"
        )

    fixed_paths = (
        "configs/experiments/r3_task25_frozen.toml",
        "artifacts/r3_task25/folds.json",
        "artifacts/r3_task25/run_manifest.json",
        "artifacts/r3_task25/completion_manifest.json",
        "artifacts/r3_task25/evaluation.json",
    )
    inputs = [
        {
            "path": relative,
            "bytes": (repository / relative).stat().st_size,
            "sha256": sha256_file(repository / relative),
        }
        for relative in fixed_paths
    ]
    return {
        "schema_version": "r3_5_task25_frozen_manifest.v1",
        "source_commit": source_commit,
        "task25_complete": True,
        "task25_selection": "none",
        "task25_research_only": True,
        "task26_allowed": False,
        "run_count": 18,
        "run_file_count": file_count,
        "checkpoint_count": checkpoint_count,
        "planned_manifest_sha256": sha256_file(plan_path),
        "inputs": inputs,
        "runs": frozen_runs,
    }


def verify_frozen_manifest(
    repository_root: str | Path,
    manifest: Mapping[str, object],
) -> None:
    """Re-hash every path named by a previously generated freeze manifest."""
    repository = Path(repository_root).resolve()
    records = list(_manifest_file_records(manifest))
    for record in records:
        path = _contained(repository, repository / str(record["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(str(record["bytes"]))
            or sha256_file(path) != record["sha256"]
        ):
            raise FrozenArtifactError(f"frozen artifact changed: {path}")


class FrozenTask25Store:
    """Hash-checked reader that cannot escape the frozen Task 25 directory."""

    def __init__(self, repository_root: str | Path, manifest: Mapping[str, object]) -> None:
        self._repository = Path(repository_root).resolve()
        self._task_root = (self._repository / "artifacts" / "r3_task25").resolve()
        self._records = {
            str(row["path"]): row for row in _manifest_file_records(manifest)
        }

    def read_json(self, relative_path: str | Path) -> object:
        path, record = self._verified_path(relative_path)
        if path.suffix not in {".json", ".jsonl"}:
            raise FrozenArtifactError("frozen JSON reader accepts only JSON artifacts")
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return json.loads(path.read_text(encoding="utf-8"))

    def _verified_path(
        self, relative_path: str | Path
    ) -> tuple[Path, Mapping[str, object]]:
        path = _contained(self._task_root, self._task_root / relative_path)
        repository_relative = path.relative_to(self._repository).as_posix()
        record = self._records.get(repository_relative)
        if record is None:
            raise FrozenArtifactError(f"path is not part of the Task 25 freeze: {path}")
        if (
            not path.is_file()
            or path.stat().st_size != int(str(record["bytes"]))
            or sha256_file(path) != record["sha256"]
        ):
            raise FrozenArtifactError(f"frozen artifact changed: {path}")
        return path, record


def _manifest_file_records(
    manifest: Mapping[str, object],
) -> Iterator[Mapping[str, object]]:
    yield from _mapping_sequence(manifest.get("inputs"), "manifest inputs")
    for run in _mapping_sequence(manifest.get("runs"), "manifest runs"):
        yield from _mapping_sequence(run.get("artifacts"), "run artifacts")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FrozenArtifactError(f"missing frozen artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FrozenArtifactError(f"expected JSON object: {path}")
    return value


def _mapping_sequence(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise FrozenArtifactError(f"{name} must be a list of objects")
    return list(value)


def _contained(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise FrozenArtifactError(f"path escapes frozen root: {candidate}")
    return resolved_candidate
