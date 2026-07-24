from __future__ import annotations

import json
import re
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType
from typing import Any

from seatflow.training.checkpointing import hash_json
from seatflow.training.preflight import TimeSplit


@dataclass(frozen=True)
class BlindManifest:
    path: Path
    sha256: str
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class LineageRecord:
    run_id: str
    run_type: str
    status: str
    model_valid: bool
    provider: str
    seed: int
    split: TimeSplit
    hashes: dict[str, str]
    metrics: dict[str, object]
    tags: tuple[str, ...]
    created_at: str
    data_scope: str = "unspecified"
    seat_detail_scope: str = "unspecified"
    label_status: str = "not_applicable"
    blind_status: str = "not_applicable"
    model_family: str = "unspecified"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "model_valid": self.model_valid,
            "provider": self.provider,
            "seed": self.seed,
            "split": self.split.to_dict(),
            "hashes": dict(self.hashes),
            "metrics": dict(self.metrics),
            "tags": list(self.tags),
            "created_at": self.created_at,
            "experiment_context": {
                "data_scope": self.data_scope,
                "seat_detail_scope": self.seat_detail_scope,
                "label_status": self.label_status,
                "blind_status": self.blind_status,
                "model_family": self.model_family,
            },
        }


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class TrackingWriteResult:
    lineage_path: Path
    backend: str
    mlflow_run_id: str | None
    fallback_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_path": str(self.lineage_path),
            "backend": self.backend,
            "mlflow_run_id": self.mlflow_run_id,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RunPackage:
    path: Path
    sha256: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256, "files": list(self.files)}


class LocalLineageStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "lineage.jsonl"

    def append(self, record: LineageRecord) -> Path:
        required_hashes = {
            "data",
            "config",
            "code",
            "seed",
            "split",
            "blind_manifest",
            "checkpoint",
        }
        missing = sorted(required_hashes.difference(record.hashes))
        if missing:
            raise ValueError(f"lineage record missing hashes: {', '.join(missing)}")
        if record.run_type.startswith("research"):
            _validate_research_context(record)
        if record.model_valid:
            _validate_model_valid_evidence(record)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = _redact_mapping(record.to_dict())
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
        return self.path


def load_tracking_config(path: str | Path) -> TrackingConfig:
    config_path = Path(path)
    if not config_path.exists():
        return TrackingConfig(
            enabled=False,
            tracking_uri="file:./mlruns",
            experiment_name="SeatFlow",
        )
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = raw.get("mlflow", {})
    if not isinstance(section, dict):
        raise ValueError("mlflow config must contain an [mlflow] table")
    return TrackingConfig(
        enabled=bool(section.get("enabled", False)),
        tracking_uri=str(section.get("tracking_uri", "file:./mlruns")),
        experiment_name=str(section.get("experiment_name", "SeatFlow")),
    )


def record_training_run(
    record: LineageRecord,
    root: str | Path,
    config_path: str | Path = "configs/mlflow.toml",
) -> TrackingWriteResult:
    lineage_path = LocalLineageStore(Path(root) / "tracking").append(record)
    config = load_tracking_config(config_path)
    if not config.enabled:
        return TrackingWriteResult(
            lineage_path=lineage_path,
            backend="local-jsonl",
            mlflow_run_id=None,
            fallback_reason="mlflow_disabled",
        )

    try:
        mlflow = import_module("mlflow")
    except ModuleNotFoundError:
        return TrackingWriteResult(
            lineage_path=lineage_path,
            backend="local-jsonl",
            mlflow_run_id=None,
            fallback_reason="mlflow_not_installed",
        )

    run_id = _mirror_to_mlflow(mlflow, config, record)
    return TrackingWriteResult(
        lineage_path=lineage_path,
        backend="mlflow+local-jsonl",
        mlflow_run_id=run_id,
        fallback_reason=None,
    )


def export_run_package(
    lineage_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    include_artifacts: Sequence[str | Path] = (),
) -> RunPackage:
    record = _find_lineage_record(Path(lineage_path), run_id)
    safe_record = _redact_mapping(record)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    package_dir = root / run_id
    package_dir.mkdir(parents=True, exist_ok=True)
    run_path = package_dir / "run.json"
    _atomic_write(
        run_path,
        json.dumps(safe_record, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    files = ["run.json"]
    for artifact in include_artifacts:
        source = Path(artifact)
        if not _is_export_safe(source):
            raise ValueError(f"refusing to export raw or secret-bearing artifact: {source}")
        if not source.is_file():
            raise ValueError(f"export artifact does not exist or is not a file: {source}")
        destination = package_dir / source.name
        shutil.copyfile(source, destination)
        files.append(destination.name)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "contains_raw_data": False,
        "contains_tokens": False,
        "files": sorted(files),
    }
    manifest_path = package_dir / "manifest.json"
    _atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    files.append("manifest.json")
    return RunPackage(
        path=package_dir,
        sha256=hash_json(_hash_export_files(package_dir)),
        files=tuple(sorted(files)),
    )


def create_blind_manifest(
    output_dir: str | Path,
    run_id: str,
    examples: tuple[dict[str, object], ...],
) -> BlindManifest:
    if not examples:
        raise ValueError("blind manifest requires at least one example")
    redacted_rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if "label" in example or "future_return" in example:
            raise ValueError("blind manifest examples must not contain labels")
        row: dict[str, object] = {
            "example_id": str(example.get("example_id") or f"sample-{index}")
        }
        for key in sorted(example):
            if key != "example_id":
                row[key] = example[key]
        redacted_rows.append(row)

    payload = {
        "run_id": run_id,
        "blind": True,
        "label_columns_redacted": ["label", "future_return"],
        "examples": redacted_rows,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.blind-manifest.json"
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    return BlindManifest(path=path, sha256=hash_json(payload), sample_count=len(redacted_rows))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _validate_model_valid_evidence(record: LineageRecord) -> None:
    tags = set(record.tags)
    if "data_scope:real" not in tags:
        raise ValueError("model_valid lineage requires data_scope:real tag")
    if "sample_only" in tags or record.provider == "sample":
        raise ValueError("model_valid lineage cannot use sample-only data")
    required_evidence_hashes = {"protocol", "evaluation", "checkpoint"}
    missing = sorted(required_evidence_hashes.difference(record.hashes))
    if missing:
        raise ValueError(f"model_valid lineage missing evidence hashes: {', '.join(missing)}")


def _validate_research_context(record: LineageRecord) -> None:
    if record.data_scope != "real":
        raise ValueError("research lineage requires data_scope=real")
    if record.seat_detail_scope not in {"institutional_proxy", "complete_seat"}:
        raise ValueError("research lineage requires a declared seat_detail_scope")
    if record.label_status not in {"observed", "unreleased", "mixed"}:
        raise ValueError("research lineage requires a declared label_status")
    if record.blind_status not in {"future_blind", "contaminated_oos", "not_applicable"}:
        raise ValueError("research lineage requires a declared blind_status")
    if not record.model_family.strip() or record.model_family == "unspecified":
        raise ValueError("research lineage requires a declared model_family")
    if "protocol" not in record.hashes:
        raise ValueError("research lineage requires a protocol hash")


def _mirror_to_mlflow(
    mlflow: ModuleType | Any,
    config: TrackingConfig,
    record: LineageRecord,
) -> str | None:
    redacted_record_obj = _redact_mapping(record.to_dict())
    if not isinstance(redacted_record_obj, dict):
        raise TypeError("redacted lineage record must be a dictionary")
    redacted_record: dict[str, object] = redacted_record_obj
    redacted_tags = redacted_record.get("tags", [])
    tag_values = redacted_tags if isinstance(redacted_tags, list) else []
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=record.run_id) as run:
        mlflow.log_params(
            {
                "run_id": record.run_id,
                "run_type": record.run_type,
                "provider": record.provider,
                "seed": record.seed,
                "model_valid": record.model_valid,
            }
        )
        numeric_metrics = {
            key: value
            for key, value in record.metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        mlflow.set_tags(
            {
                "seatflow.tags": ",".join(str(tag) for tag in tag_values),
                "seatflow.status": record.status,
                "seatflow.model_valid": str(record.model_valid).lower(),
                "seatflow.split_hash": record.hashes["split"],
            }
        )
        mlflow.log_dict(redacted_record, "lineage/run.json")
        info = getattr(run, "info", None)
        mlflow_run_id = getattr(info, "run_id", None)
        return str(mlflow_run_id) if mlflow_run_id else None


def _find_lineage_record(path: Path, run_id: str) -> dict[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("run_id") == run_id:
            return record
    raise ValueError(f"run_id not found in lineage: {run_id}")


def _redact_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _redact_mapping(item)
        return redacted
    if isinstance(value, list):
        return [_redact_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_mapping(item) for item in value)
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ("token", "secret", "password", "credential"))


def _redact_string(value: str) -> str:
    secret_key = (
        r"[A-Za-z0-9_.-]*"
        r"(?:token|secret|password|credential|api_key|access_key)"
        r"[A-Za-z0-9_.-]*"
    )
    redacted = re.sub(
        rf'(?i)("{secret_key}")\s*:\s*("[^"]*"|[^,\s}}]+)',
        lambda match: f'{match.group(1)}:"<redacted>"',
        value,
    )
    redacted = re.sub(
        rf"(?i)\b({secret_key})\s*([:=])\s*[^,\s;]+",
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        redacted,
    )
    return re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", redacted)


def _is_export_safe(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    if lowered_parts.intersection({"raw", "raw_data", "secrets", ".env"}):
        return False
    return not any(marker in lowered_name for marker in ("token", "secret", "password", ".env"))


def _hash_export_files(package_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(package_dir)): hash_json(
            {"content": path.read_text(encoding="utf-8", errors="replace")}
        )
        for path in sorted(package_dir.glob("*"))
        if path.is_file()
    }


def _atomic_write(path: Path, content: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
