from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

LABEL_KEYS = frozenset(
    {
        "label",
        "labels",
        "future_return",
        "target",
        "target_return",
        "y",
        "outcome",
    }
)
_EVALUATION_STATUSES = frozenset({"future_blind", "contaminated_oos"})


@dataclass(frozen=True)
class FrozenBlindManifest:
    path: Path
    sha256: str
    feature_file_count: int
    label_file_count: int
    split_status: str = "future_blind"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "feature_file_count": self.feature_file_count,
            "label_file_count": self.label_file_count,
            "split_status": self.split_status,
        }


def freeze_blind_manifest(
    output_path: str | Path,
    *,
    feature_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path],
    metadata: Mapping[str, object] | None = None,
) -> FrozenBlindManifest:
    return freeze_evaluation_manifest(
        output_path,
        feature_paths=feature_paths,
        label_paths=label_paths,
        split_status="future_blind",
        metadata=metadata,
    )


def freeze_evaluation_manifest(
    output_path: str | Path,
    *,
    feature_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path],
    split_status: str,
    metadata: Mapping[str, object] | None = None,
) -> FrozenBlindManifest:
    if split_status not in _EVALUATION_STATUSES:
        raise ValueError("evaluation manifest split_status is invalid")
    if not feature_paths:
        raise ValueError("evaluation manifest requires feature paths")
    if not label_paths:
        raise ValueError("evaluation manifest requires label paths to freeze")
    manifest_path = Path(output_path)
    base = manifest_path.parent
    features = tuple(_file_record(path, base) for path in feature_paths)
    labels = tuple(_file_record(path, base) for path in label_paths)
    feature_names = {str(record["path"]) for record in features}
    label_names = {str(record["path"]) for record in labels}
    overlap = sorted(feature_names.intersection(label_names))
    if overlap:
        raise ValueError(f"evaluation feature and label paths overlap: {', '.join(overlap)}")
    payload = {
        "schema_version": 2,
        "blind": split_status == "future_blind",
        "split_status": split_status,
        "label_feedback_forbidden": split_status == "future_blind",
        "label_keys_forbidden": sorted(LABEL_KEYS),
        "features": list(features),
        "labels": list(labels),
        "metadata": dict(metadata or {}),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(manifest_path, content)
    return FrozenBlindManifest(
        path=manifest_path,
        sha256=sha256(content).hexdigest(),
        feature_file_count=len(features),
        label_file_count=len(labels),
        split_status=split_status,
    )


def verify_blind_manifest(path: str | Path) -> dict[str, object]:
    verified = verify_evaluation_manifest(path)
    if verified["split_status"] != "future_blind":
        raise ValueError("blind manifest must use split_status=future_blind")
    return verified


def verify_evaluation_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    split_status = payload.get("split_status")
    if split_status not in _EVALUATION_STATUSES:
        raise ValueError("evaluation manifest has invalid split_status")
    expected_blind = split_status == "future_blind"
    if payload.get("blind") is not expected_blind:
        raise ValueError("evaluation manifest blind flag does not match split_status")
    if payload.get("label_feedback_forbidden") is not expected_blind:
        raise ValueError("evaluation manifest label feedback policy is invalid")
    features = _records(payload, "features")
    labels = _records(payload, "labels")
    if not features or not labels:
        raise ValueError("blind manifest requires feature and label records")
    base = manifest_path.parent
    for record in features + labels:
        file_path = (base / str(record["path"])).resolve()
        expected = str(record["sha256"])
        actual = _hash_file(file_path)
        if actual != expected:
            raise ValueError(f"blind manifest hash mismatch: {record['path']}")
    return {
        "path": str(manifest_path),
        "sha256": _hash_file(manifest_path),
        "verified": True,
        "split_status": split_status,
        "feature_file_count": len(features),
        "label_file_count": len(labels),
    }


def assert_no_label_feedback(payload: Mapping[str, Any]) -> None:
    leaked = sorted(_find_label_keys(payload))
    if leaked:
        raise ValueError(f"blind payload contains forbidden label keys: {', '.join(leaked)}")


def append_blind_prediction(
    path: str | Path,
    *,
    prediction_date: str,
    stock_id: str,
    predicted_rank: float,
    predicted_residual_return: float,
    model_hash: str,
    config_hash: str,
    data_cutoff: str,
    generated_at: str,
) -> dict[str, object]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_sha256(model_hash, "model_hash")
    _validate_sha256(config_hash, "config_hash")
    identity = (prediction_date, stock_id)
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            existing = json.loads(line)
            if (existing.get("prediction_date"), existing.get("stock_id")) == identity:
                raise ValueError("blind prediction already exists and cannot be overwritten")
    payload: dict[str, object] = {
        "prediction_date": prediction_date,
        "stock_id": stock_id,
        "predicted_rank": predicted_rank,
        "predicted_residual_return": predicted_residual_return,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "data_cutoff": data_cutoff,
        "generated_at": generated_at,
    }
    assert_no_label_feedback(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["file_sha256"] = sha256(canonical).hexdigest()
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


def _find_label_keys(value: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in LABEL_KEYS:
                found.add(path)
            found.update(_find_label_keys(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.update(_find_label_keys(child, f"{prefix}[{index}]"))
    return found


def _records(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"blind manifest {key} must be a list")
    records: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise ValueError(f"blind manifest {key} contains an invalid record")
        records.append(item)
    return tuple(records)


def _file_record(path: str | Path, base: Path) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"blind manifest file does not exist: {file_path}")
    resolved_base = base.resolve()
    resolved_file = file_path.resolve()
    try:
        manifest_path = str(resolved_file.relative_to(resolved_base))
    except ValueError:
        manifest_path = str(resolved_file)
    return {
        "path": manifest_path,
        "sha256": _hash_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("blind manifest must be a JSON object")
    return payload


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 hex digest")


def _atomic_write(path: Path, content: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
