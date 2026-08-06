"""Immutable manifest and prediction DTOs for Base OOF artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self

from seatflow.artifacts.atomic import load_jsonl_artifact
from seatflow.models.base import BASE_FORMAL_CLASSIFICATION, BASE_PROVENANCE


_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA_URI_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PREDICTION_ROW_KEYS = frozenset(
    {
        "sample_id",
        "cycle_id",
        "stock_id",
        "fold_id",
        "cutoff_time",
        "prediction_time",
        "daily_path",
        "cumulative",
        "feature_schema_fingerprint",
        "provenance",
        "classification",
    }
)


@dataclass(frozen=True)
class OOFPredictionRow:
    """One OOF Base prediction row. It intentionally contains no y_true."""

    sample_id: str
    cycle_id: str
    stock_id: str
    fold_id: int
    cutoff_time: datetime
    prediction_time: datetime
    daily_path: tuple[float, float, float, float, float]
    cumulative: tuple[float, float, float]
    feature_schema_fingerprint: str
    provenance: str
    classification: str

    def __post_init__(self) -> None:
        for name in ("sample_id", "cycle_id", "stock_id", "provenance", "classification"):
            if not getattr(self, name):
                raise ValueError(f"OOFPredictionRow {name} must be non-empty")
        _aware(self.cutoff_time, "cutoff_time")
        _aware(self.prediction_time, "prediction_time")
        if self.prediction_time <= self.cutoff_time:
            raise ValueError("OOFPredictionRow prediction_time must be after cutoff_time")
        if self.provenance != BASE_PROVENANCE:
            raise ValueError("OOFPredictionRow provenance must be BASE_ONLY")
        if self.classification != BASE_FORMAL_CLASSIFICATION:
            raise ValueError("OOFPredictionRow classification must be FORMAL_BASE_DAILY_PATH")
        _sha_uri(self.feature_schema_fingerprint, "feature_schema_fingerprint")
        if len(self.daily_path) != 5 or not all(math.isfinite(v) for v in self.daily_path):
            raise ValueError("OOFPredictionRow daily_path must contain five finite values")
        derived = (self.daily_path[0], sum(self.daily_path[:3]), sum(self.daily_path))
        if len(self.cumulative) != 3 or any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(self.cumulative, derived)
        ):
            raise ValueError("OOFPredictionRow cumulative must be derived from daily_path")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "cycle_id": self.cycle_id,
            "stock_id": self.stock_id,
            "fold_id": self.fold_id,
            "cutoff_time": self.cutoff_time.isoformat(),
            "prediction_time": self.prediction_time.isoformat(),
            "daily_path": list(self.daily_path),
            "cumulative": list(self.cumulative),
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "provenance": self.provenance,
            "classification": self.classification,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        keys = frozenset(payload)
        if keys != _PREDICTION_ROW_KEYS:
            forbidden = sorted(keys - _PREDICTION_ROW_KEYS)
            if any(key in {"y_true", "target", "target_path"} for key in forbidden):
                raise ValueError("prediction schema forbids y_true/target fields")
            raise ValueError("prediction schema keys do not match OOFPredictionRow")
        return cls(
            sample_id=str(payload["sample_id"]),
            cycle_id=str(payload["cycle_id"]),
            stock_id=str(payload["stock_id"]),
            fold_id=int(payload["fold_id"]),
            cutoff_time=datetime.fromisoformat(str(payload["cutoff_time"])),
            prediction_time=datetime.fromisoformat(str(payload["prediction_time"])),
            daily_path=_float5(payload["daily_path"]),
            cumulative=_float3(payload["cumulative"]),
            feature_schema_fingerprint=str(payload["feature_schema_fingerprint"]),
            provenance=str(payload["provenance"]),
            classification=str(payload["classification"]),
        )


@dataclass(frozen=True)
class FoldEvidence:
    """Per-fold evidence bound into the top-level OOF manifest."""

    fold_id: int
    train_sample_id_hash: str
    prediction_sample_id_hash: str
    transform_fingerprint: str
    checkpoint_sha256: str
    prediction_content_hash: str
    target_hash: str
    cutoff_start: datetime
    cutoff_end: datetime
    prediction_count: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    purge_trading_days: int
    embargo_trading_days: int
    label_horizon_trading_days: int
    fold_payload_hash: str

    def __post_init__(self) -> None:
        _sha(self.train_sample_id_hash, "train_sample_id_hash")
        _sha(self.prediction_sample_id_hash, "prediction_sample_id_hash")
        _sha_uri(self.transform_fingerprint, "transform_fingerprint")
        _sha(self.checkpoint_sha256, "checkpoint_sha256")
        _sha(self.prediction_content_hash, "prediction_content_hash")
        _sha(self.target_hash, "target_hash")
        _aware(self.cutoff_start, "cutoff_start")
        _aware(self.cutoff_end, "cutoff_end")
        for name in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        ):
            _aware(getattr(self, name), name)
        if self.cutoff_end < self.cutoff_start:
            raise ValueError("fold cutoff_end precedes cutoff_start")
        if self.train_end < self.train_start:
            raise ValueError("fold train_end precedes train_start")
        if self.validation_end < self.validation_start:
            raise ValueError("fold validation_end precedes validation_start")
        if self.test_end < self.test_start:
            raise ValueError("fold test_end precedes test_start")
        for name in ("purge_trading_days", "embargo_trading_days", "label_horizon_trading_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.prediction_count <= 0:
            raise ValueError("FoldEvidence prediction_count must be positive")
        _sha(self.fold_payload_hash, "fold_payload_hash")

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_sample_id_hash": self.train_sample_id_hash,
            "prediction_sample_id_hash": self.prediction_sample_id_hash,
            "transform_fingerprint": self.transform_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prediction_content_hash": self.prediction_content_hash,
            "target_hash": self.target_hash,
            "cutoff_start": self.cutoff_start.isoformat(),
            "cutoff_end": self.cutoff_end.isoformat(),
            "prediction_count": self.prediction_count,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "purge_trading_days": self.purge_trading_days,
            "embargo_trading_days": self.embargo_trading_days,
            "label_horizon_trading_days": self.label_horizon_trading_days,
            "fold_payload_hash": self.fold_payload_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            fold_id=int(payload["fold_id"]),
            train_sample_id_hash=str(payload["train_sample_id_hash"]),
            prediction_sample_id_hash=str(payload["prediction_sample_id_hash"]),
            transform_fingerprint=str(payload["transform_fingerprint"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            prediction_content_hash=str(payload["prediction_content_hash"]),
            target_hash=str(payload["target_hash"]),
            cutoff_start=datetime.fromisoformat(str(payload["cutoff_start"])),
            cutoff_end=datetime.fromisoformat(str(payload["cutoff_end"])),
            prediction_count=int(payload["prediction_count"]),
            train_start=datetime.fromisoformat(str(payload["train_start"])),
            train_end=datetime.fromisoformat(str(payload["train_end"])),
            validation_start=datetime.fromisoformat(str(payload["validation_start"])),
            validation_end=datetime.fromisoformat(str(payload["validation_end"])),
            test_start=datetime.fromisoformat(str(payload["test_start"])),
            test_end=datetime.fromisoformat(str(payload["test_end"])),
            purge_trading_days=int(payload["purge_trading_days"]),
            embargo_trading_days=int(payload["embargo_trading_days"]),
            label_horizon_trading_days=int(payload["label_horizon_trading_days"]),
            fold_payload_hash=str(payload["fold_payload_hash"]),
        )


@dataclass(frozen=True)
class OOFManifest:
    """Top-level immutable OOF manifest."""

    FORMAL_BLOCKED_CLASSIFICATION: ClassVar[str] = "FORMAL_BLOCKED"
    SYNTHETIC_CLASSIFICATION: ClassVar[str] = "SYNTHETIC_NON_FORMAL"

    classification: str
    dataset_seal_hash: str
    audit_attestation_hash: str
    lineage_hash: str
    materialization_hash: str
    target_hash: str
    split_hash: str
    calendar_hash: str
    calendar_evidence_hash: str
    config_hash: str
    git_hash: str
    feature_schema_hash: str
    seed_hash: str
    folds: tuple[FoldEvidence, ...]
    coverage: Mapping[str, int]
    prediction_content_hash: str
    eligible_sample_id_hash: str
    predicted_sample_id_hash: str
    manifest_fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        if self.classification != self.SYNTHETIC_CLASSIFICATION:
            raise ValueError("OOFManifest classification must be SYNTHETIC_NON_FORMAL in Task 6C")
        for field_name in (
            "dataset_seal_hash",
            "audit_attestation_hash",
            "lineage_hash",
            "materialization_hash",
            "target_hash",
            "split_hash",
            "calendar_hash",
            "calendar_evidence_hash",
            "config_hash",
            "git_hash",
            "seed_hash",
            "prediction_content_hash",
            "eligible_sample_id_hash",
            "predicted_sample_id_hash",
        ):
            _sha(str(getattr(self, field_name)), field_name)
        _sha_uri(self.feature_schema_hash, "feature_schema_hash")
        if not self.folds:
            raise ValueError("OOFManifest requires fold evidence")
        object.__setattr__(self, "coverage", MappingProxyType(dict(self.coverage)))
        if self.manifest_fingerprint:
            _sha(self.manifest_fingerprint, "manifest_fingerprint")
            expected = _fingerprint(self.without_fingerprint())
            if expected != self.manifest_fingerprint:
                raise ValueError("manifest fingerprint mismatch")

    @classmethod
    def create(
        cls,
        *,
        classification: str,
        dataset_seal_hash: str,
        audit_attestation_hash: str,
        lineage_hash: str,
        materialization_hash: str,
        target_hash: str,
        split_hash: str,
        calendar_hash: str,
        calendar_evidence_hash: str,
        config_hash: str,
        git_hash: str,
        feature_schema_hash: str,
        seed_hash: str,
        folds: tuple[FoldEvidence, ...],
        predictions: tuple[OOFPredictionRow, ...] | None = None,
        coverage: Mapping[str, int],
        prediction_content_hash: str | None = None,
        eligible_sample_id_hash: str | None = None,
        predicted_sample_id_hash: str | None = None,
    ) -> Self:
        if predictions is None and prediction_content_hash is None:
            raise ValueError("predictions or prediction_content_hash is required")
        bound_prediction_hash = (
            prediction_rows_hash(predictions)
            if predictions is not None
            else str(prediction_content_hash)
        )
        if predictions is None:
            if eligible_sample_id_hash is None or predicted_sample_id_hash is None:
                raise ValueError("sample-id hashes are required without predictions")
            bound_eligible_sample_id_hash = eligible_sample_id_hash
            bound_predicted_sample_id_hash = predicted_sample_id_hash
        else:
            bound_predicted_sample_id_hash = sample_ids_hash(row.sample_id for row in predictions)
            bound_eligible_sample_id_hash = bound_predicted_sample_id_hash
        base = cls(
            classification=classification,
            dataset_seal_hash=dataset_seal_hash,
            audit_attestation_hash=audit_attestation_hash,
            lineage_hash=lineage_hash,
            materialization_hash=materialization_hash,
            target_hash=target_hash,
            split_hash=split_hash,
            calendar_hash=calendar_hash,
            calendar_evidence_hash=calendar_evidence_hash,
            config_hash=config_hash,
            git_hash=git_hash,
            feature_schema_hash=feature_schema_hash,
            seed_hash=seed_hash,
            folds=folds,
            coverage=coverage,
            prediction_content_hash=bound_prediction_hash,
            eligible_sample_id_hash=bound_eligible_sample_id_hash,
            predicted_sample_id_hash=bound_predicted_sample_id_hash,
        )
        return cls(**base.without_fingerprint(), manifest_fingerprint=_fingerprint(base.without_fingerprint()))

    def without_fingerprint(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "dataset_seal_hash": self.dataset_seal_hash,
            "audit_attestation_hash": self.audit_attestation_hash,
            "lineage_hash": self.lineage_hash,
            "materialization_hash": self.materialization_hash,
            "target_hash": self.target_hash,
            "split_hash": self.split_hash,
            "calendar_hash": self.calendar_hash,
            "calendar_evidence_hash": self.calendar_evidence_hash,
            "config_hash": self.config_hash,
            "git_hash": self.git_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "seed_hash": self.seed_hash,
            "folds": self.folds,
            "coverage": dict(self.coverage),
            "prediction_content_hash": self.prediction_content_hash,
            "eligible_sample_id_hash": self.eligible_sample_id_hash,
            "predicted_sample_id_hash": self.predicted_sample_id_hash,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.without_fingerprint()
        payload["folds"] = [fold.as_dict() for fold in self.folds]
        payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            classification=str(payload["classification"]),
            dataset_seal_hash=str(payload["dataset_seal_hash"]),
            audit_attestation_hash=str(payload["audit_attestation_hash"]),
            lineage_hash=str(payload["lineage_hash"]),
            materialization_hash=str(payload["materialization_hash"]),
            target_hash=str(payload["target_hash"]),
            split_hash=str(payload["split_hash"]),
            calendar_hash=str(payload["calendar_hash"]),
            calendar_evidence_hash=str(payload["calendar_evidence_hash"]),
            config_hash=str(payload["config_hash"]),
            git_hash=str(payload["git_hash"]),
            feature_schema_hash=str(payload["feature_schema_hash"]),
            seed_hash=str(payload["seed_hash"]),
            folds=tuple(FoldEvidence.from_dict(item) for item in payload["folds"]),
            coverage={str(k): int(v) for k, v in dict(payload["coverage"]).items()},
            prediction_content_hash=str(payload["prediction_content_hash"]),
            eligible_sample_id_hash=str(payload["eligible_sample_id_hash"]),
            predicted_sample_id_hash=str(payload["predicted_sample_id_hash"]),
            manifest_fingerprint=str(payload["manifest_fingerprint"]),
        )


def validate_oof_artifacts(*, manifest_path: Path, prediction_path: Path) -> OOFManifest:
    """Validate manifest and prediction content against all bound hashes."""
    if manifest_path.suffix == ".tmp" or ".tmp" in manifest_path.name:
        raise ValueError("partial manifest artifact is not loadable")
    raw_rows = load_jsonl_artifact(prediction_path)
    try:
        prediction_rows = tuple(OOFPredictionRow.from_dict(row) for row in raw_rows)
    except ValueError as exc:
        raise ValueError("prediction content hash mismatch") from exc
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = OOFManifest.from_dict(payload)
    if prediction_rows_hash(prediction_rows) != manifest.prediction_content_hash:
        raise ValueError("prediction content hash mismatch")
    expected_coverage = {
        "input_count",
        "predicted_count",
        "fold_count",
        "eligible_count",
    }
    if set(manifest.coverage) != expected_coverage:
        raise ValueError("coverage keys mismatch")
    if int(manifest.coverage["predicted_count"]) != len(prediction_rows):
        raise ValueError("coverage predicted_count mismatch")
    if int(manifest.coverage["eligible_count"]) != len(prediction_rows):
        raise ValueError("coverage eligible_count mismatch")
    if int(manifest.coverage["fold_count"]) != len(manifest.folds):
        raise ValueError("coverage fold_count mismatch")
    actual_sample_hash = sample_ids_hash(row.sample_id for row in prediction_rows)
    if manifest.eligible_sample_id_hash != manifest.predicted_sample_id_hash:
        raise ValueError("eligible/predicted sample-id hash mismatch")
    if manifest.predicted_sample_id_hash != actual_sample_hash:
        raise ValueError("top-level prediction sample-id hash mismatch")
    fold_ids = [fold.fold_id for fold in manifest.folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("duplicate fold evidence")
    row_fold_ids = {row.fold_id for row in prediction_rows}
    if row_fold_ids != set(fold_ids):
        raise ValueError("missing or unknown fold ids in predictions")
    expected_by_fold = {fold.fold_id: fold.prediction_content_hash for fold in manifest.folds}
    actual_by_fold = {
        fold_id: prediction_rows_hash(tuple(row for row in prediction_rows if row.fold_id == fold_id))
        for fold_id in row_fold_ids
    }
    if expected_by_fold != actual_by_fold:
        raise ValueError("fold prediction content hash mismatch")
    for fold in manifest.folds:
        fold_rows = tuple(row for row in prediction_rows if row.fold_id == fold.fold_id)
        count = len(fold_rows)
        if count != fold.prediction_count:
            raise ValueError("fold prediction count mismatch")
        if fold.prediction_sample_id_hash != sample_ids_hash(row.sample_id for row in fold_rows):
            raise ValueError("fold prediction sample-id hash mismatch")
    return manifest


def prediction_rows_hash(rows: tuple[OOFPredictionRow, ...]) -> str:
    return _fingerprint([row.as_dict() for row in rows])


def sample_ids_hash(sample_ids: Any) -> str:
    return _fingerprint(tuple(sorted(str(sample_id) for sample_id in sample_ids)))


def stable_hash(value: Any) -> str:
    return _fingerprint(value)


def _fingerprint(payload: Any) -> str:
    normalized = _normalize(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, FoldEvidence):
        return value.as_dict()
    if isinstance(value, OOFPredictionRow):
        return value.as_dict()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, frozenset | set):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    return value


def _sha(value: str, name: str) -> None:
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character sha256 hex fingerprint")


def _sha_uri(value: str, name: str) -> None:
    if not _SHA_URI_RE.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 hex>")


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _float5(value: Any) -> tuple[float, float, float, float, float]:
    items = tuple(float(item) for item in value)
    if len(items) != 5:
        raise ValueError("expected five floats")
    return (items[0], items[1], items[2], items[3], items[4])


def _float3(value: Any) -> tuple[float, float, float]:
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError("expected three floats")
    return (items[0], items[1], items[2])
