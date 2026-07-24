"""Fail-closed loader for source-hashed executable-label V2 artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from seatflow.labels import (
    LABEL_V2_VERSION,
    StockLabelV2,
)


@dataclass(frozen=True)
class V2LabelArtifact:
    """Verified V2 label artifact; no promotion or model-validity claim is implied."""

    path: Path
    artifact_sha256: str
    labels: tuple[StockLabelV2, ...]

    def labels_by_key(self) -> dict[tuple[date, str], tuple[StockLabelV2, ...]]:
        grouped: dict[tuple[date, str], list[StockLabelV2]] = defaultdict(list)
        for label in self.labels:
            grouped[(label.as_of_date, label.target_stock_id)].append(label)
        return {
            key: tuple(sorted(values, key=lambda label: label.horizon))
            for key, values in grouped.items()
        }


def load_v2_label_artifact(path: str | Path) -> V2LabelArtifact:
    """Load only a hash-valid, explicitly fail-closed executable-label artifact."""

    artifact_path = Path(path)
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V2 label artifact root must be an object")
    expected_hash = raw.pop("sha256", None)
    if not isinstance(expected_hash, str) or not _is_sha256(expected_hash):
        raise ValueError("V2 label artifact requires a SHA-256 payload hash")
    actual_hash = sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("V2 label artifact payload hash does not match")
    if raw.get("label_schema_version") != LABEL_V2_VERSION:
        raise ValueError("artifact does not declare executable-label V2 schema")
    if raw.get("status") != "evaluated_fail_closed" or raw.get("model_valid") is not False:
        raise ValueError("V2 label artifact must remain explicitly fail-closed")
    raw_labels = raw.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("V2 label artifact requires a labels array")
    labels = tuple(_label_from_dict(item) for item in raw_labels)
    if not labels:
        raise ValueError("V2 label artifact contains no labels")
    return V2LabelArtifact(artifact_path, expected_hash, labels)


def _label_from_dict(payload: object) -> StockLabelV2:
    if not isinstance(payload, Mapping):
        raise ValueError("V2 label record must be an object")
    if payload.get("label_version") != LABEL_V2_VERSION:
        raise ValueError("V2 label record version does not match artifact schema")
    try:
        return StockLabelV2(
            as_of_date=date.fromisoformat(str(payload["as_of_date"])),
            cutoff_time=datetime.fromisoformat(str(payload["cutoff_time"])),
            target_stock_id=str(payload["target_stock_id"]),
            horizon=int(payload["horizon"]),
            label_start_date=date.fromisoformat(str(payload["label_start_date"])),
            label_end_date=date.fromisoformat(str(payload["label_end_date"])),
            benchmark_id=str(payload["benchmark_id"]),
            price_policy=str(payload["price_policy"]),
            adjustment_policy=str(payload["adjustment_policy"]),
            calendar_policy=str(payload["calendar_policy"]),
            label_available_time=datetime.fromisoformat(str(payload["label_available_time"])),
            disposition=str(payload["disposition"]),
            source_hash=str(payload["source_hash"]),
            stock_return=_decimal_or_none(payload.get("stock_return")),
            benchmark_return=_decimal_or_none(payload.get("benchmark_return")),
            excess_return=_decimal_or_none(payload.get("excess_return")),
            direction_label=_int_or_none(payload.get("direction_label")),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("invalid V2 label record") from error


def _decimal_or_none(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _int_or_none(value: object) -> int | None:
    return None if value is None else int(str(value))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
