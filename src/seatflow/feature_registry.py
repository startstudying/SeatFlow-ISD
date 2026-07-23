from __future__ import annotations

from enum import StrEnum


class FeatureKind(StrEnum):
    CONTINUOUS = "continuous"
    BOUNDED_RATIO = "bounded_ratio"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ENTITY_ID = "entity_id"
    TIMESTAMP = "timestamp"
    MASK = "mask"


def infer_feature_kind(name: str) -> FeatureKind:
    """Classify a feature name conservatively for tensor-schema validation."""
    normalized = name.lower()
    if normalized.endswith(("_id", "_hash")) or normalized in {"id", "hash"}:
        return FeatureKind.ENTITY_ID
    if normalized.endswith("_mask"):
        return FeatureKind.MASK
    if normalized.endswith(("_date", "_time", "_timestamp")):
        return FeatureKind.TIMESTAMP
    if normalized.startswith(("is_", "has_")) or normalized.endswith("_available"):
        return FeatureKind.BINARY
    if normalized.endswith(("_ratio", "_rate", "_share", "_decay")):
        return FeatureKind.BOUNDED_RATIO
    return FeatureKind.CONTINUOUS


def is_numeric_tensor_feature(name: str) -> bool:
    """Only non-identity feature kinds may enter continuous tensor matrices."""
    return infer_feature_kind(name) not in {FeatureKind.ENTITY_ID, FeatureKind.CATEGORICAL}
