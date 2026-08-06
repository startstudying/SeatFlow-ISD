"""Bronze record models for raw data ingestion.

Every Bronze record carries:
- ``event_time``: when the business fact occurred
- ``available_time``: when the system could first access it
- ``ingested_at``: when the record entered local storage
- ``source``: provider identifier
- ``source_record_id``: provider-native record key
- ``source_version``: interface or file-batch version
- ``revision_no``: data revision number
- ``raw_payload_hash``: SHA-256 of the canonical raw payload
- ``ingestion_run_id``: run that ingested this record
- ``source_confidence``: provider confidence level
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from seatflow.domain.time import assert_aware


class SourceConfidence(str, Enum):
    """Confidence level assigned to a data source."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DEGRADED = "DEGRADED"


class BronzeRecord(BaseModel):
    """Immutable raw record as ingested from a provider."""

    event_time: datetime
    available_time: datetime
    ingested_at: datetime
    source: str
    source_record_id: str
    source_version: str
    revision_no: int = 0
    raw_payload: dict[str, Any]
    raw_payload_hash: str
    ingestion_run_id: str
    source_confidence: SourceConfidence = SourceConfidence.MEDIUM

    model_config = {"frozen": True}

    @field_validator("event_time", "available_time", "ingested_at")
    @classmethod
    def _must_be_aware(cls, v: datetime, info: ValidationInfo) -> datetime:
        assert_aware(v, info.field_name or "timestamp")
        return v


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Compute SHA-256 over canonical UTF-8 JSON."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def source_identity_material(
    provider_namespace: str,
    source_dataset: str,
    payload: dict[str, Any],
    *,
    stable_key_fields: Iterable[str] = (),
) -> str:
    """Return canonical material used for a provider-scoped source identity."""
    provider = provider_namespace.strip().lower()
    dataset = source_dataset.strip().lower()
    if not provider or not dataset:
        raise ValueError("provider_namespace and source_dataset must be non-empty")
    stable_keys = {
        field: payload[field]
        for field in stable_key_fields
        if field in payload and payload[field] not in (None, "")
    }
    return _canonical_json(
        {
            "provider_namespace": provider,
            "source_dataset": dataset,
            "stable_source_keys": stable_keys,
            "raw_payload_hash": compute_payload_hash(payload),
        }
    )


def compute_source_record_id(
    provider_namespace: str,
    source_dataset: str,
    payload: dict[str, Any],
    *,
    stable_key_fields: Iterable[str] = (),
) -> str:
    """Build a process-stable source ID without Python runtime hashing."""
    material = source_identity_material(
        provider_namespace,
        source_dataset,
        payload,
        stable_key_fields=stable_key_fields,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{provider_namespace.strip().lower()}:{source_dataset.strip().lower()}:{digest}"


class AvailabilityPolicy(BaseModel):
    """Policy that governs whether pre-announced records are allowed."""

    allow_pre_announced: bool = False

    def validate(  # type: ignore[override]
        self,
        event_time: datetime,
        available_time: datetime,
    ) -> None:
        from seatflow.domain.time import validate_available_after_event

        validate_available_after_event(
            event_time,
            available_time,
            allow_pre_announced=self.allow_pre_announced,
        )


class IngestionManifest(BaseModel):
    """Manifest recording what was ingested in a run."""

    ingestion_run_id: str
    source: str
    source_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    record_count: int = 0
    record_hashes: list[str] = Field(default_factory=list)
    manifest_hash: str = ""
    schema_version: str = "seatflow.bronze.v1"

    model_config = {"frozen": False}

    def compute_manifest_hash(self) -> str:
        """Compute a deterministic content hash for the ingested records.

        Run identifiers and timestamps remain on the manifest for auditability,
        but they are operational metadata rather than part of content identity.
        """
        payload = {
            "source": self.source,
            "source_version": self.source_version,
            "record_count": self.record_count,
            "record_hashes": sorted(self.record_hashes),
            "schema_version": self.schema_version,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.manifest_hash
