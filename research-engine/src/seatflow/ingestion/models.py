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
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

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
    def _must_be_aware(cls, v: datetime, info) -> datetime:
        assert_aware(v, info.field_name)
        return v


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Compute SHA-256 over canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AvailabilityPolicy(BaseModel):
    """Policy that governs whether pre-announced records are allowed."""

    allow_pre_announced: bool = False

    def validate(
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
