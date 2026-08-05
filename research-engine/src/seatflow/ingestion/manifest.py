"""Ingestion manifest builder."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from seatflow.ingestion.models import BronzeRecord, IngestionManifest
from seatflow.domain.time import utc_now


def build_manifest(
    records: Iterable[BronzeRecord],
    *,
    ingestion_run_id: str,
    source: str,
    source_version: str,
    started_at: datetime | None = None,
) -> IngestionManifest:
    """Build an IngestionManifest from an iterable of Bronze records."""
    started = started_at or utc_now()
    record_hashes: list[str] = []
    count = 0
    for record in records:
        record_hashes.append(record.raw_payload_hash)
        count += 1

    manifest = IngestionManifest(
        ingestion_run_id=ingestion_run_id,
        source=source,
        source_version=source_version,
        started_at=started,
        completed_at=utc_now(),
        record_count=count,
        record_hashes=record_hashes,
    )
    manifest.compute_manifest_hash()
    return manifest
