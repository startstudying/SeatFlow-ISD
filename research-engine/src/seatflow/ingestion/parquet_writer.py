"""Parquet partition writer for Bronze records."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from seatflow.ingestion.models import BronzeRecord


def write_bronze_partition(
    records: list[BronzeRecord],
    output_path: str | Path,
    *,
    partition_columns: list[str] | None = None,
) -> str:
    """Write Bronze records to a Parquet file.

    Returns the path of the written file as a string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        # Write an empty file with the schema for traceability.
        schema = pa.schema([
            ("event_time", pa.timestamp("us", tz="UTC")),
            ("available_time", pa.timestamp("us", tz="UTC")),
            ("ingested_at", pa.timestamp("us", tz="UTC")),
            ("source", pa.string()),
            ("source_record_id", pa.string()),
            ("source_version", pa.string()),
            ("revision_no", pa.int32()),
            ("raw_payload", pa.string()),
            ("raw_payload_hash", pa.string()),
            ("ingestion_run_id", pa.string()),
            ("source_confidence", pa.string()),
        ])
        table = pa.table({col: [] for col in schema.names}, schema=schema)
    else:
        data = {
            "event_time": [r.event_time for r in records],
            "available_time": [r.available_time for r in records],
            "ingested_at": [r.ingested_at for r in records],
            "source": [r.source for r in records],
            "source_record_id": [r.source_record_id for r in records],
            "source_version": [r.source_version for r in records],
            "revision_no": [r.revision_no for r in records],
            "raw_payload": [r.model_dump_json() for r in records],
            "raw_payload_hash": [r.raw_payload_hash for r in records],
            "ingestion_run_id": [r.ingestion_run_id for r in records],
            "source_confidence": [r.source_confidence.value for r in records],
        }
        table = pa.table(data)

    pq.write_table(table, str(output_path))
    return str(output_path)
