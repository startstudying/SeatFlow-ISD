"""Graph state store for managing snapshot lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from seatflow.graph.snapshot import GraphSnapshot


class GraphStateStore:
    """In-memory store for graph snapshots.

    In production this would be backed by MinIO/Parquet.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, GraphSnapshot] = {}

    def save(self, snapshot: GraphSnapshot) -> None:
        if snapshot.snapshot_id in self._snapshots:
            raise ValueError(
                f"Snapshot {snapshot.snapshot_id} already exists — snapshots are immutable"
            )
        self._snapshots[snapshot.snapshot_id] = snapshot

    def get(self, snapshot_id: str) -> GraphSnapshot:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot {snapshot_id} not found")
        return self._snapshots[snapshot_id]

    def get_by_cutoff(self, cutoff_time: datetime) -> Optional[GraphSnapshot]:
        """Get the latest snapshot at or before cutoff_time."""
        candidates = [
            s for s in self._snapshots.values()
            if s.cutoff_time <= cutoff_time
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.cutoff_time)

    def list_snapshots(self) -> list[str]:
        return sorted(self._snapshots.keys())
