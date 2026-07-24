from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TypeAlias, cast

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]


class SnapshotNotFoundError(LookupError):
    """Raised when a requested on-disk snapshot does not exist."""


@dataclass(frozen=True)
class DatasetSummary:
    store: str
    dataset: str
    snapshot_count: int
    row_count: int
    latest_trade_date: str | None
    latest_manifest: JsonDict | None

    def to_dict(self) -> JsonDict:
        return {
            "store": self.store,
            "dataset": self.dataset,
            "snapshot_count": self.snapshot_count,
            "row_count": self.row_count,
            "latest_trade_date": self.latest_trade_date,
            "latest_manifest": self.latest_manifest,
        }


class SnapshotQueryService:
    """Read-only facade over SnapshotStore-compatible data directories."""

    _STORES = ("raw", "processed", "features", "graphs")
    _PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

    def snapshot_summary(self, provider: str) -> JsonDict:
        self._validate_provider(provider)
        summaries: list[DatasetSummary] = []
        for store in self._STORES:
            provider_dir = self.data_root / store / provider
            if not provider_dir.exists():
                continue
            for dataset_dir in sorted(path for path in provider_dir.iterdir() if path.is_dir()):
                dataset_summary = self._dataset_summary(store, dataset_dir)
                if dataset_summary.snapshot_count:
                    summaries.append(dataset_summary)

        if not summaries:
            raise SnapshotNotFoundError(f"no snapshots found for provider '{provider}'")

        return {
            "provider": provider,
            "datasets": [summary.to_dict() for summary in summaries],
            "snapshot_count": sum(summary.snapshot_count for summary in summaries),
            "row_count": sum(summary.row_count for summary in summaries),
        }

    def graph_snapshot(self, provider: str, trade_date: date) -> JsonDict:
        self._validate_provider(provider)
        snapshot_dir = (
            self.data_root / "graphs" / provider / "hetero_graph" / f"date={trade_date.isoformat()}"
        )
        data_path = snapshot_dir / "data.json"
        manifest_path = snapshot_dir / "manifest.json"
        if not data_path.exists() or not manifest_path.exists():
            raise SnapshotNotFoundError(
                f"graph snapshot not found for provider '{provider}' on {trade_date.isoformat()}"
            )

        rows = self._read_json_list(data_path)
        if not rows:
            raise SnapshotNotFoundError(
                f"graph snapshot is empty for provider '{provider}' on {trade_date.isoformat()}"
            )
        graph = rows[0]
        if not isinstance(graph, dict):
            raise ValueError(f"graph snapshot row must be an object: {data_path}")

        return {
            "provider": provider,
            "dataset": "hetero_graph",
            "trade_date": trade_date.isoformat(),
            "manifest": self._read_json_dict(manifest_path),
            "graph": graph,
        }

    def _dataset_summary(self, store: str, dataset_dir: Path) -> DatasetSummary:
        manifests = [
            self._read_json_dict(manifest_path)
            for manifest_path in sorted(dataset_dir.glob("date=*/manifest.json"))
        ]
        latest_manifest = max(
            manifests,
            key=lambda manifest: str(manifest.get("trade_date") or ""),
            default=None,
        )
        return DatasetSummary(
            store=store,
            dataset=dataset_dir.name,
            snapshot_count=len(manifests),
            row_count=sum(self._int_value(manifest.get("row_count")) for manifest in manifests),
            latest_trade_date=(
                str(latest_manifest["trade_date"])
                if latest_manifest and "trade_date" in latest_manifest
                else None
            ),
            latest_manifest=latest_manifest,
        )

    @staticmethod
    def _read_json_dict(path: Path) -> JsonDict:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"snapshot manifest must be an object: {path}")
        return cast(JsonDict, raw)

    @staticmethod
    def _read_json_list(path: Path) -> list[JsonValue]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"snapshot data must be a list: {path}")
        return cast(list[JsonValue], raw)

    @staticmethod
    def _int_value(value: JsonValue) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    @classmethod
    def _validate_provider(cls, provider: str) -> None:
        if not cls._PROVIDER_PATTERN.fullmatch(provider):
            raise SnapshotNotFoundError("provider identifier is invalid")
