from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class SerializableRecord(Protocol):
    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SnapshotArtifact:
    data_path: Path
    manifest_path: Path
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "data_path": str(self.data_path),
            "manifest_path": str(self.manifest_path),
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(
        self,
        provider: str,
        dataset: str,
        trade_date: date,
        rows: Iterable[SerializableRecord | Mapping[str, object]],
        metadata: Mapping[str, object] | None = None,
    ) -> SnapshotArtifact:
        serialized_rows = [self._to_dict(row) for row in rows]
        output_dir = self.root / provider / dataset / f"date={trade_date.isoformat()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / "data.json"
        data_bytes = json.dumps(
            serialized_rows,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=self._json_default,
        ).encode("utf-8")
        digest = sha256(data_bytes).hexdigest()
        self._atomic_write(data_path, data_bytes)

        manifest = {
            "provider": provider,
            "dataset": dataset,
            "trade_date": trade_date.isoformat(),
            "row_count": len(serialized_rows),
            "sha256": digest,
            "written_at": datetime.now().astimezone().isoformat(),
            "metadata": dict(metadata or {}),
        }
        manifest_path = output_dir / "manifest.json"
        self._atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        )
        return SnapshotArtifact(data_path, manifest_path, len(serialized_rows), digest)

    def read_all(self, provider: str, dataset: str) -> tuple[dict[str, object], ...]:
        dataset_dir = self.root / provider / dataset
        if not dataset_dir.exists():
            return ()
        rows: list[dict[str, object]] = []
        for data_path in sorted(dataset_dir.glob("date=*/data.json")):
            raw = json.loads(data_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError(f"snapshot data must be a list: {data_path}")
            rows.extend(raw)
        return tuple(rows)

    @staticmethod
    def _to_dict(row: SerializableRecord | Mapping[str, object]) -> dict[str, object]:
        if isinstance(row, Mapping):
            return dict(row)
        return row.to_dict()

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        with NamedTemporaryFile(
            "wb", delete=False, dir=path.parent, prefix=f".{path.name}."
        ) as stream:
            stream.write(content)
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
