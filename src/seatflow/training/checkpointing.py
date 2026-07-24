from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class CheckpointArtifact:
    checkpoint_path: Path
    manifest_path: Path
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "manifest_path": str(self.manifest_path),
            "sha256": self.sha256,
        }


def write_json_checkpoint(
    output_dir: str | Path,
    run_id: str,
    payload: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> CheckpointArtifact:
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    root = Path(output_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    checkpoint_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    digest = sha256(checkpoint_bytes).hexdigest()
    _atomic_write(checkpoint_path, checkpoint_bytes)

    manifest_path = root / "checkpoint.manifest.json"
    manifest = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "sha256": digest,
        "written_at": datetime.now().astimezone().isoformat(),
        "metadata": dict(metadata or {}),
    }
    _atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    return CheckpointArtifact(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        sha256=digest,
    )


def hash_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def hash_json(payload: object) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    with NamedTemporaryFile("wb", delete=False, dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(content)
        temporary_path = Path(stream.name)
    temporary_path.replace(path)
