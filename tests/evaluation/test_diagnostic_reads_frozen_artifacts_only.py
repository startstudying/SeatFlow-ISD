from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from seatflow.evaluation.frozen_artifacts import FrozenArtifactError, FrozenTask25Store


def test_diagnostic_reads_frozen_artifacts_only(tmp_path: Path) -> None:
    task_root = tmp_path / "artifacts" / "r3_task25"
    task_root.mkdir(parents=True)
    artifact = task_root / "runs" / "run.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "inputs": [],
        "runs": [
            {
                "artifacts": [
                    {
                        "path": "artifacts/r3_task25/runs/run.json",
                        "bytes": artifact.stat().st_size,
                        "sha256": digest,
                    }
                ]
            }
        ],
    }
    store = FrozenTask25Store(tmp_path, manifest)

    assert store.read_json("runs/run.json") == {"frozen": True}
    with pytest.raises(FrozenArtifactError, match="escapes frozen root"):
        store.read_json("../../outside.json")
    with pytest.raises(FrozenArtifactError, match="not part of"):
        store.read_json("unfrozen.json")
