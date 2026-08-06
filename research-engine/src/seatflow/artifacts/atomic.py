"""Atomic local artifact publication for research-engine-local outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


JsonMapping = Mapping[str, Any]


def atomic_write_json(path: Path, payload: JsonMapping) -> None:
    """Write a deterministic JSON object via fsync + os.replace.

    Existing final artifacts are immutable and are never overwritten.
    """
    _ensure_new_final(path)
    data = _json_dumps(payload) + "\n"
    _atomic_write_text(path, data)


def atomic_write_jsonl(path: Path, rows: Iterable[JsonMapping]) -> None:
    """Write deterministic JSONL rows via fsync + os.replace."""
    _ensure_new_final(path)
    _atomic_write_jsonl_stream(path, rows)


def load_jsonl_artifact(path: Path) -> list[dict[str, Any]]:
    """Load a final JSONL artifact and reject partial temp artifacts."""
    if path.suffix == ".tmp" or ".tmp" in path.name:
        raise ValueError("partial JSONL artifact is not loadable")
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"blank JSONL row at line {line_number}")
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("JSONL artifact is empty")
    return rows


def _ensure_new_final(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"final artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, data: str) -> None:
    parent = path.parent
    error: BaseException | None = None
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_name = handle.name
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception as exc:
            error = exc
    if error is not None:
        Path(temp_name).unlink(missing_ok=True)
        raise error
    _publish_no_clobber(Path(temp_name), path)


def _atomic_write_jsonl_stream(path: Path, rows: Iterable[JsonMapping]) -> None:
    parent = path.parent
    count = 0
    error: BaseException | None = None
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_name = handle.name
        try:
            for row in rows:
                handle.write(_json_dumps(row) + "\n")
                count += 1
            if count == 0:
                raise ValueError("JSONL artifact must contain at least one row")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception as exc:
            error = exc
    if error is not None:
        Path(temp_name).unlink(missing_ok=True)
        raise error
    _publish_no_clobber(Path(temp_name), path)


def _publish_no_clobber(temp_path: Path, final_path: Path) -> None:
    try:
        os.link(temp_path, final_path)
        _fsync_directory(final_path.parent)
    except FileExistsError:
        raise
    except OSError as exc:
        if final_path.exists():
            raise FileExistsError(f"final artifact already exists: {final_path}") from exc
        raise
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_dumps(payload: JsonMapping) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
