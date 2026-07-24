from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest


def _module() -> object:
    path = Path("scripts/materialize_r0_v2_batch.py")
    spec = importlib.util.spec_from_file_location("r0_v2_batch", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load batch materialization script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_checkpoint_is_atomic_and_fail_closed(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "progress.json"
    args = argparse.Namespace(limit=20, start_date="2024-03-27", end_date="2024-04-25")
    processed = [{"date": "2024-03-27", "tensor_root": "tensor"}]

    module._write_progress(
        output,
        args,
        processed,
        status="batch_in_progress_fail_closed",
        selected_dates=["2024-03-27", "2024-03-28"],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "batch_in_progress_fail_closed"
    assert payload["model_valid"] is False
    assert payload["processed"] == processed
    assert payload["selected_dates"] == ["2024-03-27", "2024-03-28"]
    assert not output.with_suffix(".json.tmp").exists()


def test_child_environment_always_includes_project_source_root(monkeypatch) -> None:
    module = _module()
    monkeypatch.delenv("PYTHONPATH", raising=False)

    environment = module._child_environment()

    expected = str(Path("scripts/materialize_r0_v2_batch.py").resolve().parents[1] / "src")
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == expected


def test_child_failure_reports_captured_output(monkeypatch) -> None:
    module = _module()

    class _Result:
        returncode = 120
        stdout = "child standard output"
        stderr = "child standard error"

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: _Result())

    with pytest.raises(RuntimeError, match="child standard error"):
        module._run_child(["python", "child.py"], {"PYTHONPATH": "src"})
