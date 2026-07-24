from __future__ import annotations

import importlib.util
from pathlib import Path


def _module() -> object:
    path = Path("scripts/audit_r0_v2_batch_attempts.py")
    spec = importlib.util.spec_from_file_location("r0_v2_batch_attempt_audit", path)
    if spec is None or spec.loader is None:
        raise AssertionError("failed to load R0 V2 batch attempt audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attempt_audit_marks_nonempty_stderr_as_recorded_failure(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "r0_v2_first.err.log").write_text("failure", encoding="utf-8")
    (tmp_path / "r0_v2_second.err.log").write_text("", encoding="utf-8")

    rows = module._attempts(tmp_path)

    assert [row["stderr_state"] for row in rows] == [
        "recorded_failure",
        "no_recorded_stderr",
    ]
    assert rows[0]["stderr_byte_count"] == 7
