"""Task 6C CLI status contract tests."""

from __future__ import annotations

import json

import pytest

from seatflow.training.base_oof_cli import main


def _json_from_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_synthetic_smoke_cli_is_completed_but_non_formal(tmp_path, capsys) -> None:
    exit_code = main(["--synthetic-smoke", "--output-dir", str(tmp_path)])
    payload = _json_from_stdout(capsys)
    assert exit_code == 0
    assert payload["status"] == "COMPLETED"
    assert payload["classification"] == "SYNTHETIC_NON_FORMAL"
    assert "metrics" not in payload
    assert payload["predictions_path"]
    assert payload["manifest_path"]


def test_formal_cli_is_blocked_until_issued_task5e_attestation_exists(capsys) -> None:
    exit_code = main(["--formal"])
    payload = _json_from_stdout(capsys)
    assert exit_code == 2
    assert payload["status"] == "BLOCKED"
    assert "FORMAL_AUDIT_ATTESTATION_REQUIRED" in payload["reasons"]
    assert "FORMAL_DATASET_SEAL_REQUIRED" in payload["reasons"]
    assert "metrics" not in payload


def test_cli_unexpected_error_returns_machine_json_exit1(capsys) -> None:
    exit_code = main(["--synthetic-smoke"])
    payload = _json_from_stdout(capsys)
    assert exit_code == 1
    assert payload["status"] == "ERROR"
    assert "output_dir" in payload["reason"]
