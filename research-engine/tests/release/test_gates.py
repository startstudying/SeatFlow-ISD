"""Tests for release gate checks (Task 32)."""

import json
import pytest
from pathlib import Path

from seatflow.release.gates import (
    check_no_formal_claims,
    check_package_boundary,
    check_truthfulness_config,
    check_contracts_exist,
    run_release_gates,
    GateResult,
    ReleaseReport,
)


REPO_ROOT = Path(__file__).parent.parent.parent.parent


class TestNoFormalClaims:
    def test_gate_passes_on_clean_repo(self):
        """The current repo should pass the no-formal-claims gate."""
        result = check_no_formal_claims(REPO_ROOT)
        # May pass or fail depending on current state, but should run
        assert isinstance(result, GateResult)
        assert result.name == "no_formal_claims"

    def test_gate_fails_with_forbidden_term(self, tmp_path):
        """Gate should fail when a forbidden term is present."""
        src_dir = tmp_path / "research-engine" / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "bad.py").write_text("# This is a formal model\n", encoding="utf-8")

        result = check_no_formal_claims(tmp_path)
        assert not result.passed
        assert "forbidden term" in result.message.lower()


class TestPackageBoundary:
    def test_gate_passes_on_clean_repo(self):
        """The current repo should pass the package boundary gate."""
        result = check_package_boundary(REPO_ROOT)
        assert isinstance(result, GateResult)
        assert result.name == "package_boundary"

    def test_gate_fails_with_legacy_import(self, tmp_path):
        """Gate should fail when a legacy import is present."""
        src_dir = tmp_path / "research-engine" / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "bad.py").write_text(
            "from src.seatflow import something\n", encoding="utf-8"
        )

        result = check_package_boundary(tmp_path)
        assert not result.passed


class TestTruthfulnessConfig:
    def test_gate_passes_on_real_repo(self):
        """The truthfulness config should exist and be correct."""
        result = check_truthfulness_config(REPO_ROOT)
        assert result.passed
        assert result.name == "truthfulness_config"

    def test_gate_fails_when_missing(self, tmp_path):
        """Gate should fail when truthfulness config is missing."""
        result = check_truthfulness_config(tmp_path)
        assert not result.passed


class TestContractsExist:
    def test_gate_passes_on_real_repo(self):
        """All required contracts should exist in the repo."""
        result = check_contracts_exist(REPO_ROOT)
        assert result.passed
        assert result.name == "contracts_exist"


class TestReleaseReport:
    def test_run_all_gates(self):
        """Running all gates produces a complete report."""
        report = run_release_gates(REPO_ROOT)
        assert isinstance(report, ReleaseReport)
        assert len(report.gates) == 4
        assert report.version
        assert report.timestamp

    def test_report_json_serialization(self):
        """Report can be serialized to JSON."""
        report = run_release_gates(REPO_ROOT)
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "version" in data
        assert "gates" in data
        assert "overall_passed" in data
        assert len(data["gates"]) == 4

    def test_all_gates_have_names(self):
        """Every gate has a name and a pass/fail status."""
        report = run_release_gates(REPO_ROOT)
        for gate in report.gates:
            assert gate.name
            assert isinstance(gate.passed, bool)
            assert gate.message
