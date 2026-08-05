"""Release gate checks (Task 32).

Validates that all hard gates pass before a release can proceed.
No formal model claims are allowed. All checks are fail-closed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GateResult:
    """Result of a single gate check."""

    name: str
    passed: bool
    message: str
    details: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseReport:
    """Full release gate report."""

    version: str
    timestamp: str
    gates: list[GateResult]
    overall_passed: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def check_no_formal_claims(repo_root: Path) -> GateResult:
    """Gate 1: No formal model claims in the codebase."""
    forbidden_terms = [
        "formal model",
        "production model",
        "deployed model",
        "live trading",
        "real money",
    ]
    violations: list[str] = []
    for py_file in (repo_root / "research-engine" / "src").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        lower = content.lower()
        for term in forbidden_terms:
            if term in lower:
                violations.append(f"{py_file}: contains '{term}'")

    if violations:
        return GateResult(
            name="no_formal_claims",
            passed=False,
            message=f"Found {len(violations)} forbidden term(s)",
            details={"violations": "; ".join(violations[:10])},
        )
    return GateResult(
        name="no_formal_claims",
        passed=True,
        message="No formal model claims found",
    )


def check_package_boundary(repo_root: Path) -> GateResult:
    """Gate 2: V3 code does not import from legacy packages."""
    forbidden_imports = ["src.seatflow", "backend-java", "apps.services"]
    violations: list[str] = []
    for py_file in (repo_root / "research-engine" / "src").rglob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        for imp in forbidden_imports:
            if f"import {imp}" in content or f"from {imp}" in content:
                violations.append(f"{py_file}: imports '{imp}'")

    if violations:
        return GateResult(
            name="package_boundary",
            passed=False,
            message=f"Found {len(violations)} legacy import(s)",
            details={"violations": "; ".join(violations[:10])},
        )
    return GateResult(
        name="package_boundary",
        passed=True,
        message="No legacy imports detected",
    )


def check_truthfulness_config(repo_root: Path) -> GateResult:
    """Gate 3: Truthfulness config enforces research-only mode."""
    config_path = repo_root / "research-console" / "lib" / "config" / "truthfulness.ts"
    if not config_path.exists():
        return GateResult(
            name="truthfulness_config",
            passed=False,
            message="Truthfulness config file not found",
        )
    content = config_path.read_text(encoding="utf-8")
    required_fields = [
        'deploymentEnabled: false',
        'liveTradingEnabled: false',
        'formalPredictionsEnabled: false',
    ]
    missing = [f for f in required_fields if f not in content]
    if missing:
        return GateResult(
            name="truthfulness_config",
            passed=False,
            message=f"Missing required fields: {missing}",
        )
    return GateResult(
        name="truthfulness_config",
        passed=True,
        message="Truthfulness config enforces research-only mode",
    )


def check_contracts_exist(repo_root: Path) -> GateResult:
    """Gate 4: All required contracts exist."""
    required_contracts = [
        "contracts/json-schema/stock.schema.json",
        "contracts/json-schema/seat.schema.json",
        "contracts/json-schema/event-cycle.schema.json",
        "contracts/json-schema/prediction-sample.schema.json",
        "contracts/openapi/control-api.yaml",
    ]
    missing = []
    for contract in required_contracts:
        if not (repo_root / contract).exists():
            missing.append(contract)

    if missing:
        return GateResult(
            name="contracts_exist",
            passed=False,
            message=f"Missing {len(missing)} contract(s)",
            details={"missing": "; ".join(missing)},
        )
    return GateResult(
        name="contracts_exist",
        passed=True,
        message="All required contracts present",
    )


def run_release_gates(repo_root: Optional[Path] = None) -> ReleaseReport:
    """Run all release gate checks."""
    from datetime import datetime, timezone

    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    gates: list[GateResult] = [
        check_no_formal_claims(repo_root),
        check_package_boundary(repo_root),
        check_truthfulness_config(repo_root),
        check_contracts_exist(repo_root),
    ]

    overall = all(g.passed for g in gates)

    return ReleaseReport(
        version="3.0.0-dev",
        timestamp=datetime.now(timezone.utc).isoformat(),
        gates=gates,
        overall_passed=overall,
    )


def main() -> int:
    """Run release gates and print report. Exit 0 if all pass."""
    report = run_release_gates()
    print(report.to_json())
    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
