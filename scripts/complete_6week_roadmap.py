"""
6-week roadmap runner (Phase 0 / Issue #1 - fail-closed).

Previous version directly wrote fake metrics and ``passed=True`` into
phase-state.json, bypassing ``evaluate-promotion``. This version is
strictly fail-closed:

1. Each sub-task (ablations / industry-rerun) is executed via its
   executable runner, which validates outputs and writes provenance-
   tracked JSON. If any runner exits non-zero, this script exits
   non-zero **without** touching phase-state.json.

2. Only when all sub-tasks succeed, this script:
   - Reads the real ablation result JSON
   - Calls ``evaluate-promotion --metrics-file <real>``
   - The CLI atomically writes promotion decision + phase-state

3. Backups: before phase-state is updated, a backup is written as
   ``artifacts/status/phase-state.json.backup.<timestamp>``.

4. Evidence provenance: each result JSON contains dataset_version_sha,
   git_commit_sha, config_sha, seed×fold, started_at/ended_at, real_metrics,
   output_file_sha, and failure_reason.

5. Guards: if any file is missing, seed/fold count is insufficient, or
   metrics contain NaN/Inf, the runner exits non-zero and phase-state
   is never updated.

Usage:
    python scripts/complete_6week_roadmap.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ABLATION_RESULT_PATH = Path("artifacts/ablations/ablation_result.json")
INDUSTRY_RERUN_RESULT_PATH = Path(
    "artifacts/r5/r54_industry_rerun/r5_status.json"
)
PHASE_STATE_PATH = Path("artifacts/status/phase-state.json")


@dataclass(frozen=True)
class TaskResult:
    name: str
    success: bool
    path: Path
    reason: str | None


def main(argv: list[str] | None) -> int:
    config = _parse_args(argv)
    print("=" * 60)
    print("6-week roadmap runner (fail-closed Phase 0)")
    print("=" * 60)

    tasks = [
        (
            "seat_profile ablation",
            lambda: _run_ablation("seat_profile", config),
        ),
        (
            "time_edges ablation",
            lambda: _run_ablation("time_edges", config),
        ),
        (
            "direction_edges ablation",
            lambda: _run_ablation("direction_edges", config),
        ),
        ("industry rerun", lambda: _run_industry_rerun(config)),
    ]

    results: list[TaskResult] = []
    for name, fn in tasks:
        print(f"\n[{name}] starting...")
        try:
            proc = subprocess.run(fn(), capture_output=True, text=True)
            success = proc.returncode == 0
            path = (
                ABLATION_RESULT_PATH
                if "ablation" in name
                else INDUSTRY_RERUN_RESULT_PATH
            )
            reason = proc.stderr.strip()[:500] if proc.stderr else None
            results.append(TaskResult(name, success, path, reason))
            print(f"[{name}] {'ok' if success else 'FAILED'}")
            if not success:
                _cleanup_and_fail(results, "Sub-task failed")
                return 1
        except Exception as error:
            print(f"[{name}] EXCEPTION: {error}")
            path = (
                ABLATION_RESULT_PATH
                if "ablation" in name
                else INDUSTRY_RERUN_RESULT_PATH
            )
            results.append(TaskResult(name, False, path, str(error)))
            _cleanup_and_fail(results, "Sub-task raised exception")
            return 1

    print("\nAll sub-tasks succeeded; reading real artifacts...")

    ablation_payload = _load_ablation_result()
    industry_payload = _load_industry_rerun_result()
    metrics = _combine_metrics(ablation_payload, industry_payload)

    metrics_path = Path("artifacts/ablations/combined_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Combined metrics written: {metrics_path}")

    print("\nInvoking evaluate-promotion CLI...")
    _run_evaluate_promotion(metrics_path)
    print("evaluate-promotion completed; phase-state updated atomically")

    print("\n6-week roadmap runner: ok")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed 6-week roadmap runner (Phase 0)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/historical_backfill/tushare_2023_2026_v2"),
    )
    parser.add_argument(
        "--as-of-start", default="2024-01-02"
    )
    parser.add_argument(
        "--as-of-end", default="2024-03-29"
    )
    parser.add_argument(
        "--observed-label-end", default="2024-04-05"
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
    )
    return parser.parse_args(argv)


def _run_ablation(
    group: str, config: argparse.Namespace
) -> list[str]:
    seeds = config.seed if config.seed else []
    cmd = [
        sys.executable,
        "scripts/run_ablations.py",
        "--data-root",
        str(config.data_root),
        "--output-root",
        "artifacts/ablations",
        "--as-of-start",
        config.as_of_start,
        "--as-of-end",
        config.as_of_end,
        "--observed-label-end",
        config.observed_label_end,
    ] + [f"--seed={seed}" for seed in seeds]
    return cmd


def _run_industry_rerun(config: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/r54_industry_rerun.py",
        "--data-root",
        "data/historical_backfill/tushare_2023_2026_v2/r1_industry_membership",
    ]
    return cmd


def _run_evaluate_promotion(metrics_path: Path) -> None:
    _backup_phase_state()

    cmd = [
        sys.executable,
        "-m",
        "seatflow",
        "evaluate-promotion",
        "--metrics-file",
        str(metrics_path),
        "--output",
        "artifacts/evaluation/promotion-decision.json",
        "--status-file",
        str(PHASE_STATE_PATH),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"evaluate-promotion exit {proc.returncode}: {proc.stderr.strip()}"
        )
    print(proc.stdout.strip()[:2000])


def _load_ablation_result() -> dict[str, Any]:
    if not ABLATION_RESULT_PATH.exists():
        raise FileNotFoundError(f"ablation result missing: {ABLATION_RESULT_PATH}")
    try:
        raw = json.loads(ABLATION_RESULT_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ablation result unreadable: {error}") from error
    _validate_ablation_result(raw)
    return raw


def _load_industry_rerun_result() -> dict[str, Any]:
    if not INDUSTRY_RERUN_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"industry rerun result missing: {INDUSTRY_RERUN_RESULT_PATH}"
        )
    try:
        raw = json.loads(
            INDUSTRY_RERUN_RESULT_PATH.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"industry rerun result unreadable: {error}") from error
    _validate_industry_rerun_result(raw)
    return raw


def _validate_ablation_result(raw: Mapping[str, Any]) -> None:
    if raw.get("status") != "ok":
        reason = raw.get("failure_reason", "unknown")
        raise ValueError(f"ablation runner failed: {reason}")

    for key in ("dataset_version_sha", "git_commit_sha", "config_sha"):
        if not raw.get(key):
            raise ValueError(f"ablation result missing {key}")

    seeds = raw.get("seeds", [])
    if len(seeds) < 5:
        raise ValueError(f"ablation result insufficient seeds: {len(seeds)} < 5")

    folds = raw.get("folds", [])
    if len(folds) < 3:
        raise ValueError(f"ablation result insufficient folds: {len(folds)} < 3")

    metrics = raw.get("real_metrics", {})
    for row in metrics.get("metrics", []):
        for key in ("auc", "rank_ic"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"ablation metric {key} is NaN/Inf: {value}")

    for row in metrics.get("ablations", []):
        for key in ("auc_contribution", "rank_ic_contribution"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"ablation row {key} is NaN/Inf: {value}"
                )


def _validate_industry_rerun_result(raw: Mapping[str, Any]) -> None:
    if raw.get("status") != "ok":
        reason = raw.get("failure_reason", "unknown")
        raise ValueError(f"industry rerun failed: {reason}")

    e0 = raw.get("e0_mean_rank_ic")
    e1 = raw.get("e1_mean_rank_ic")
    improvement = raw.get("improvement")

    if e0 is None or e1 is None or improvement is None:
        raise ValueError("industry rerun result missing rank_ic values")

    if not (
        math.isfinite(e0)
        and math.isfinite(e1)
        and math.isfinite(improvement)
    ):
        raise ValueError("industry rerun result has NaN/Inf")

    if improvement <= 0:
        raise ValueError(f"industry rerun E1 ≤ E0 (improvement={improvement})")


def _combine_metrics(
    ablation: dict[str, Any], industry: dict[str, Any]
) -> dict[str, Any]:
    real = ablation.get("real_metrics", {})
    combined = {
        "audited": bool(real.get("audited")),
        "candidate_model": real.get("candidate_model"),
        "strong_baseline": real.get("strong_baseline"),
        "metrics": real.get("metrics", []),
        "ablations": real.get("ablations", []),
        "final_blind_manifest": real.get("final_blind_manifest"),
        "split_status": real.get("split_status"),
        "audit": real.get("audit", {}),
        "industry_rerun": {
            "e0_mean_rank_ic": industry.get("e0_mean_rank_ic"),
            "e1_mean_rank_ic": industry.get("e1_mean_rank_ic"),
            "improvement": industry.get("improvement"),
        },
    }
    return combined


def _cleanup_and_fail(results: list[TaskResult], global_reason: str) -> None:
    print(f"\n{global_reason}; phase-state not updated")
    for r in results:
        if not r.success:
            print(f"  - {r.name}: {r.reason}")


def _backup_phase_state() -> None:
    if not PHASE_STATE_PATH.exists():
        return
    backup_path = PHASE_STATE_PATH.with_suffix(
        f".backup.{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    backup_path.write_bytes(PHASE_STATE_PATH.read_bytes())
    print(f"Phase state backed up: {backup_path}")


if __name__ == "__main__":
    raise SystemExit(main(None))