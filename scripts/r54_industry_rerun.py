"""
R5.4 industry rerun script (Issue #4).

Previous version was incomplete: it called a non-existent module entry point
and did not verify the output. This version is fail-closed.

It runs ``dt-rolling-run`` (E0 baseline + E1 with industry), then reads the
real ``artifacts/r5/r54_industry_rerun/r5_status.json`` artifact and validates
that E1 > E0. If the result is missing, invalid, or E1 ≤ E0, it raises
SystemExit(1) and writes a failure report, ensuring the roadmap runner
keeps the blocker.

Usage:
    python scripts/r54_industry_rerun.py \
        --data-root data/historical_backfill/tushare_2023_2026_v2/r1_industry_membership
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

STATUS_FILE = "r5_status.json"
INDUSTRY_MEMBERSHIP_ROOT = "data/historical_backfill/tushare_2023_2026_v2/r1_industry_membership"
R54_OUTPUT_ROOT = "artifacts/r5/r54_industry_rerun"


@dataclass(frozen=True)
class RerunConfig:
    data_root: Path
    output_root: Path
    config_path: Path
    protocol_path: Path


def main(argv: list[str] | None) -> int:
    config = _parse_args(argv)
    config.output_root.mkdir(parents=True, exist_ok=True)
    status_path = config.output_root / STATUS_FILE
    started_at = _now_iso()

    if not config.data_root.exists():
        _write_failure(
            status_path,
            config,
            started_at,
            f"industry membership data root missing: {config.data_root}",
        )
        raise SystemExit(1)

    result_path = config.output_root / "dt_rolling_result.json"
    cmd = _build_cli_command(config, result_path)

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        reason = f"dt-rolling-run exit {proc.returncode}: {proc.stderr.strip()[:2000]}"
        _write_failure(status_path, config, started_at, reason)
        raise SystemExit(1) from None

    try:
        raw = json.loads(result_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        reason = f"dt-rolling-run produced unreadable result: {error}"
        _write_failure(status_path, config, started_at, reason)
        raise SystemExit(1) from error

    try:
        _validate_result(raw)
    except ValueError as error:
        _write_failure(status_path, config, started_at, str(error))
        raise SystemExit(1) from error

    e0_rank_ic = _extract_rank_ic(raw, "strong_baseline")
    e1_rank_ic = _extract_rank_ic(raw, "dt_seathgt")

    if e0_rank_ic is None or e1_rank_ic is None:
        _write_failure(
            status_path,
            config,
            started_at,
            f"rank_ic missing: E0={e0_rank_ic} E1={e1_rank_ic}",
        )
        raise SystemExit(1)

    if not (math.isfinite(e0_rank_ic) and math.isfinite(e1_rank_ic)):
        _write_failure(
            status_path,
            config,
            started_at,
            f"rank_ic non-finite: E0={e0_rank_ic} E1={e1_rank_ic}",
        )
        raise SystemExit(1)

    improvement = e1_rank_ic - e0_rank_ic
    print(f"E0 mean_rank_ic: {e0_rank_ic}")
    print(f"E1 mean_rank_ic: {e1_rank_ic}")
    print(f"E1 vs E0 improvement: {improvement:.4f}")

    if improvement <= 0:
        reason = f"E1 ≤ E0 (improvement={improvement:.4f})"
        _write_failure(status_path, config, started_at, reason)
        raise SystemExit(1)

    payload = {
        "status": "ok",
        "started_at": started_at,
        "ended_at": _now_iso(),
        "e0_mean_rank_ic": e0_rank_ic,
        "e1_mean_rank_ic": e1_rank_ic,
        "improvement": round(improvement, 6),
        "failure_reason": None,
        "raw_command": cmd,
    }
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("R5.4 industry rerun ok")
    return 0


def _parse_args(argv: list[str] | None) -> RerunConfig:
    parser = argparse.ArgumentParser(
        description="Fail-closed R5.4 industry rerun: E1 > E0 validation"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(INDUSTRY_MEMBERSHIP_ROOT),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(R54_OUTPUT_ROOT)
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument(
        "--research-protocol-config",
        type=Path,
        default=Path("configs/experiments/research_protocol_v2.toml"),
    )
    args = parser.parse_args(argv)
    return RerunConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        config_path=args.config,
        protocol_path=args.research_protocol_config,
    )


def _build_cli_command(config: RerunConfig, result_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "seatflow",
        "dt-rolling-run",
        "--config",
        str(config.config_path),
        "--data-root",
        str(config.data_root.parent),
        "--output-root",
        str(config.output_root),
        "--result-file",
        str(result_path),
        "--as-of-start",
        "2024-01-02",
        "--as-of-end",
        "2024-03-29",
        "--observed-label-end",
        "2024-04-05",
        "--research-protocol-config",
        str(config.protocol_path),
    ]


def _validate_result(raw: Mapping[str, Any]) -> None:
    if not raw.get("audited"):
        raise ValueError("dt-rolling-run result is not audited (audited != true)")
    if raw.get("status") not in {"evaluated_fail_closed"}:
        raise ValueError(f"dt-rolling-run did not reach evaluated_fail_closed: {raw.get('status')}")

    for row in raw.get("metrics", []):
        if not isinstance(row, Mapping):
            raise ValueError("metric row is not an object")
        for key in ("auc", "rank_ic"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"metric row {key} is NaN/Inf: {value}")


def _extract_rank_ic(raw: Mapping[str, Any], model: str) -> float | None:
    rows = raw.get("metrics", [])
    if not isinstance(rows, list):
        return None
    values: list[float] = []
    for row in rows:
        if isinstance(row, Mapping) and row.get("model") == model:
            rank_ic = row.get("rank_ic")
            if isinstance(rank_ic, (int, float)) and math.isfinite(rank_ic):
                values.append(rank_ic)
    if not values:
        return None
    return sum(values) / len(values)


def _write_failure(
    status_path: Path,
    config: RerunConfig,
    started_at: str,
    reason: str,
) -> None:
    payload = {
        "status": "failed",
        "started_at": started_at,
        "ended_at": _now_iso(),
        "e0_mean_rank_ic": None,
        "e1_mean_rank_ic": None,
        "improvement": None,
        "failure_reason": reason,
        "raw_command": None,
    }
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(None))