"""
Executable ablation runner (Issue #3).

Previous version of this script was a broken stub: it called
``run_dt_seathgt_rolling_evaluation`` with a non-existent ``ablation_group``
keyword and ``load_verified_historical_snapshots()`` with no arguments, so it
could never run. This version is fail-closed and exec.

The DT-SeatHGT rolling evaluator computes *all three* required ablation groups
(seat_profile / time_edges / direction_edges) in a single pass, so this runner
runs one evaluation over real verified snapshots, then captures the full
provenance a fail-closed evidence chain needs:

    dataset_version_sha   — sha256 of the snapshot manifest hashes
    git_commit_sha        — HEAD commit at run start
    config_sha            — sha256 of the serialized run configuration
    seeds / folds         — pre-registered grid actually executed
    started_at / ended_at — ISO-8601 wall-clock bounds
    real_metrics          — audited rolling metrics (auc / rank_ic / ablations)
    output_file_sha       — sha256 of the emitted artifact
    status / failure_reason — "ok" or a human-readable blocker

The runner NEVER fabricates metrics. If the underlying training cannot run
(missing data root, missing PyTorch, NaN/Inf metrics, insufficient seeds or
folds, mismatched dataset version) it raises SystemExit(1) and writes nothing
into phase-state.

Usage:
    python scripts/run_ablations.py \
        --data-root data/historical_backfill/tushare_2023_2026_v2 \
        --output-root artifacts/ablations \
        --as-of-start 2024-01-02 --as-of-end 2024-03-29 \
        --observed-label-end 2024-04-05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_GROUPS: tuple[str, ...] = ("seat_profile", "time_edges", "direction_edges")
DEFAULT_SEEDS: tuple[int, ...] = (2025, 2026, 2027, 2028, 2029)
MIN_FOLD_COUNT = 3
RESULT_FILENAME = "ablation_result.json"


@dataclass(frozen=True)
class RunnerConfig:
    data_root: Path
    output_root: Path
    as_of_start: str
    as_of_end: str
    observed_label_end: str
    seeds: tuple[int, ...]
    config_path: Path
    protocol_path: Path
    selection_horizon: int

    def to_provenance(self) -> dict[str, object]:
        return {
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "as_of_start": self.as_of_start,
            "as_of_end": self.as_of_end,
            "observed_label_end": self.observed_label_end,
            "seeds": list(self.seeds),
            "config_path": str(self.config_path),
            "protocol_path": str(self.protocol_path),
            "selection_horizon": self.selection_horizon,
        }


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    config.output_root.mkdir(parents=True, exist_ok=True)
    result_path = config.output_root / RESULT_FILENAME
    started_at = _now_iso()

    if not config.data_root.exists():
        _write_failure(result_path, config, started_at, f"data root missing: {config.data_root}")
        raise SystemExit(1)

    git_sha = _git_commit_sha()
    config_sha = _sha256_json(config.to_provenance())

    raw_result_path = config.output_root / "dt_rolling_result.json"
    cmd = _build_cli_command(config, raw_result_path)

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        reason = f"dt-rolling-run exit {proc.returncode}: {proc.stderr.strip()[:2000]}"
        _write_failure(
            result_path, config, started_at, reason,
            git_sha=git_sha, config_sha=config_sha,
        )
        raise SystemExit(1) from None

    try:
        raw = json.loads(raw_result_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        reason = f"dt-rolling-run produced unreadable result: {error}"
        _write_failure(
            result_path, config, started_at, reason,
            git_sha=git_sha, config_sha=config_sha,
        )
        raise SystemExit(1) from error

    try:
        dataset_sha = _dataset_version_sha(raw)
        _validate_result(raw, config, dataset_sha)
    except ValueError as error:
        _write_failure(
            result_path, config, started_at, str(error),
            git_sha=git_sha, config_sha=config_sha,
        )
        raise SystemExit(1) from error

    output_sha = _sha256_file(raw_result_path)
    provenance = {
        "status": "ok",
        "dataset_version_sha": dataset_sha,
        "git_commit_sha": git_sha,
        "config_sha": config_sha,
        "seeds": list(config.seeds),
        "folds": _fold_ids(raw),
        "started_at": started_at,
        "ended_at": _now_iso(),
        "real_metrics": _real_metrics(raw),
        "output_file": str(raw_result_path),
        "output_file_sha": output_sha,
        "failure_reason": None,
        "raw_command": cmd,
    }
    result_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"ablation runner ok -> {result_path}")
    return 0


def _parse_args(argv: list[str] | None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Executable fail-closed ablation runner")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/ablations"))
    parser.add_argument("--as-of-start", required=True)
    parser.add_argument("--as-of-end", required=True)
    parser.add_argument("--observed-label-end", required=True)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument(
        "--research-protocol-config",
        type=Path,
        default=Path("configs/experiments/research_protocol_v2.toml"),
    )
    parser.add_argument("--selection-horizon", type=int, default=1)
    args = parser.parse_args(argv)

    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    if len(set(seeds)) != len(seeds):
        parser.error("seeds must be unique")
    return RunnerConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        as_of_start=args.as_of_start,
        as_of_end=args.as_of_end,
        observed_label_end=args.observed_label_end,
        seeds=seeds,
        config_path=args.config,
        protocol_path=args.research_protocol_config,
        selection_horizon=args.selection_horizon,
    )


def _build_cli_command(config: RunnerConfig, result_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "seatflow",
        "dt-rolling-run",
        "--config",
        str(config.config_path),
        "--data-root",
        str(config.data_root),
        "--output-root",
        str(config.output_root),
        "--result-file",
        str(result_path),
        "--as-of-start",
        config.as_of_start,
        "--as-of-end",
        config.as_of_end,
        "--observed-label-end",
        config.observed_label_end,
        "--research-protocol-config",
        str(config.protocol_path),
        "--selection-horizon",
        str(config.selection_horizon),
        *[f"--seed={seed}" for seed in config.seeds],
    ]


def _validate_result(
    raw: Mapping[str, Any], config: RunnerConfig, dataset_sha: str
) -> None:
    if not raw.get("audited"):
        raise ValueError("dt-rolling-run result is not audited (audited != true)")
    if raw.get("status") not in {"evaluated_fail_closed"}:
        raise ValueError(
            f"dt-rolling-run did not reach evaluated_fail_closed: {raw.get('status')}"
        )

    seeds_seen = {str(seed) for seed in raw.get("seeds", [])}
    expected_seeds = {str(seed) for seed in config.seeds}
    if seeds_seen != expected_seeds:
        raise ValueError(
            "seed mismatch: runner expected "
            f"{sorted(expected_seeds)} but result has {sorted(seeds_seen)}"
        )

    folds = _fold_ids(raw)
    if len(folds) < MIN_FOLD_COUNT:
        raise ValueError(f"insufficient folds: {len(folds)} < {MIN_FOLD_COUNT}")

    ablation_rows = raw.get("ablations", [])
    if not isinstance(ablation_rows, list) or not ablation_rows:
        raise ValueError("ablation rows missing from dt-rolling-run result")
    present_groups = {
        _canonical(row.get("group"))
        for row in ablation_rows
        if isinstance(row, Mapping)
    }
    missing = {g for g in REQUIRED_GROUPS if g not in present_groups}
    if missing:
        raise ValueError(f"ablation groups missing: {sorted(missing)}")

    for row in ablation_rows:
        for key in ("auc_contribution", "rank_ic_contribution"):
            value = row.get(key)
            if value is None:
                raise ValueError(f"ablation row {row.get('group')} missing {key}")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"ablation row {row.get('group')} {key} is NaN/Inf: {value}")

    for row in raw.get("metrics", []):
        if not isinstance(row, Mapping):
            raise ValueError("metric row is not an object")
        for key in ("auc", "rank_ic"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"metric row {key} is NaN/Inf: {value}")

    audit = raw.get("audit", {})
    if isinstance(audit, Mapping):
        audit_sha = audit.get("snapshot_manifest_hash")
        if isinstance(audit_sha, str) and audit_sha and audit_sha != dataset_sha:
            raise ValueError(
                f"dataset version mismatch: audit={audit_sha} derived={dataset_sha}"
            )


def _dataset_version_sha(raw: Mapping[str, Any]) -> str:
    audit = raw.get("audit", {})
    if isinstance(audit, Mapping):
        manifest_hash = audit.get("snapshot_manifest_hash")
        if isinstance(manifest_hash, str) and manifest_hash:
            return manifest_hash
    manifest = raw.get("manifest_hashes", {})
    if isinstance(manifest, Mapping) and manifest:
        return _sha256_json(dict(sorted(manifest.items())))
    raise ValueError("dataset version sha unavailable: no audit.snapshot_manifest_hash")


def _fold_ids(raw: Mapping[str, Any]) -> list[str]:
    folds = raw.get("folds", [])
    if not isinstance(folds, list):
        return []
    ids: list[str] = []
    for fold in folds:
        if isinstance(fold, Mapping):
            fold_id = fold.get("fold_id") or fold.get("id")
            if isinstance(fold_id, str):
                ids.append(fold_id)
    return ids


def _real_metrics(raw: Mapping[str, Any]) -> dict[str, object]:
    return {
        "audited": bool(raw.get("audited")),
        "candidate_model": raw.get("candidate_model"),
        "strong_baseline": raw.get("strong_baseline"),
        "metrics": raw.get("metrics", []),
        "ablations": raw.get("ablations", []),
        "final_blind_manifest": raw.get("final_blind_manifest"),
        "split_status": raw.get("split_status"),
        "audit": raw.get("audit", {}),
    }


def _write_failure(
    result_path: Path,
    config: RunnerConfig,
    started_at: str,
    reason: str,
    *,
    git_sha: str | None = None,
    config_sha: str | None = None,
) -> None:
    payload = {
        "status": "failed",
        "dataset_version_sha": None,
        "git_commit_sha": git_sha,
        "config_sha": config_sha,
        "seeds": list(config.seeds),
        "folds": [],
        "started_at": started_at,
        "ended_at": _now_iso(),
        "real_metrics": None,
        "output_file": None,
        "output_file_sha": None,
        "failure_reason": reason,
        "raw_command": None,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _canonical(group: object) -> str:
    if not isinstance(group, str):
        return ""
    return group.strip().lower().replace("-", "_")


def _git_commit_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(None))
