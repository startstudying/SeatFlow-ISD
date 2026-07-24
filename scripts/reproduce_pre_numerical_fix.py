"""Freeze and verify the immutable pre-numerical-fix DT-SeatHGT baseline.

This is deliberately an artifact reproducer, not a training entry point.  The
legacy tensor schema and checkpoint format are about to become invalid, so
rerunning them with changed code would not reproduce the evidence.  ``--freeze``
copies the identified legacy inputs once; ``--verify`` checks the resulting
hash manifest and the fail-closed outcome recorded in result.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPOSITORY_ROOT / "artifacts" / "baselines" / "pre_numerical_fix"
LEGACY_ROOT = REPOSITORY_ROOT / "artifacts" / "research" / "dtseathgt_real_bounded_20260721"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "research_protocol_v2.toml"
MANIFEST_NAME = "baseline.manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def legacy_files() -> list[Path]:
    """Return the precise legacy payload required to audit the old result."""
    selected = [LEGACY_ROOT / "result.json", LEGACY_ROOT / "matured_labels.sqlite3", CONFIG_PATH]
    selected.extend(sorted(LEGACY_ROOT.glob("dtseathgt-fold_*-seed-*/dtseathgt.pt")))
    selected.extend(sorted((LEGACY_ROOT / "tensorized_chunks").glob("*.pkl")))
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing legacy baseline inputs: " + ", ".join(map(str, missing)))
    return selected


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate_model"]
    metrics = [row for row in result["metrics"] if row["model"] == candidate]
    if len(metrics) != 15:
        raise ValueError(f"Expected 15 candidate seed/fold rows, found {len(metrics)}")
    def mean(field: str) -> float:
        return sum(float(row[field]) for row in metrics) / len(metrics)
    return {
        "candidate_model": candidate,
        "strong_baseline": result["strong_baseline"],
        "status": result["status"],
        "model_valid": result["model_valid"],
        "negative_result": result["negative_result"],
        "schema_version": result["schema_version"],
        "seed_count": len(result["seeds"]),
        "fold_count": result["audit"]["fold_count"],
        "candidate_seed_fold_rows": len(metrics),
        "mean_auc": mean("auc"),
        "mean_rank_ic": mean("rank_ic"),
        "mean_topk_excess_return_after_cost": mean("topk_excess_return_after_cost"),
        "positive_topk_rows": sum(row["topk_excess_return_after_cost"] > 0 for row in metrics),
        "audit": result["audit"],
    }


def freeze() -> None:
    if BASELINE_ROOT.exists() and any(BASELINE_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing baseline: {BASELINE_ROOT}")
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    files = legacy_files()
    entries: list[dict[str, Any]] = []
    for source in files:
        source_relative = relative_to_root(source)
        destination = BASELINE_ROOT / "payload" / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append(
            {
                "source": source_relative,
                "path": destination.relative_to(BASELINE_ROOT).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    result_path = BASELINE_ROOT / "payload" / relative_to_root(LEGACY_ROOT / "result.json")
    summary = result_summary(json.loads(result_path.read_text(encoding="utf-8")))
    manifest = {
        "baseline_id": "pre_numerical_fix",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Frozen legacy evidence before numerical-stability and tensor-schema changes.",
        "source_run": relative_to_root(LEGACY_ROOT),
        "source_result_sha256": sha256(result_path),
        "payload_file_count": len(entries),
        "payload_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
        "result_summary": summary,
        "reproduction": {
            "command": "python scripts/reproduce_pre_numerical_fix.py --verify",
            "scope": (
                "Integrity and outcome reproduction from frozen legacy artifacts; "
                "not a retraining command."
            ),
        },
    }
    (BASELINE_ROOT / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Frozen {len(entries)} files ({manifest['payload_bytes']} bytes) at {BASELINE_ROOT}")


def verify() -> None:
    manifest_path = BASELINE_ROOT / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Baseline manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in manifest["files"]:
        path = BASELINE_ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"hash mismatch: {entry['path']}")
    result_path = BASELINE_ROOT / "payload" / relative_to_root(LEGACY_ROOT / "result.json")
    if result_path.is_file():
        summary = result_summary(json.loads(result_path.read_text(encoding="utf-8")))
        expected = manifest["result_summary"]
        for key in (
            "status",
            "model_valid",
            "negative_result",
            "candidate_seed_fold_rows",
            "mean_auc",
            "mean_rank_ic",
        ):
            if summary[key] != expected[key]:
                failures.append(f"result summary mismatch: {key}")
        if (
            summary["status"] != "evaluated_fail_closed"
            or summary["model_valid"]
            or not summary["negative_result"]
        ):
            failures.append(
                "legacy result no longer represents the expected fail-closed negative baseline"
            )
    else:
        failures.append("frozen result.json is missing")
    if failures:
        raise RuntimeError("Baseline verification failed:\n- " + "\n- ".join(failures))
    print(
        "Verified pre_numerical_fix baseline: "
        f"{manifest['payload_file_count']} files, {manifest['payload_bytes']} bytes, "
        f"status={manifest['result_summary']['status']}, "
        f"model_valid={manifest['result_summary']['model_valid']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", action="store_true", help="copy the legacy payload and write its manifest"
    )
    parser.add_argument(
        "--verify", action="store_true", help="verify hashes and the recorded fail-closed outcome"
    )
    args = parser.parse_args()
    if args.freeze == args.verify:
        parser.error("choose exactly one of --freeze or --verify")
    try:
        if args.freeze:
            freeze()
        else:
            verify()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
