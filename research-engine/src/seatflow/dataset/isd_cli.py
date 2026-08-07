"""Build and audit CLI for SeatFlow-ISD V1.0 (Task 5).

Usage
-----
    seatflow-isd build --source-root PATH --output-root PATH \
        --dataset-version VERSION [--materialized-at ISO8601] \
        [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] \
        [--require-formal]

    seatflow-isd audit --dataset-root PATH [--require-formal]

Exit codes
----------
    0 — engineering build or audit succeeded
    2 — invalid arguments or corrupt inputs
    3 — ``--require-formal`` requested and formal validity is blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from seatflow.dataset.isd_builder import (
    ISDBuildConfig,
    audit_isd_dataset,
    build_isd_dataset,
)


def _parse_iso_or_now(value: str | None) -> datetime:
    """Parse an ISO-8601 timestamp or return current UTC time."""
    if value is None:
        return datetime.now(tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO-8601 timestamp: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _print_json(payload: dict[str, Any]) -> None:
    """Print a JSON payload to stdout with stable key order."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _cmd_build(args: argparse.Namespace) -> int:
    """Execute a dataset build."""
    try:
        materialized_at = _parse_iso_or_now(args.materialized_at)
    except argparse.ArgumentTypeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    source_root = Path(args.source_root)
    if not source_root.is_dir():
        sys.stderr.write(f"error: source root not found: {source_root}\n")
        return 2

    output_root = Path(args.output_root)

    config = ISDBuildConfig(
        source_root=source_root,
        output_root=output_root,
        dataset_version=args.dataset_version,
        materialized_at=materialized_at,
        start_date=args.start_date if args.start_date else "2023-01-03",
        end_date=args.end_date if args.end_date else "2026-08-06",
    )

    try:
        result = build_isd_dataset(config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        sys.stderr.write(f"error: build failed: {exc}\n")
        return 2

    payload = result.as_dict()
    _print_json(payload)

    if args.require_formal and not result.dataset_valid:
        blockers = result.quality_report.formal_blockers
        sys.stderr.write(
            f"formal validity blocked: {len(blockers)} blocker(s)\n"
        )
        for b in blockers:
            sys.stderr.write(f"  - {b}\n")
        return 3

    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Execute a dataset audit."""
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        sys.stderr.write(f"error: dataset root not found: {dataset_root}\n")
        return 2

    try:
        audit = audit_isd_dataset(dataset_root)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        sys.stderr.write(f"error: audit failed: {exc}\n")
        return 2

    payload = audit.as_dict()
    _print_json(payload)

    if args.require_formal and not audit.dataset_valid:
        sys.stderr.write("formal validity blocked: dataset_valid=false\n")
        return 3

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="seatflow-isd",
        description="Build and audit SeatFlow-ISD V1.0 datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ───────────────────────────────────────────────────
    build_p = sub.add_parser("build", help="Build a dataset version.")
    build_p.add_argument(
        "--source-root", required=True,
        help="Root directory of verified Tushare source snapshots.",
    )
    build_p.add_argument(
        "--output-root", required=True,
        help="Root directory for dataset output.",
    )
    build_p.add_argument(
        "--dataset-version", required=True,
        help="Deterministic dataset version label.",
    )
    build_p.add_argument(
        "--materialized-at", default=None,
        help="ISO-8601 timestamp for label maturity cutoff (default: now UTC).",
    )
    build_p.add_argument(
        "--start-date", default=None,
        help="Optional start date filter (YYYY-MM-DD).",
    )
    build_p.add_argument(
        "--end-date", default=None,
        help="Optional end date filter (YYYY-MM-DD).",
    )
    build_p.add_argument(
        "--require-formal", action="store_true",
        help="Exit 3 if formal validity is blocked.",
    )

    # ── audit ───────────────────────────────────────────────────
    audit_p = sub.add_parser("audit", help="Audit a built dataset version.")
    audit_p.add_argument(
        "--dataset-root", required=True,
        help="Path to a built dataset version directory.",
    )
    audit_p.add_argument(
        "--require-formal", action="store_true",
        help="Exit 3 if formal validity is blocked.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns exit code 0/2/3 per the contract.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return code

    if args.command == "build":
        return _cmd_build(args)
    elif args.command == "audit":
        return _cmd_audit(args)
    else:
        parser.print_help(sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
