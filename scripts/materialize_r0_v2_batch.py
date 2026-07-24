"""Resume-safe bounded runner for per-date V2 label/tensor materialization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = _arguments()
    event_root = Path(args.data_root) / "processed" / args.provider / "dragon_tiger_seat_event"
    dates = sorted(
        path.name.removeprefix("date=") for path in event_root.glob("date=*") if path.is_dir()
    )
    scoped_dates = [
        value
        for value in dates
        if (args.start_date is None or value >= args.start_date)
        and (args.end_date is None or value <= args.end_date)
    ]
    selected = [date for date in scoped_dates if not _completed(args.output_root, date)][
        : args.limit
    ]
    results: list[dict[str, object]] = []
    output = Path(args.output_root) / args.progress_file
    child_environment = _child_environment()
    _write_progress(
        output,
        args,
        results,
        status="batch_in_progress_fail_closed",
        selected_dates=selected,
    )
    for value in selected:
        event = event_root / f"date={value}" / "data.json"
        label = Path(args.output_root) / f"R0_V2_UNIVERSE_{value.replace('-', '_')}" / "result.json"
        tensor = Path(args.output_root) / f"R0_V2_TENSORS_{value.replace('-', '_')}"
        commands = (
            [
                sys.executable,
                "scripts/materialize_r0_v2_labels.py",
                "--as-of",
                value,
                "--event-snapshot",
                str(event),
                "--output",
                str(label),
            ] + (
                ["--observed-label-end", args.observed_label_end]
                if args.observed_label_end
                else []
            ),
            [
                sys.executable,
                "scripts/build_r0_v2_tensorized_snapshot.py",
                "--data-root",
                args.data_root,
                "--provider",
                args.provider,
                "--label-artifact",
                str(label),
                "--output-root",
                str(tensor),
            ],
        )
        for command in commands:
            _run_child(command, child_environment)
        results.append({"date": value, "label_artifact": str(label), "tensor_root": str(tensor)})
        _write_progress(
            output,
            args,
            results,
            status="batch_in_progress_fail_closed",
            selected_dates=selected,
        )
    _write_progress(output, args, results, status="batch_fail_closed", selected_dates=selected)
    remaining = sum(not _completed(args.output_root, date) for date in scoped_dates)
    print(
        json.dumps(
            {
                "processed_dates": len(results),
                "remaining_dates_in_scope": remaining,
                "progress_file": str(output),
            }
        )
    )


def _write_progress(
    output: Path,
    args: argparse.Namespace,
    results: list[dict[str, object]],
    *,
    status: str,
    selected_dates: list[str],
) -> None:
    """Atomically persist a fail-closed checkpoint after every completed date."""
    payload = {
        "status": status,
        "model_valid": False,
        "requested_limit": args.limit,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "selected_dates": selected_dates,
        "processed": results,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)


def _child_environment() -> dict[str, str]:
    """Make nested script calls independent of the parent's PYTHONPATH setup."""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else os.pathsep.join((source_root, existing))
    )
    return environment


def _run_child(command: list[str], environment: dict[str, str]) -> None:
    """Capture child output so detached batch jobs never inherit a stale stdout handle."""
    result = subprocess.run(command, check=False, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "child materialization command failed "
            f"(exit={result.returncode}, command={command!r}, "
            f"stdout={result.stdout[-2000:]!r}, stderr={result.stderr[-2000:]!r})"
        )


def _completed(root: str, date: str) -> bool:
    path = Path(root) / f"R0_V2_TENSORS_{date.replace('-', '_')}" / "manifest.json"
    if not path.exists():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("status") == "materialized_fail_closed" and raw.get("model_valid") is False


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/historical_backfill/tushare_2023_2026_v2")
    parser.add_argument("--provider", default="tushare")
    parser.add_argument("--output-root", default="artifacts/diagnostics")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start-date", help="Inclusive ISO as-of date for this owned batch shard.")
    parser.add_argument("--end-date", help="Inclusive ISO as-of date for this owned batch shard.")
    parser.add_argument(
        "--observed-label-end",
        help="latest date whose market data is permitted in training-label evidence",
    )
    parser.add_argument(
        "--progress-file",
        default="R0_V2_BATCH_PROGRESS.json",
        help="Progress filename beneath --output-root; use a distinct name per concurrent shard.",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must not be after --end-date")
    if args.observed_label_end:
        try:
            __import__("datetime").date.fromisoformat(args.observed_label_end)
        except ValueError:
            parser.error("--observed-label-end must be an ISO date")
    if Path(args.progress_file).is_absolute() or Path(args.progress_file).parent != Path("."):
        parser.error("--progress-file must be a filename, not a path")
    return args


if __name__ == "__main__":
    main()
