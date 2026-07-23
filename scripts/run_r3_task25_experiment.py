from __future__ import annotations

import argparse
from pathlib import Path

from seatflow.training.r3_task25 import (
    DATASET_PATH,
    create_immutable_run_manifest,
    execute_all_planned_runs,
    load_dataset,
)

CONFIG = Path("configs/experiments/r3_task25_frozen.toml")
OUTPUT = Path("artifacts/r3_task25")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute frozen R3 Task25 loss study")
    parser.add_argument("command", choices=("manifest", "run"))
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    samples = load_dataset(args.dataset)
    if args.command == "manifest":
        create_immutable_run_manifest(
            samples,
            config_path=args.config,
            output_root=args.output,
        )
    else:
        execute_all_planned_runs(
            samples,
            config_path=args.config,
            manifest_path=args.output / "run_manifest.json",
            output_root=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
