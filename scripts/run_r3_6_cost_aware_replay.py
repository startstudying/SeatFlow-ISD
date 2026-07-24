from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from seatflow.evaluation.r3_6_replay import run_r3_6_replay
from seatflow.training.r3_task25 import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen, training-free SeatFlow R3.6 strategy replay"
    )
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    samples = load_dataset(args.dataset) if args.dataset else load_dataset()
    result = run_r3_6_replay(args.repository, samples)
    selection = cast("dict[str, object]", result["selection"])
    print(f"status={selection['status']}")
    print(f"selected_strategy={selection['selected_strategy']}")
    print(f"task48_allowed={selection['task48_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
