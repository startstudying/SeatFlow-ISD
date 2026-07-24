from __future__ import annotations

import argparse
import json
from pathlib import Path

from seatflow.training.r3_task25 import (
    DATASET_PATH,
    load_dataset,
    run_l2_cuda_memory_preflight,
    write_preflight_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run outcome-blind R3 Task25 CUDA preflight")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/r3_task25/preflight/cuda_memory.json"),
    )
    args = parser.parse_args()
    profiles = run_l2_cuda_memory_preflight(load_dataset(args.dataset))
    output = write_preflight_report(profiles, args.output)
    print(json.dumps({"output": str(output), "profiles": len(profiles)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
