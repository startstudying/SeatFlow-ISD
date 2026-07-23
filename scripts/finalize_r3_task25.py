from __future__ import annotations

from pathlib import Path

from seatflow.evaluation.r3_task25 import (
    create_completion_manifest,
    evaluate_completed_study,
)
from seatflow.training.r3_task25 import load_dataset


def main() -> int:
    root = Path("artifacts/r3_task25")
    planned = root / "run_manifest.json"
    completion = create_completion_manifest(
        planned,
        root / "completion_manifest.json",
    )
    evaluate_completed_study(
        load_dataset(),
        planned_manifest_path=planned,
        completion_manifest_path=completion,
        output_path=root / "evaluation.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
