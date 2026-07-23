from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seatflow.evaluation.stability import PredictionRow, RunMetric, analyze_stability


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze R2 cross-seed stability")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_stability(
        (PredictionRow.from_mapping(row) for row in payload.get("predictions", [])),
        (RunMetric.from_mapping(row) for row in payload.get("run_metrics", [])),
        ablations=payload.get("ablations"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

