from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from seatflow.evaluation.r3_6_audits import run_r3_6_protocol_audits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SeatFlow R3.6 return-unit and temporal-holding audits"
    )
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_r3_6_protocol_audits(args.repository)
    temporal = cast("dict[str, object]", result["temporal"])
    return_units = cast("dict[str, object]", result["return_units"])
    print(f"temporal_runs={temporal['run_count']}")
    print(f"return_metrics={return_units['metric_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
