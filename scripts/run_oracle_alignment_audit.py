from __future__ import annotations

import argparse
from pathlib import Path

from seatflow.evaluation.r3_5_audit import run_r3_5_audit
from seatflow.training.r3_task25 import load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the read-only SeatFlow R3.5 audit")
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--skip-relation-audit",
        action="store_true",
        help="Skip checkpoint inference masking and record insufficient evidence",
    )
    args = parser.parse_args()
    run_r3_5_audit(
        args.repository,
        load_dataset(),
        run_relation_audit=not args.skip_relation_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
