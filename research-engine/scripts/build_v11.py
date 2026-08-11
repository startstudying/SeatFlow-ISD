"""V1.1 build runner: re-emit the ISD V1.0 dataset with all formal gates
satisfied by sourcing CSI500, adj_factor, tradeability, and SW2021
industry membership from the existing Tushare backfill on disk.

Usage:
    python -m research_engine.scripts.build_v11

No data is re-fetched. The runner reads
``D:/SeatFlow/data/historical_backfill/tushare_2023_2026_v2`` and
emits ``D:/SeatFlow/datasets/seatflow_isd_v1/v1.1-20260806``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-version", default="v1.1-20260806")
    parser.add_argument(
        "--materialized-at",
        default="2026-08-06T16:00:00+08:00",
    )
    args = parser.parse_args()

    from seatflow.dataset.isd_builder import ISDBuildConfig, build_isd_dataset

    materialized_at = datetime.fromisoformat(args.materialized_at)
    if materialized_at.tzinfo is None:
        materialized_at = materialized_at.replace(tzinfo=timezone.utc)

    source_root = Path(args.source_root)
    industry_root = source_root / "r1_industry_membership/tushare"

    config = ISDBuildConfig(
        source_root=source_root,
        output_root=Path(args.output_root),
        dataset_version=args.dataset_version,
        materialized_at=materialized_at,
        start_date="2023-01-03",
        end_date="2026-08-06",
        benchmark_root=source_root / "raw/tushare",
        benchmark_id="CSI500",
        adjustment_root=source_root / "raw/tushare",
        tradeability_root=source_root / "raw/tushare",
        industry_root=industry_root,
    )

    result = build_isd_dataset(config)
    import json
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0 if result.dataset_valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
