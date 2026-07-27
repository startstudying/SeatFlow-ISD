"""Run the R4 structure-only gate and fail closed on missing daily evidence."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, cast

from seatflow.evaluation.structure_preflight import (
    evaluate_structure_preflight,
    fail_closed_structure_gate,
    structure_gate_decision,
)
from seatflow.evaluation.turnover_floor import DailyTurnoverFloor

DAILY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "prediction_date",
    "persistent_candidate_pool_size",
    "candidate_pool_temporal_jaccard",
    "candidate_exit_turnover_floor",
    "simulated_rule_b_turnover",
    "average_retained_holdings",
    "full_turnover_day_ratio",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--daily-input",
        type=Path,
        help="Immutable daily R4 structure parquet; omit to record insufficient evidence",
    )
    args = parser.parse_args()
    repo_root = cast(Path, args.repo_root).resolve()
    output_dir = repo_root / "artifacts" / "r4"
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_output = output_dir / "task58_daily_structure.parquet"
    pd = cast(Any, importlib.import_module("pandas"))

    daily_input = cast(Path | None, args.daily_input)
    if daily_input is None or not daily_input.is_file():
        frame = pd.DataFrame(columns=DAILY_COLUMNS)
        frame.to_parquet(daily_output, index=False, engine="pyarrow")
        result_payload = _insufficient_payload()
        decision = fail_closed_structure_gate()
    else:
        frame = pd.read_parquet(daily_input)
        missing_columns = sorted(set(DAILY_COLUMNS) - set(frame.columns))
        if missing_columns or frame.empty:
            frame = pd.DataFrame(columns=DAILY_COLUMNS)
            frame.to_parquet(daily_output, index=False, engine="pyarrow")
            result_payload = _insufficient_payload(
                f"daily structure evidence missing columns: {missing_columns}"
            )
            decision = fail_closed_structure_gate()
        else:
            frame = frame.loc[:, DAILY_COLUMNS].copy()
            frame.to_parquet(daily_output, index=False, engine="pyarrow")
            p0_turnover = _load_p0_turnover(output_dir / "task51_turnover_floor.json")
            floor_rows = [
                _floor_row(index, row)
                for index, row in enumerate(frame.to_dict(orient="records"))
            ]
            result = evaluate_structure_preflight(
                p0_mean_turnover=p0_turnover,
                turnover_floor_rows=floor_rows,
                persistent_pool_sizes=[
                    int(value) for value in frame["persistent_candidate_pool_size"]
                ],
                persistent_pool_jaccards=[
                    float(value) for value in frame["candidate_pool_temporal_jaccard"]
                ],
                simulated_rule_b_turnovers=[
                    float(value) for value in frame["simulated_rule_b_turnover"]
                ],
                retained_holding_counts=[
                    int(value) for value in frame["average_retained_holdings"]
                ],
            )
            result_payload = {
                "schema_version": "r4_task58_structure_preflight.v1",
                **result.to_dict(),
                "stop_reason": None if result.status == "passed" else result.status,
            }
            decision = structure_gate_decision(result)

    (output_dir / "task58_structure_preflight.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "task58_gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if decision["passed"] else 2


def _insufficient_payload(
    reason: str = (
        "complete daily persistent_candidate_universe and simulated_rule_b_turnover "
        "evidence were not available as immutable artifacts"
    ),
) -> dict[str, object]:
    return {
        "schema_version": "r4_task58_structure_preflight.v1",
        "status": "insufficient_evidence",
        "evidence_complete": False,
        "persistent_candidate_pool_size": None,
        "candidate_pool_temporal_jaccard": None,
        "candidate_exit_turnover_floor": None,
        "simulated_rule_b_turnover": None,
        "average_retained_holdings": None,
        "full_turnover_day_ratio": None,
        "turnover_reduction_vs_p0": None,
        "stop_reason": reason,
        "source_files_checked": [
            "artifacts/r3_6/strategy_replay.json",
            "artifacts/r3_6/strategy_daily_results.json",
            "artifacts/r3_6/temporal_holdings.json",
            "artifacts/r4/task51_turnover_floor.json",
        ],
    }


def _load_p0_turnover(path: Path) -> float | None:
    payload = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    value = payload.get("mean_total_turnover")
    return float(value) if isinstance(value, (int, float)) else None


def _floor_row(index: int, row: MappingRow) -> DailyTurnoverFloor:
    return DailyTurnoverFloor(
        run_id=str(row["run_id"]),
        prediction_date=str(row["prediction_date"]),
        previous_prediction_date="",
        portfolio_size=5,
        previous_holding_count=5,
        current_candidate_count=_as_int(row["persistent_candidate_pool_size"]),
        candidate_exit_count=0,
        candidate_exit_turnover_floor=_as_float(row["candidate_exit_turnover_floor"]),
        total_turnover=_as_float(row["simulated_rule_b_turnover"]),
        removable_turnover_above_floor=0.0,
        candidate_pool_temporal_jaccard=_as_float(
            row["candidate_pool_temporal_jaccard"]
        ),
        candidate_pool_retention_rate=0.0,
        has_previous_holdings=index > 0,
    )


MappingRow = dict[str, object]


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("daily structure numeric fields must be numeric")
    return float(value)


def _as_int(value: object) -> int:
    return int(_as_float(value))


if __name__ == "__main__":
    raise SystemExit(main())
