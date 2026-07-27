"""Build R4 Task 51 Rule-A structural turnover-floor artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
from functools import cache
from pathlib import Path
from typing import Any, cast

from seatflow.evaluation.turnover_floor import (
    DailyTurnoverFloor,
    analyze_candidate_turnover_floor,
    summarize_turnover_floor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--portfolio-size", type=int, default=5)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = repo_root / "artifacts" / "r4"
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_rows: list[dict[str, object]] = []
    try:
        runs = _load_daily_evidence(repo_root)
        daily_objects: list[DailyTurnoverFloor] = []
        for run_id, evidence in runs.items():
            daily, _summary = analyze_candidate_turnover_floor(
                daily_candidates=evidence["candidates"],
                daily_holdings=evidence["holdings"],
                portfolio_size=args.portfolio_size,
                run_id=run_id,
            )
            daily_objects.extend(daily)
            daily_rows.extend(row.to_dict() for row in daily)
        summary = summarize_turnover_floor(daily_objects)
        payload: dict[str, Any] = {
            "schema_version": "r4_task51_turnover_floor.v1",
            "status": "ok",
            "gate_passed": True,
            "evidence_complete": True,
            "portfolio_size": args.portfolio_size,
            "missing_fields": [],
            "turnover_decomposition_complete": False,
            "unavailable_decomposition_fields": [
                "untradable_turnover",
                "corporate_action_turnover",
            ],
            "source_files": [
                "artifacts/r3_6/strategy_daily_results.json",
                "artifacts/r3_task25/runs/*/*/*/predictions.json",
            ],
            **summary,
        }
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        payload = {
            "schema_version": "r4_task51_turnover_floor.v1",
            "status": "insufficient_evidence",
            "gate_passed": False,
            "evidence_complete": False,
            "portfolio_size": args.portfolio_size,
            "missing_fields": [str(exc)],
            "turnover_decomposition_complete": False,
            "unavailable_decomposition_fields": [
                "untradable_turnover",
                "corporate_action_turnover",
            ],
            "mean_candidate_exit_turnover_floor": None,
            "median_candidate_exit_turnover_floor": None,
            "full_candidate_exit_day_ratio": None,
            "candidate_pool_temporal_jaccard": None,
            "candidate_pool_retention_rate": None,
            "maximum_possible_reduction_under_rule_A": None,
        }

    (output_dir / "task51_turnover_floor.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_daily(output_dir / "task51_daily_turnover_floor.parquet", daily_rows)


def _load_daily_evidence(
    repo_root: Path,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    daily_results_path = repo_root / "artifacts" / "r3_6" / "strategy_daily_results.json"
    raw = json.loads(daily_results_path.read_text(encoding="utf-8"))
    raw_rows = raw["rows"]
    evidence: dict[str, dict[str, dict[str, list[str]]]] = {}
    for row in raw_rows:
        if row.get("strategy_id") != "S0/P0" or row.get("cost_multiplier") != 1.0:
            continue
        run_id = _as_str(row["run_id"])
        predictions = _load_predictions_for_run(repo_root, run_id)
        date = _as_str(row["prediction_date"])
        if date not in predictions:
            raise ValueError(f"missing predictions for {run_id} {date}")
        run_evidence = evidence.setdefault(run_id, {"candidates": {}, "holdings": {}})
        holdings = row["holdings"]
        if not isinstance(holdings, list):
            raise ValueError(f"holdings for {run_id} {date} is not a list")
        run_evidence["candidates"][date] = predictions[date]
        run_evidence["holdings"][date] = [str(stock_id) for stock_id in holdings]
    if not evidence:
        raise ValueError("no S0/P0 daily rows found")
    return evidence


@cache
def _load_predictions_for_run(repo_root: Path, run_id: str) -> dict[str, list[str]]:
    # R3 run ids are L0-f0-s2026; prediction paths are runs/L0/fold_0/seed_2026.
    parts = run_id.split("-")
    if len(parts) != 3 or not parts[1].startswith("f") or not parts[2].startswith("s"):
        raise ValueError(f"unsupported run_id format: {run_id}")
    level = parts[0]
    fold = f"fold_{parts[1][1:]}"
    seed = f"seed_{parts[2][1:]}"
    path = repo_root / "artifacts" / "r3_task25" / "runs" / level / fold / seed / "predictions.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    predictions: dict[str, list[str]] = {}
    for date, payload in raw.items():
        stock_ids = payload["stock_ids"]
        if not isinstance(stock_ids, list):
            raise ValueError(f"stock_ids for {run_id} {date} is not a list")
        predictions[str(date)] = [str(stock_id) for stock_id in stock_ids]
    return predictions


def _write_daily(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        pd = importlib.import_module("pandas")
    except ImportError:
        fallback_path = path.with_suffix(".json")
        fallback_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        raise RuntimeError(f"pandas is required to write {path}") from None
    data_frame_factory = cast(Any, pd).DataFrame
    data_frame_factory(rows).to_parquet(path, index=False)


def _as_str(value: object) -> str:
    return str(value)


if __name__ == "__main__":
    main()
