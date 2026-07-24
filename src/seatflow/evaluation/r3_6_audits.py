"""Protocol audits that precede the R3.6 strategy gate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from seatflow.evaluation.frozen_artifacts import (
    FrozenTask25Store,
    build_frozen_manifest,
    verify_frozen_manifest,
)
from seatflow.evaluation.temporal_holdings import (
    analyze_temporal_holdings,
    cross_seed_same_day_jaccard,
)


def run_r3_6_protocol_audits(repository_root: str | Path) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    output_root = repository / "artifacts" / "r3_6"
    output_root.mkdir(parents=True, exist_ok=True)
    frozen = build_frozen_manifest(repository, source_commit="0e2ccfb")
    verify_frozen_manifest(repository, frozen)
    store = FrozenTask25Store(repository, frozen)
    run_manifest = _mapping(store.read_json("run_manifest.json"))

    temporal_runs: dict[str, object] = {}
    cross_seed_rows: dict[tuple[str, int], list[dict[str, object]]] = {}
    for run in _mapping_sequence(run_manifest["run_order"]):
        objective = str(run["objective"])
        if objective not in {"L0", "L2"}:
            continue
        relative = Path(str(run["output_directory"])) / "daily_predictions.jsonl"
        predictions = [dict(row) for row in _mapping_sequence(store.read_json(relative))]
        run_id = str(run["run_id"])
        temporal_runs[run_id] = {
            "objective": objective,
            "fold": run["fold"],
            "seed": run["seed"],
            "analysis": analyze_temporal_holdings(predictions, top_k=(5, 10)),
        }
        for row in predictions:
            cross_seed_rows.setdefault(
                (objective, int(str(run["fold"]))), []
            ).append({**row, "seed": run["seed"]})

    cross_seed = {
        f"{objective}/fold_{fold}": {
            "top5": cross_seed_same_day_jaccard(rows, top_k=5),
            "top10": cross_seed_same_day_jaccard(rows, top_k=10),
        }
        for (objective, fold), rows in sorted(cross_seed_rows.items())
    }
    temporal_payload = {
        "schema_version": "r3_6_temporal_holdings.v1",
        "run_count": len(temporal_runs),
        "runs": temporal_runs,
        "cross_seed_same_day": cross_seed,
        "exit_reason_availability": {
            "candidate_exit": "available",
            "rank_change": "available",
            "untradable": "unavailable_no_tradability_field",
            "forced_sell": "unavailable_no_forced_sell_field",
            "corporate_action": "unavailable_no_corporate_action_field",
            "missing_market_data": "unavailable_no_market_data_status_field",
        },
    }
    _write_json(output_root / "temporal_holdings.json", temporal_payload)

    replay = _mapping(
        json.loads((output_root / "strategy_replay.json").read_text(encoding="utf-8"))
    )
    unit_rows = []
    for audit_row in _mapping_sequence(replay["runs"]):
        for metric in (
            "mean_gross_spread",
            "mean_net_spread",
            "mean_cost_diff",
            "worst_daily_net_spread",
            "max_additive_drawdown",
        ):
            units = _mapping(audit_row[f"{metric}_units"])
            required = {
                "decimal",
                "percent",
                "bps",
                "aggregation_method",
                "number_of_days",
            }
            if set(units) != required:
                raise ValueError(f"{metric} unit metadata is incomplete")
            if int(str(units["number_of_days"])) != int(
                str(audit_row["day_count"])
            ):
                raise ValueError(f"{metric} day count does not match replay")
            unit_rows.append(
                {
                    "strategy_id": audit_row["strategy_id"],
                    "run_id": audit_row["run_id"],
                    "cost_multiplier": audit_row["cost_multiplier"],
                    "metric": metric,
                    **units,
                }
            )
    unit_payload = {
        "schema_version": "r3_6_return_unit_audit.v1",
        "status": "passed",
        "metric_count": len(unit_rows),
        "metrics": unit_rows,
        "semantic_rule": (
            "daily_arithmetic_mean is a daily expectation; sum and compound "
            "are separate cumulative aggregations and are never substituted"
        ),
    }
    _write_json(output_root / "return_unit_audit.json", unit_payload)
    verify_frozen_manifest(repository, frozen)
    return {"temporal": temporal_payload, "return_units": unit_payload}


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise TypeError("expected list of mappings")
    return list(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
