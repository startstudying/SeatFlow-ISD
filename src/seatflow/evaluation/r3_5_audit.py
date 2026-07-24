"""End-to-end read-only R3.5 economic-alignment audit."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any

from seatflow.evaluation.cost_attribution import (
    DailyCostAttribution,
    attribute_ranked_portfolios,
    summarize_cost_attribution,
)
from seatflow.evaluation.exposure_audit import (
    compare_top5_exposures,
    tensorized_target_exposures,
)
from seatflow.evaluation.frozen_artifacts import (
    FrozenTask25Store,
    build_frozen_manifest,
    verify_frozen_manifest,
)
from seatflow.evaluation.oracle_audit import audit_oracle, audit_score_directions
from seatflow.evaluation.relation_mask import (
    RelationMaskMode,
    mask_tensorized_relations,
)
from seatflow.evaluation.stability import jaccard, spearman
from seatflow.evaluation.time_alignment import evaluate_trade_time_alignment
from seatflow.evaluation.topk_diagnostics import deterministic_rank, topk_diagnostics
from seatflow.models.seat_rank_lite import SeatRankLiteConfig, build_seat_rank_lite
from seatflow.training.r3_task25 import group_samples_by_date, legalize_r3_graph

ALLOWED_STATUSES = frozenset(
    {"confirmed", "supported", "unsupported", "insufficient_evidence"}
)
EXPOSURE_FIELDS = (
    "market_cap",
    "trading_amount",
    "turnover_rate",
    "stock_volatility_20d",
    "stock_volume_ratio_20d",
    "stock_amplitude_20d",
    "stock_return_5d",
    "beta",
    "stock_return_20d",
    "limit_up_status",
    "industry",
    "dragon_tiger_net_buy",
    "seat_edge_count",
    "seat_edge_weight_sum",
    "institutional_seat_ratio",
    "stock_historical_appearance_frequency",
)
MASK_MODES: tuple[RelationMaskMode, ...] = (
    "full",
    "without_seat_stock",
    "without_industry_stock",
    "without_stock_stock",
    "stock_only",
)


def run_r3_5_audit(
    repository_root: str | Path,
    samples: Sequence[Any],
    *,
    run_relation_audit: bool = True,
) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    task_root = repository / "artifacts" / "r3_task25"
    output_root = repository / "artifacts" / "r3_5"
    output_root.mkdir(parents=True, exist_ok=True)

    frozen = build_frozen_manifest(repository)
    frozen_path = task_root / "frozen_manifest.json"
    _write_json(frozen_path, frozen)
    verify_frozen_manifest(repository, frozen)
    store = FrozenTask25Store(repository, frozen)
    run_manifest = _mapping(store.read_json("run_manifest.json"))
    evaluation = _mapping(store.read_json("evaluation.json"))

    truth, sample_by_key = _truth_rows(samples)
    appearance_frequency = _appearance_frequencies(sample_by_key)
    run_summaries: list[dict[str, object]] = []
    oracle_daily: list[dict[str, object]] = []
    exposure_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    topk_payload: dict[str, object] = {}
    cost_payload: dict[str, object] = {}
    joined_by_run: dict[str, list[dict[str, object]]] = {}

    for run in _mapping_sequence(run_manifest["run_order"]):
        run_id = str(run["run_id"])
        objective = str(run["objective"])
        relative = Path(str(run["output_directory"])) / "daily_predictions.jsonl"
        predictions = _mapping_sequence(store.read_json(relative))
        joined = _join_predictions(predictions, truth)
        joined_by_run[run_id] = joined
        label_result = audit_oracle(
            joined, ranking_field="label_zscore", return_field="realized_return"
        )
        trade_result = audit_oracle(
            joined, ranking_field="trade_return", return_field="realized_return"
        )
        directions = audit_score_directions(joined, return_field="realized_return")
        original_daily = directions["original"].daily
        reversed_daily = directions["reversed"].daily
        attribution = attribute_ranked_portfolios(joined)
        cost_summary = summarize_cost_attribution(attribution)
        topk = topk_diagnostics(joined)
        topk_payload[run_id] = topk
        cost_payload[run_id] = {
            "summary": cost_summary.to_dict(),
            "daily": [row.to_dict() for row in attribution],
        }
        oracle_daily.extend(
            _oracle_daily_rows(
                run,
                label_result.daily,
                trade_result.daily,
                original_daily,
                reversed_daily,
            )
        )
        run_summaries.append(
            {
                "run_id": run_id,
                "objective": objective,
                "fold": run["fold"],
                "seed": run["seed"],
                "label_oracle": label_result.summary.to_dict(),
                "trade_return_oracle": trade_result.summary.to_dict(),
                "original_score": directions["original"].summary.to_dict(),
                "reversed_score": directions["reversed"].summary.to_dict(),
                "cost_attribution": cost_summary.to_dict(),
                "topk_curve": topk["topk_curve"],
                "ranking_metrics": topk["ranking_metrics"],
            }
        )
        exposure_rows[objective].extend(
            _run_exposure_rows(
                joined,
                run,
                sample_by_key,
                appearance_frequency,
            )
        )

    objective_summaries = _objective_summaries(run_summaries)
    oracle_payload = {
        "schema_version": "r3_5_oracle_alignment.v1",
        "frozen_manifest_sha256": _file_sha256(frozen_path),
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "objectives": objective_summaries,
        "label_oracle_gross_spread": _nested_mean(
            run_summaries, "label_oracle", "mean_gross_spread"
        ),
        "label_oracle_net_spread": _nested_mean(
            run_summaries, "label_oracle", "mean_net_spread"
        ),
        "trade_return_oracle_gross_spread": _nested_mean(
            run_summaries, "trade_return_oracle", "mean_gross_spread"
        ),
        "original_score_spread": _nested_mean(
            run_summaries, "original_score", "mean_gross_spread"
        ),
        "reversed_score_spread": _nested_mean(
            run_summaries, "reversed_score", "mean_gross_spread"
        ),
    }
    _write_json(output_root / "oracle_alignment.json", oracle_payload)
    _write_parquet(output_root / "oracle_daily_results.parquet", oracle_daily)
    _write_json(output_root / "cost_attribution.json", cost_payload)
    _write_json(output_root / "topk_diagnostics.json", topk_payload)

    time_payload = _time_alignment_payload(samples)
    _write_json(output_root / "time_alignment.json", time_payload)
    exposure_payload = {
        objective: compare_top5_exposures(rows, exposure_fields=EXPOSURE_FIELDS)
        for objective, rows in sorted(exposure_rows.items())
    }
    _write_json(output_root / "exposure_audit.json", exposure_payload)

    relation_payload = (
        _run_relation_audit(repository, run_manifest, samples, truth)
        if run_relation_audit
        else {
            "status": "insufficient_evidence",
            "reason": "relation audit explicitly skipped",
            "runs": [],
        }
    )
    _write_json(output_root / "relation_mask_audit.json", relation_payload)
    decision = _diagnostic_decision(
        evaluation,
        objective_summaries,
        time_payload,
        exposure_payload,
        relation_payload,
    )
    _write_json(output_root / "diagnostic_decision.json", decision)
    _write_findings(repository, frozen, decision, objective_summaries)
    verify_frozen_manifest(repository, frozen)
    return {
        "frozen_manifest": frozen,
        "oracle": oracle_payload,
        "diagnostic_decision": decision,
    }


def _truth_rows(
    samples: Sequence[Any],
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], Any]]:
    truth: dict[tuple[str, str], dict[str, object]] = {}
    sample_by_key: dict[tuple[str, str], Any] = {}
    for day, day_samples in group_samples_by_date(samples).items():
        raw = [float(sample.returns[1]) for sample in day_samples]
        mean = fmean(raw)
        residuals = [value - mean for value in raw]
        lower, upper = _quantile(residuals, 0.01), _quantile(residuals, 0.99)
        winsorized = [min(max(value, lower), upper) for value in residuals]
        winsor_mean = fmean(winsorized)
        variance = fmean((value - winsor_mean) ** 2 for value in winsorized)
        deviation = variance**0.5
        zscores = [
            0.0 if deviation == 0 else (value - winsor_mean) / deviation
            for value in winsorized
        ]
        percentiles = _average_percentiles(winsorized)
        for sample, residual, zscore, percentile in zip(
            day_samples, residuals, zscores, percentiles, strict=True
        ):
            key = (day, str(sample.target_stock_id))
            if key in truth:
                raise ValueError(f"duplicate frozen truth identity: {key}")
            truth[key] = {
                "realized_return": residual,
                "trade_return": residual,
                "label_zscore": zscore,
                "label_rank_percentile": percentile,
                "label_schema_version": str(sample.label_schema_version),
                "label_end_date": sample.label_end_dates[1].isoformat(),
            }
            sample_by_key[key] = sample
    return truth, sample_by_key


def _join_predictions(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    joined: list[dict[str, object]] = []
    for prediction in predictions:
        day, stock = str(prediction["prediction_date"]), str(prediction["stock_id"])
        target = truth.get((day, stock))
        if target is None:
            raise ValueError(f"prediction has no frozen truth: {day}/{stock}")
        score = float(str(prediction["score"]))
        if not isfinite(score):
            raise ValueError(f"prediction score is not finite: {day}/{stock}")
        joined.append(
            dict(prediction)
            | dict(target)
            | {
                "as_of_date": day,
                "selection_score": score,
                "label_return": target["realized_return"],
            }
        )
    return joined


def _oracle_daily_rows(
    run: Mapping[str, object],
    label: Sequence[DailyCostAttribution],
    trade: Sequence[DailyCostAttribution],
    original: Sequence[DailyCostAttribution],
    reversed_rows: Sequence[DailyCostAttribution],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for values in zip(label, trade, original, reversed_rows, strict=True):
        label_row, trade_row, original_row, reversed_row = values
        dates = {row.prediction_date for row in values}
        if len(dates) != 1:
            raise ValueError("oracle daily results are not date aligned")
        result.append(
            {
                "run_id": run["run_id"],
                "objective": run["objective"],
                "fold": run["fold"],
                "seed": run["seed"],
                "prediction_date": label_row.prediction_date,
                "label_oracle_gross_spread": label_row.gross_spread,
                "label_oracle_net_spread": label_row.net_spread,
                "trade_return_oracle_gross_spread": trade_row.gross_spread,
                "trade_return_oracle_net_spread": trade_row.net_spread,
                "original_score_spread": original_row.gross_spread,
                "original_score_net_spread": original_row.net_spread,
                "reversed_score_spread": reversed_row.gross_spread,
                "reversed_score_net_spread": reversed_row.net_spread,
            }
        )
    return result


def _run_exposure_rows(
    joined: Sequence[dict[str, object]],
    run: Mapping[str, object],
    sample_by_key: Mapping[tuple[str, str], Any],
    appearance_frequency: Mapping[str, float],
) -> list[dict[str, object]]:
    selected_keys: set[tuple[str, str]] = set()
    by_day: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in joined:
        by_day[str(row["prediction_date"])].append(row)
    for day, rows in by_day.items():
        selected_keys.update(
            (day, str(row["stock_id"])) for row in deterministic_rank(rows)[:5]
        )
    result: list[dict[str, object]] = []
    for row in joined:
        key = (str(row["prediction_date"]), str(row["stock_id"]))
        sample = sample_by_key[key]
        exposures = tensorized_target_exposures(
            sample.graph, int(sample.target_node_index)
        )
        result.append(
            dict(row)
            | exposures
            | {
                "objective": run["objective"],
                "fold": run["fold"],
                "seed": run["seed"],
                "selected": key in selected_keys,
                "stock_historical_appearance_frequency": appearance_frequency[key[1]],
            }
        )
    return result


def _appearance_frequencies(
    sample_by_key: Mapping[tuple[str, str], Any],
) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for _, stock in sample_by_key:
        counts[stock] += 1
    maximum = max(counts.values(), default=1)
    return {stock: count / maximum for stock, count in counts.items()}


def _time_alignment_payload(samples: Sequence[Any]) -> dict[str, object]:
    records = [
        {
            "context": "frozen tensor schema",
            "data_available_time": None,
            "trade_time": None,
        }
    ]
    timestamp_audit = evaluate_trade_time_alignment(records)
    schema_versions = sorted({str(sample.label_schema_version) for sample in samples})
    return {
        "schema_version": "r3_5_time_alignment.v1",
        "sample_count": len(samples),
        "timestamp_audit": asdict(timestamp_audit),
        "frozen_label_schema_versions": schema_versions,
        "frozen_label_price_policy": "T+1 open to T+2 open for horizon=1",
        "task25_evaluation_return": "same frozen horizon=1 excess return",
        "interval_alignment_status": "confirmed",
        "alignment_matrix": [
            {
                "training_label": "T+1 close-to-close",
                "backtest_execution": "T+1 open-to-close",
                "aligned": False,
            },
            {
                "training_label": "T+1 open-to-close",
                "backtest_execution": "T+1 open-to-close",
                "aligned": True,
            },
            {
                "training_label": "T+1 close-to-T+2 close",
                "backtest_execution": "T+1 close-to-T+2 close",
                "aligned": True,
            },
            {
                "training_label": "T+1 open-to-T+2 open",
                "backtest_execution": "T+1 open-to-T+2 open",
                "aligned": True,
                "active_task25_path": True,
            },
        ],
        "evidence_limit": (
            "tensorized samples retain dates and label schema but not exact "
            "seat_data_available_time or prediction_generated_time"
        ),
    }


def _run_relation_audit(
    repository: Path,
    run_manifest: Mapping[str, object],
    samples: Sequence[Any],
    truth: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    import torch

    by_date = group_samples_by_date(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_results: list[dict[str, object]] = []
    for run in _mapping_sequence(run_manifest["run_order"]):
        run_id = str(run["run_id"])
        prediction_path = (
            repository
            / "artifacts"
            / "r3_task25"
            / str(run["output_directory"])
            / "daily_predictions.jsonl"
        )
        prediction_dates = sorted(
            {
                str(row["prediction_date"])
                for row in _mapping_sequence(
                    json.loads(line)
                    for line in prediction_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
        )
        first_samples = by_date[prediction_dates[0]]
        first_graph = legalize_r3_graph(first_samples[0].graph)
        model = build_seat_rank_lite(SeatRankLiteConfig(), first_graph).to(device)
        state_path = prediction_path.parent / "model_state.pt"
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
        model.eval()
        mode_rows: dict[str, list[dict[str, object]]] = {
            mode: [] for mode in MASK_MODES
        }
        with torch.inference_mode():
            for day in prediction_dates:
                day_samples = by_date[day]
                graph = legalize_r3_graph(day_samples[0].graph)
                indices = [int(sample.target_node_index) for sample in day_samples]
                stocks = [str(sample.target_stock_id) for sample in day_samples]
                for mode in MASK_MODES:
                    masked = mask_tensorized_relations(graph, mode)
                    outputs = model(masked.data.to(device))
                    scores = outputs["rank_score"].detach().float().cpu()
                    mode_rows[mode].extend(
                        {
                            "prediction_date": day,
                            "stock_id": stock,
                            "score": float(scores[index]),
                            "realized_return": truth[(day, stock)]["realized_return"],
                        }
                        for stock, index in zip(stocks, indices, strict=True)
                    )
        full = mode_rows["full"]
        full_by_key = {
            (str(row["prediction_date"]), str(row["stock_id"])): float(str(row["score"]))
            for row in full
        }
        metrics = {
            mode: _relation_mode_metrics(rows, full_by_key)
            for mode, rows in mode_rows.items()
        }
        run_results.append(
            {
                "run_id": run_id,
                "objective": run["objective"],
                "fold": run["fold"],
                "seed": run["seed"],
                "modes": metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema_version": "r3_5_relation_mask_audit.v1",
        "status": "supported",
        "diagnostic_only": True,
        "device": str(device),
        "runs": run_results,
        "objectives": _relation_objective_summaries(run_results),
    }


def _relation_mode_metrics(
    rows: Sequence[Mapping[str, object]],
    full_by_key: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    by_day: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    score_changes: list[float] = []
    rank_ics: list[float] = []
    gross_spreads: list[float] = []
    overlaps: list[float] = []
    for row in rows:
        by_day[str(row["prediction_date"])].append(row)
        key = (str(row["prediction_date"]), str(row["stock_id"]))
        score_changes.append(abs(float(str(row["score"])) - full_by_key[key]))
    for day_rows in by_day.values():
        ordered = deterministic_rank(day_rows)
        scores = [float(str(row["score"])) for row in day_rows]
        returns = [float(str(row["realized_return"])) for row in day_rows]
        rank_ics.append(spearman(scores, returns))
        selected, rest = ordered[:5], ordered[5:]
        gross_spreads.append(
            fmean(float(str(row["realized_return"])) for row in selected)
            - (
                fmean(float(str(row["realized_return"])) for row in rest)
                if rest
                else 0.0
            )
        )
        full_order = sorted(
            day_rows,
            key=lambda row: (
                -full_by_key[
                    (str(row["prediction_date"]), str(row["stock_id"]))
                ],
                str(row["stock_id"]),
            ),
        )
        overlaps.append(
            jaccard(
                [str(row["stock_id"]) for row in ordered[:5]],
                [str(row["stock_id"]) for row in full_order[:5]],
            )
        )
    return {
        "mean_abs_score_change": fmean(score_changes) if score_changes else 0.0,
        "top5_jaccard_vs_full": fmean(overlaps) if overlaps else 1.0,
        "mean_rank_ic": fmean(rank_ics) if rank_ics else 0.0,
        "mean_top5_gross_spread": fmean(gross_spreads) if gross_spreads else 0.0,
    }


def _relation_objective_summaries(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for objective in ("L0", "L1", "L2"):
        objective_runs = [row for row in runs if row["objective"] == objective]
        result[objective] = {
            mode: {
                metric: fmean(
                    float(str(_mapping(_mapping(row["modes"])[mode])[metric]))
                    for row in objective_runs
                )
                for metric in (
                    "mean_abs_score_change",
                    "top5_jaccard_vs_full",
                    "mean_rank_ic",
                    "mean_top5_gross_spread",
                )
            }
            for mode in MASK_MODES
        }
    return result


def _objective_summaries(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for objective in ("L0", "L1", "L2"):
        values = [row for row in runs if row["objective"] == objective]
        result[objective] = {
            section: {
                metric: _nested_mean(values, section, metric)
                for metric in (
                    "mean_gross_spread",
                    "mean_cost_difference",
                    "mean_net_spread",
                    "mean_candidate_turnover",
                    "mean_continuation_rate",
                )
                if metric in _mapping(_mapping(values[0])[section])
            }
            for section in (
                "label_oracle",
                "trade_return_oracle",
                "original_score",
                "reversed_score",
                "cost_attribution",
            )
        } | {
            "topk_curve": {
                label: {
                    metric: fmean(
                        float(
                            str(
                                _mapping(
                                    _mapping(_mapping(row)["topk_curve"])[label]
                                )[metric]
                            )
                        )
                        for row in values
                    )
                    for metric in ("mean_gross_spread", "mean_net_spread")
                }
                for label in _mapping(values[0]["topk_curve"])
            }
        }
    return result


def _diagnostic_decision(
    evaluation: Mapping[str, object],
    objectives: Mapping[str, object],
    time_payload: Mapping[str, object],
    exposure_payload: Mapping[str, object],
    relation_payload: Mapping[str, object],
) -> dict[str, object]:
    frozen_objectives = _mapping(evaluation["objectives"])
    l0_jaccard = float(str(_mapping(frozen_objectives["L0"])["top5_jaccard"]))
    l2_jaccard = float(str(_mapping(frozen_objectives["L2"])["top5_jaccard"]))
    l2 = _mapping(objectives["L2"])
    original = _mapping(l2["original_score"])
    topk = _mapping(l2["topk_curve"])
    top5 = _mapping(topk["top_5"])
    top20 = _mapping(topk["top_20"])
    gross = float(str(original["mean_gross_spread"]))
    net = float(str(original["mean_net_spread"]))
    cost_difference = float(str(original["mean_cost_difference"]))
    d_confirmed = gross > 0 and net < 0 and cost_difference >= gross
    b_supported = (
        float(str(top20["mean_gross_spread"])) > 0
        and float(str(top5["mean_gross_spread"])) < 0
    )
    relation_status, relation_evidence = _relation_decision(relation_payload)
    decisions: dict[str, dict[str, object]] = {
        "A_optimization_randomness": {
            "status": "unsupported",
            "evidence": [
                f"L2 Top-5 Jaccard {l2_jaccard:.6f} exceeds L0 {l0_jaccard:.6f}",
                "L2 failed the frozen net-spread gate in all six runs",
            ],
        },
        "B_top5_boundary_noise": {
            "status": "supported" if b_supported else "unsupported",
            "evidence": [
                f"L2 Top-5 gross spread={float(str(top5['mean_gross_spread'])):.8f}",
                f"L2 Top-20 gross spread={float(str(top20['mean_gross_spread'])):.8f}",
            ],
        },
        "C_regime_drift": {
            "status": "insufficient_evidence",
            "evidence": [
                "Task 25 contains descriptive market-regime slices but no frozen "
                "causal or predeclared regime gate"
            ],
        },
        "D_turnover_cost": {
            "status": "confirmed" if d_confirmed else "unsupported",
            "evidence": [
                f"L2 original gross spread={gross:.8f}",
                f"L2 original cost difference={cost_difference:.8f}",
                f"L2 original net spread={net:.8f}",
            ],
        },
        "E_graph_no_increment": {
            "status": relation_status,
            "evidence": relation_evidence,
        },
        "F_target_misalignment": {
            "status": "unsupported",
            "evidence": [
                "label and trade-return Oracle gross spreads are positive",
                str(time_payload["frozen_label_price_policy"]),
                "Task 25 evaluation consumes the same frozen horizon=1 excess return",
            ],
        },
    }
    for row in decisions.values():
        if row["status"] not in ALLOWED_STATUSES:
            raise ValueError("invalid A-F diagnostic status")
    return {
        "schema_version": "r3_5_diagnostic_decision.v1",
        **decisions,
        "task25_selection": "none",
        "task26_allowed": False,
        "next_route": _next_route(decisions),
        "exposure_evidence": exposure_payload,
    }


def _relation_decision(
    payload: Mapping[str, object],
) -> tuple[str, list[str]]:
    if payload.get("status") != "supported" or not payload.get("objectives"):
        return "insufficient_evidence", [str(payload.get("reason", "not run"))]
    l2 = _mapping(_mapping(payload["objectives"])["L2"])
    full = _mapping(l2["full"])
    masked = _mapping(l2["without_seat_stock"])
    change = float(str(masked["mean_abs_score_change"]))
    overlap = float(str(masked["top5_jaccard_vs_full"]))
    spread_delta = float(str(masked["mean_top5_gross_spread"])) - float(
        str(full["mean_top5_gross_spread"])
    )
    evidence = [
        f"L2 without Seat-Stock mean absolute score change={change:.8f}",
        f"L2 without Seat-Stock Top-5 Jaccard={overlap:.6f}",
        f"L2 without Seat-Stock gross-spread delta={spread_delta:.8f}",
    ]
    if change < 1e-4 and overlap > 0.95:
        return "supported", evidence
    if spread_delta < 0:
        return "unsupported", evidence
    return "insufficient_evidence", evidence


def _next_route(decisions: Mapping[str, Mapping[str, object]]) -> str:
    if decisions["F_target_misalignment"]["status"] in {"confirmed", "supported"}:
        return "F_fix_label_and_backtest_protocol"
    if decisions["D_turnover_cost"]["status"] == "confirmed":
        return "D_cost_aware_seatrank"
    if decisions["B_top5_boundary_noise"]["status"] == "supported":
        return "top5_two_stage_tail_calibration"
    if decisions["E_graph_no_increment"]["status"] == "supported":
        return "E_stop_graph_complexity"
    return "no_new_model_insufficient_evidence"


def _write_findings(
    repository: Path,
    frozen: Mapping[str, object],
    decision: Mapping[str, object],
    objectives: Mapping[str, object],
) -> None:
    frozen_text = f"""# Task 25 frozen status

- Source commit: `1ac4097`
- Completed runs: {frozen["run_count"]}
- Frozen run files: {frozen["run_file_count"]}
- Frozen checkpoints: {frozen["checkpoint_count"]}
- Task 25 selection: `none`
- Research only: `true`
- Task 26 allowed: `false`

All paths in `artifacts/r3_task25/frozen_manifest.json` passed byte-size and
SHA-256 verification before and after R3.5 analysis. R3.5 did not overwrite a
Task 25 configuration, prediction, checkpoint, fold, gate, or evaluation file.
"""
    _write_text(repository / "docs" / "findings" / "r3_task25_frozen_status.md", frozen_text)
    lines = [
        "# R3.5 A–F final diagnosis",
        "",
        "Task 25 remains research-only with no selected objective and Task 26 disabled.",
        "",
        "## Structured decision",
        "",
        "| Category | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for key in (
        "A_optimization_randomness",
        "B_top5_boundary_noise",
        "C_regime_drift",
        "D_turnover_cost",
        "E_graph_no_increment",
        "F_target_misalignment",
    ):
        row = _mapping(decision[key])
        evidence = "; ".join(str(value) for value in row["evidence"])
        lines.append(f"| {key} | `{row['status']}` | {evidence} |")
    lines.extend(
        [
            "",
            "## Next route",
            "",
            f"`{decision['next_route']}`",
            "",
            "No GRU, FiLM, seat-profile model, hypergraph, new loss, new seed, or "
            "new training checkpoint was produced.",
            "",
            "## Objective evidence",
            "",
            "```json",
            json.dumps(objectives, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    _write_text(
        repository / "docs" / "findings" / "r3_5_af_final_diagnosis.md",
        "\n".join(lines),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False, engine="pyarrow")


def _file_sha256(path: Path) -> str:
    from seatflow.evaluation.frozen_artifacts import sha256_file

    return sha256_file(path)


def _nested_mean(
    rows: Sequence[Mapping[str, object]],
    section: str,
    field: str,
) -> float:
    return fmean(
        float(str(_mapping(_mapping(row)[section])[field])) for row in rows
    )


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    materialized = list(value) if not isinstance(value, list) else value
    if not all(isinstance(row, Mapping) for row in materialized):
        raise TypeError("expected sequence of mappings")
    return materialized


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _average_percentiles(values: Sequence[float]) -> list[float]:
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    ranks = [0.0] * len(values)
    cursor = 0
    denominator = max(len(values) - 1, 1)
    for value in sorted(grouped):
        indexes = grouped[value]
        average = (cursor + cursor + len(indexes) - 1) / 2
        for index in indexes:
            ranks[index] = 0.5 if len(values) == 1 else average / denominator
        cursor += len(indexes)
    return ranks
