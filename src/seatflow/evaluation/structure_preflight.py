"""R4 structure-only gate before any economic training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import fmean

from seatflow.evaluation.turnover_floor import DailyTurnoverFloor

STRUCTURE_GATE_THRESHOLD = 0.30


@dataclass(frozen=True)
class StructurePreflightResult:
    status: str
    evidence_complete: bool
    persistent_candidate_pool_size: float | None
    candidate_pool_temporal_jaccard: float | None
    candidate_exit_turnover_floor: float | None
    simulated_rule_b_turnover: float | None
    average_retained_holdings: float | None
    full_turnover_day_ratio: float | None
    turnover_reduction_vs_p0: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_structure_preflight(
    *,
    p0_mean_turnover: float | None,
    turnover_floor_rows: Sequence[DailyTurnoverFloor],
    persistent_pool_sizes: Sequence[int],
    persistent_pool_jaccards: Sequence[float],
    simulated_rule_b_turnovers: Sequence[float],
    retained_holding_counts: Sequence[int],
    portfolio_size: int = 5,
) -> StructurePreflightResult:
    """Evaluate the 30% structural turnover gate using only structure evidence."""

    sequence_lengths = {
        len(turnover_floor_rows),
        len(persistent_pool_sizes),
        len(persistent_pool_jaccards),
        len(simulated_rule_b_turnovers),
        len(retained_holding_counts),
    }
    evidence_complete = (
        p0_mean_turnover is not None
        and p0_mean_turnover > 0.0
        and bool(turnover_floor_rows)
        and bool(persistent_pool_sizes)
        and bool(persistent_pool_jaccards)
        and bool(simulated_rule_b_turnovers)
        and bool(retained_holding_counts)
        and len(sequence_lengths) == 1
    )
    if not evidence_complete:
        return StructurePreflightResult(
            status="insufficient_evidence",
            evidence_complete=False,
            persistent_candidate_pool_size=None,
            candidate_pool_temporal_jaccard=None,
            candidate_exit_turnover_floor=None,
            simulated_rule_b_turnover=None,
            average_retained_holdings=None,
            full_turnover_day_ratio=None,
            turnover_reduction_vs_p0=None,
        )
    assert p0_mean_turnover is not None
    mean_rule_b_turnover = fmean(simulated_rule_b_turnovers)
    turnover_reduction = (p0_mean_turnover - mean_rule_b_turnover) / p0_mean_turnover
    return StructurePreflightResult(
        status="passed" if turnover_reduction >= STRUCTURE_GATE_THRESHOLD else "failed",
        evidence_complete=True,
        persistent_candidate_pool_size=fmean(persistent_pool_sizes),
        candidate_pool_temporal_jaccard=fmean(persistent_pool_jaccards),
        candidate_exit_turnover_floor=fmean(
            row.candidate_exit_turnover_floor for row in turnover_floor_rows
        ),
        simulated_rule_b_turnover=mean_rule_b_turnover,
        average_retained_holdings=fmean(retained_holding_counts),
        full_turnover_day_ratio=fmean(
            1.0 if turnover >= 1.0 else 0.0 for turnover in simulated_rule_b_turnovers
        ),
        turnover_reduction_vs_p0=turnover_reduction,
    )


def structure_gate_decision(result: StructurePreflightResult) -> dict[str, object]:
    passed = (
        result.evidence_complete
        and result.turnover_reduction_vs_p0 is not None
        and result.turnover_reduction_vs_p0 >= STRUCTURE_GATE_THRESHOLD
    )
    return {
        "gate_name": "persistent_universe_structure_gate",
        "threshold": STRUCTURE_GATE_THRESHOLD,
        "observed": result.turnover_reduction_vs_p0,
        "passed": passed,
        "evidence_complete": result.evidence_complete,
        "stop_reason": None if passed else result.status,
        "task59_allowed": passed,
        "task60_allowed": False,
        "task61_allowed": False,
    }


def fail_closed_structure_gate(*, stop_reason: str = "insufficient_evidence") -> dict[str, object]:
    result = StructurePreflightResult(
        status=stop_reason,
        evidence_complete=False,
        persistent_candidate_pool_size=None,
        candidate_pool_temporal_jaccard=None,
        candidate_exit_turnover_floor=None,
        simulated_rule_b_turnover=None,
        average_retained_holdings=None,
        full_turnover_day_ratio=None,
        turnover_reduction_vs_p0=None,
    )
    return structure_gate_decision(result)
