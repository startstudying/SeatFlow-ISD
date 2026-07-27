"""Structural turnover floor imposed by candidate-universe exits."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean, median


@dataclass(frozen=True)
class TurnoverDecomposition:
    """Mutually exclusive one-way turnover components for one decision day."""

    candidate_exit_turnover: float
    voluntary_rank_turnover: float
    untradable_turnover: float
    corporate_action_turnover: float

    @property
    def total_turnover(self) -> float:
        return (
            self.candidate_exit_turnover
            + self.voluntary_rank_turnover
            + self.untradable_turnover
            + self.corporate_action_turnover
        )

    def validate(self, observed_total_turnover: float, *, tolerance: float = 1e-12) -> None:
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("turnover components must be non-negative")
        if abs(self.total_turnover - observed_total_turnover) > tolerance:
            raise ValueError(
                "turnover decomposition identity does not equal observed total turnover"
            )


@dataclass(frozen=True)
class DailyTurnoverFloor:
    run_id: str
    prediction_date: str
    previous_prediction_date: str
    portfolio_size: int
    previous_holding_count: int
    current_candidate_count: int
    candidate_exit_count: int
    candidate_exit_turnover_floor: float
    total_turnover: float
    removable_turnover_above_floor: float
    candidate_pool_temporal_jaccard: float
    candidate_pool_retention_rate: float
    has_previous_holdings: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_candidate_turnover_floor(
    *,
    daily_candidates: Mapping[str, Collection[str]],
    daily_holdings: Mapping[str, Sequence[str]],
    portfolio_size: int = 5,
    run_id: str = "default",
) -> tuple[list[DailyTurnoverFloor], dict[str, float | int | None]]:
    """Measure Rule A's minimum turnover using complete daily pools and holdings.

    ``daily_holdings[date]`` is the portfolio selected on ``date``.  For decision
    date ``t``, the Rule A floor compares holdings selected on the preceding
    available trading date with the full candidate pool on ``t``.
    """
    if portfolio_size <= 0:
        raise ValueError("portfolio_size must be positive")
    candidate_dates = set(daily_candidates)
    holding_dates = set(daily_holdings)
    if candidate_dates != holding_dates:
        missing_candidates = sorted(holding_dates - candidate_dates)
        missing_holdings = sorted(candidate_dates - holding_dates)
        raise ValueError(
            "daily evidence dates must match; "
            f"missing_candidates={missing_candidates}, missing_holdings={missing_holdings}"
        )
    dates = sorted(candidate_dates)
    daily: list[DailyTurnoverFloor] = []
    for previous_date, prediction_date in zip(dates, dates[1:], strict=False):
        previous_holdings = _unique_stock_set(daily_holdings[previous_date])
        current_holdings = _unique_stock_set(daily_holdings[prediction_date])
        current_candidates = _unique_stock_set(daily_candidates[prediction_date])
        previous_candidates = _unique_stock_set(daily_candidates[previous_date])
        if len(previous_holdings) > portfolio_size or len(current_holdings) > portfolio_size:
            raise ValueError("daily holdings exceed portfolio_size")
        if not current_holdings.issubset(current_candidates):
            raise ValueError("current holdings must be selected from current candidates")
        candidate_exit_count = len(previous_holdings - current_candidates)
        candidate_exit_floor = candidate_exit_count / portfolio_size
        total_turnover = len(previous_holdings - current_holdings) / portfolio_size
        if total_turnover + 1e-12 < candidate_exit_floor:
            raise ValueError("observed turnover cannot be below the Rule A floor")
        daily.append(
            DailyTurnoverFloor(
                run_id=run_id,
                prediction_date=prediction_date,
                previous_prediction_date=previous_date,
                portfolio_size=portfolio_size,
                previous_holding_count=len(previous_holdings),
                current_candidate_count=len(current_candidates),
                candidate_exit_count=candidate_exit_count,
                candidate_exit_turnover_floor=candidate_exit_floor,
                total_turnover=total_turnover,
                removable_turnover_above_floor=total_turnover - candidate_exit_floor,
                candidate_pool_temporal_jaccard=_jaccard(
                    previous_candidates, current_candidates
                ),
                candidate_pool_retention_rate=_retention_rate(
                    previous_candidates, current_candidates
                ),
            )
        )
    return daily, summarize_turnover_floor(daily)


def compute_candidate_exit_turnover_floor(
    rows: Sequence[Mapping[str, object]],
    *,
    portfolio_size: int = 5,
) -> list[DailyTurnoverFloor]:
    """Compute floors from row-oriented frozen replay evidence."""
    if portfolio_size <= 0:
        raise ValueError("portfolio_size must be positive")
    required = {
        "run_id",
        "prediction_date",
        "holdings",
        "current_candidates",
        "total_turnover",
    }
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        grouped.setdefault(str(row["run_id"]), []).append(row)
    result: list[DailyTurnoverFloor] = []
    for run_id, run_rows in sorted(grouped.items()):
        previous_holdings: set[str] | None = None
        previous_candidates: set[str] | None = None
        previous_date: str | None = None
        for row in sorted(run_rows, key=lambda item: str(item["prediction_date"])):
            prediction_date = str(row["prediction_date"])
            holdings = _as_stock_set(row["holdings"], field="holdings")
            candidates = _as_stock_set(
                row["current_candidates"], field="current_candidates"
            )
            if len(holdings) > portfolio_size:
                raise ValueError("daily holdings exceed portfolio_size")
            if not holdings.issubset(candidates):
                raise ValueError("current holdings must be selected from current candidates")
            total_turnover = _as_non_negative_float(
                row["total_turnover"], field="total_turnover"
            )
            has_previous = previous_holdings is not None
            candidate_exit_count = (
                len(previous_holdings - candidates)
                if previous_holdings is not None
                else 0
            )
            floor = candidate_exit_count / portfolio_size
            if has_previous and total_turnover + 1e-12 < floor:
                raise ValueError("observed turnover cannot be below the Rule A floor")
            result.append(
                DailyTurnoverFloor(
                    run_id=run_id,
                    prediction_date=prediction_date,
                    previous_prediction_date=previous_date or "",
                    portfolio_size=portfolio_size,
                    previous_holding_count=(
                        len(previous_holdings) if previous_holdings is not None else 0
                    ),
                    current_candidate_count=len(candidates),
                    candidate_exit_count=candidate_exit_count,
                    candidate_exit_turnover_floor=floor,
                    total_turnover=total_turnover,
                    removable_turnover_above_floor=(
                        total_turnover - floor if has_previous else 0.0
                    ),
                    candidate_pool_temporal_jaccard=(
                        _jaccard(previous_candidates, candidates)
                        if previous_candidates is not None
                        else 1.0
                    ),
                    candidate_pool_retention_rate=(
                        _retention_rate(previous_candidates, candidates)
                        if previous_candidates is not None
                        else 1.0
                    ),
                    has_previous_holdings=has_previous,
                )
            )
            previous_holdings = holdings
            previous_candidates = candidates
            previous_date = prediction_date
    return result


def summarize_turnover_floor(
    daily: Sequence[DailyTurnoverFloor],
) -> dict[str, float | int | None]:
    comparison_rows = [row for row in daily if row.has_previous_holdings]
    if not comparison_rows:
        return {
            "comparison_day_count": 0,
            "mean_candidate_exit_turnover_floor": None,
            "median_candidate_exit_turnover_floor": None,
            "full_candidate_exit_day_ratio": None,
            "candidate_pool_temporal_jaccard": None,
            "candidate_pool_retention_rate": None,
            "mean_total_turnover": None,
            "maximum_possible_reduction_under_rule_A": None,
        }
    floors = [row.candidate_exit_turnover_floor for row in comparison_rows]
    total_turnovers = [row.total_turnover for row in comparison_rows]
    mean_floor = fmean(floors)
    mean_total = fmean(total_turnovers)
    maximum_reduction = (
        (mean_total - mean_floor) / mean_total if mean_total > 0.0 else 0.0
    )
    return {
        "comparison_day_count": len(comparison_rows),
        "mean_candidate_exit_turnover_floor": mean_floor,
        "median_candidate_exit_turnover_floor": median(floors),
        "full_candidate_exit_day_ratio": fmean(
            1.0 if value >= 1.0 else 0.0 for value in floors
        ),
        "candidate_pool_temporal_jaccard": fmean(
            row.candidate_pool_temporal_jaccard for row in comparison_rows
        ),
        "candidate_pool_retention_rate": fmean(
            row.candidate_pool_retention_rate for row in comparison_rows
        ),
        "mean_total_turnover": mean_total,
        "maximum_possible_reduction_under_rule_A": maximum_reduction,
    }


def _unique_stock_set(stocks: Collection[str]) -> set[str]:
    return {str(stock) for stock in stocks}


def _as_stock_set(value: object, *, field: str) -> set[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ValueError(f"{field} must be a stock collection")
    return {str(stock) for stock in value}


def _as_non_negative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _retention_rate(previous: set[str], current: set[str]) -> float:
    return 1.0 if not previous else len(previous & current) / len(previous)
