"""Temporal Top-K holding analytics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

Row = Mapping[str, object]
Stock = str


def deterministic_rank(
    rows: Sequence[Row],
    *,
    score_field: str,
    stock_field: str,
) -> list[Row]:
    return sorted(rows, key=lambda row: (-_as_float(row[score_field]), str(row[stock_field])))


def cross_seed_same_day_jaccard(
    rows: Sequence[Row],
    *,
    top_k: int,
    date_field: str = "prediction_date",
    seed_field: str = "seed",
    stock_field: str = "stock_id",
    score_field: str = "score",
) -> dict[str, float]:
    """Average pairwise Top-K Jaccard overlap for the same date across seeds."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    by_day_seed = _group_rows_by_seed(
        rows,
        date_field=date_field,
        seed_field=seed_field,
        stock_field=stock_field,
    )
    if not by_day_seed:
        raise ValueError("cross-seed Jaccard requires at least one row")
    per_day: list[float] = []
    for day_seeded in by_day_seed.values():
        seed_ids = sorted(day_seeded)
        if len(seed_ids) < 2:
            raise ValueError("cross-seed Jaccard requires at least two seeds")
        ranked_by_seed = {
            seed: deterministic_rank(
                tuple(rows_for_seed.values()),
                score_field=score_field,
                stock_field=stock_field,
            )
            for seed, rows_for_seed in day_seeded.items()
        }
        stock_sets = {
            tuple(row[stock_field] for row in rows_) for rows_ in ranked_by_seed.values()
        }
        if len({frozenset(stock_set) for stock_set in stock_sets}) != 1:
            raise ValueError("cross-seed comparison requires identical stock universe per date")
        day_overlaps: list[float] = []
        for left_idx, left_seed in enumerate(seed_ids):
            for right_seed in seed_ids[left_idx + 1 :]:
                left = [str(row[stock_field]) for row in ranked_by_seed[left_seed][:top_k]]
                right = [str(row[stock_field]) for row in ranked_by_seed[right_seed][:top_k]]
                day_overlaps.append(jaccard(left, right))
        per_day.extend(day_overlaps)
    return {
        "mean_jaccard": fmean(per_day),
        "pair_count": float(len(per_day)),
        "sample_count": len(per_day),
    }


def analyze_temporal_holdings(
    rows: Sequence[Row],
    *,
    top_k: tuple[int, int] = (5, 10),
    date_field: str = "prediction_date",
    stock_field: str = "stock_id",
    score_field: str = "score",
) -> dict[str, object]:
    """Compute top-5/top-10 turnover and continuity diagnostics day by day."""
    for size in top_k:
        if size <= 0:
            raise ValueError("top_k values must be positive")
    by_day = _group_rows(rows, date_field=date_field)
    days = sorted(by_day)
    if not days:
        return _empty_report(top_k=top_k)
    previous: dict[int, list[Stock]] = {size: [] for size in top_k}
    summaries: dict[int, _TopKAccumulator] = {
        size: _TopKAccumulator() for size in top_k
    }
    spells = _HoldSpellTracker()
    exit_reasons: dict[str, int] = {
        "candidate_exit": 0,
        "rank_change": 0,
        "other_unavailable": 0,
        "untradable": 0,
        "forced_sell": 0,
        "corporate_action": 0,
        "missing_market_data": 0,
    }
    daily_rows: list[dict[str, object]] = []
    for day in days:
        day_rows = by_day[day]
        today_stocks = {str(row[stock_field]) for row in day_rows}
        ordered = deterministic_rank(
            day_rows, score_field=score_field, stock_field=stock_field
        )
        selected_by_size: dict[int, tuple[Stock, ...]] = {
            size: tuple(str(row[stock_field]) for row in ordered[:size])
            for size in top_k
        }
        for size in top_k:
            selected = selected_by_size[size]
            previous_selected = previous[size]
            jaccard_value: float | None = None
            retained = 0
            replaced = 0
            turnovers = {
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "one_way_turnover": 0.0,
            }
            if previous_selected:
                jaccard_value = jaccard(previous_selected, selected)
                summaries[size].top_k_jaccard_sum += jaccard_value
                if not set(previous_selected).intersection(selected):
                    summaries[size].full_turnover_days += 1
                if set(previous_selected) == set(selected):
                    summaries[size].zero_turnover_days += 1
                retained = len(set(previous_selected) & set(selected))
                replaced = len(set(previous_selected) - set(selected))
                summaries[size].retained_count += retained
                summaries[size].replaced_count += replaced
                turnovers = _weight_turnover(previous_selected, selected)
                summaries[size].buy_turnover_sum += turnovers["buy_turnover"]
                summaries[size].sell_turnover_sum += turnovers["sell_turnover"]
                summaries[size].one_way_turnover_sum += turnovers["one_way_turnover"]
                for stock in previous_selected:
                    if stock in selected:
                        continue
                    if stock not in today_stocks:
                        exit_reasons["candidate_exit"] += 1
                    elif stock in today_stocks and stock not in selected:
                        exit_reasons["rank_change"] += 1
                    else:
                        exit_reasons["other_unavailable"] += 1
                summaries[size].comparison_days += 1
            previous[size] = list(selected)
            daily_rows.append(
                {
                    "prediction_date": day,
                    "top_k": size,
                    "top_k_jaccard": jaccard_value,
                    "retained": 0 if previous_selected == [] else retained,
                    "replaced": 0 if previous_selected == [] else replaced,
                    "buy_turnover": (
                        0.0 if previous_selected == [] else turnovers["buy_turnover"]
                    ),
                    "sell_turnover": (
                        0.0 if previous_selected == [] else turnovers["sell_turnover"]
                    ),
                    "one_way_turnover": (
                        0.0 if previous_selected == [] else turnovers["one_way_turnover"]
                    ),
                }
            )
        if 5 in top_k:
            spells.advance(selected_by_size[5])
        elif 10 in top_k:
            spells.advance(selected_by_size[10])
    spells.close_final_spells()
    return {
        "daily": sorted(
            daily_rows,
            key=lambda row: (str(row["prediction_date"]), _as_int(row["top_k"])),
        ),
        "summaries": {
            str(size): summaries[size].to_dict(
                day_count=len(days),
                hold_buckets=(
                    spells.buckets
                    if size == 5
                    else {"1d": 0, "2-5d": 0, "6-10d": 0, "11+d": 0}
                ),
            )
            for size in top_k
        },
        "exit_reasons": {
            "candidate_exit": float(exit_reasons["candidate_exit"]),
            "rank_change": float(exit_reasons["rank_change"]),
            "other_unavailable": float(exit_reasons["other_unavailable"]),
            "untradable": float(exit_reasons["untradable"]),
            "forced_sell": float(exit_reasons["forced_sell"]),
            "corporate_action": float(exit_reasons["corporate_action"]),
            "missing_market_data": float(exit_reasons["missing_market_data"]),
            "total": float(
                exit_reasons["candidate_exit"]
                + exit_reasons["rank_change"]
                + exit_reasons["other_unavailable"]
                + exit_reasons["untradable"]
                + exit_reasons["forced_sell"]
                + exit_reasons["corporate_action"]
                + exit_reasons["missing_market_data"]
            ),
        },
    }


def jaccard(left: Sequence[Stock], right: Sequence[Stock]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _group_rows_by_seed(
    rows: Sequence[Row],
    *,
    date_field: str,
    seed_field: str,
    stock_field: str,
) -> dict[str, dict[str, dict[Stock, Row]]]:
    grouped: dict[str, dict[str, dict[Stock, Row]]] = {}
    for row in rows:
        day_key = str(row[date_field])
        seed_key = str(row[seed_field])
        grouped.setdefault(day_key, {}).setdefault(seed_key, {})[
            str(row[stock_field])
        ] = row
    return grouped


def _group_rows(
    rows: Sequence[Row],
    *,
    date_field: str,
) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(str(row[date_field]), []).append(row)
    return grouped


def _weight_turnover(previous: Sequence[Stock], current: Sequence[Stock]) -> dict[str, float]:
    prev_weight = 1.0 / len(previous) if previous else 0.0
    curr_weight = 1.0 / len(current) if current else 0.0
    previous_weights = {stock: prev_weight for stock in previous}
    current_weights = {stock: curr_weight for stock in current}
    stock_ids = set(previous_weights) | set(current_weights)
    buy = 0.0
    sell = 0.0
    for stock in stock_ids:
        delta = current_weights.get(stock, 0.0) - previous_weights.get(stock, 0.0)
        if delta > 0:
            buy += delta
        elif delta < 0:
            sell += -delta
    return {
        "buy_turnover": buy,
        "sell_turnover": sell,
        "one_way_turnover": 0.5 * (buy + sell),
    }


@dataclass
class _TopKAccumulator:
    retained_count: float = 0.0
    replaced_count: float = 0.0
    comparison_days: int = 0
    top_k_jaccard_sum: float = 0.0
    full_turnover_days: int = 0
    zero_turnover_days: int = 0
    buy_turnover_sum: float = 0.0
    sell_turnover_sum: float = 0.0
    one_way_turnover_sum: float = 0.0

    def to_dict(
        self, *, day_count: int, hold_buckets: dict[str, int]
    ) -> dict[str, float | int | dict[str, int]]:
        mean_top_k_jaccard = (
            self.top_k_jaccard_sum / self.comparison_days if self.comparison_days else 0.0
        )
        return {
            "mean_top_k_jaccard": mean_top_k_jaccard,
            "retained_count": self.retained_count,
            "replaced_count": self.replaced_count,
            "full_turnover_ratio": (
                self.full_turnover_days / day_count if day_count else 0.0
            ),
            "zero_turnover_ratio": (
                self.zero_turnover_days / day_count if day_count else 0.0
            ),
            "buy_turnover": self.buy_turnover_sum / day_count if day_count else 0.0,
            "sell_turnover": self.sell_turnover_sum / day_count if day_count else 0.0,
            "one_way_turnover": self.one_way_turnover_sum / day_count if day_count else 0.0,
            "holding_buckets": hold_buckets,
        }


@dataclass
class _HoldSpellTracker:
    active: dict[Stock, int] = field(default_factory=dict)
    buckets: dict[str, int] = field(
        default_factory=lambda: {"1d": 0, "2-5d": 0, "6-10d": 0, "11+d": 0}
    )

    def advance(self, selected: Sequence[Stock]) -> None:
        selected_set = set(selected)
        next_active: dict[Stock, int] = {}
        for stock in selected:
            next_active[stock] = self.active.get(stock, 0) + 1
        for stock, days in self.active.items():
            if stock not in selected_set:
                self._finalize_stock(stock, days)
        self.active = next_active

    def close_final_spells(self) -> None:
        for stock, days in tuple(self.active.items()):
            self._finalize_stock(stock, days)
        self.active = {}

    def _finalize_stock(self, stock: Stock, days: int) -> None:
        self.buckets[_bucket_name(days)] = self.buckets.get(_bucket_name(days), 0) + 1


def _bucket_name(days: int) -> str:
    if days <= 1:
        return "1d"
    if days <= 5:
        return "2-5d"
    if days <= 10:
        return "6-10d"
    return "11+d"


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _empty_report(top_k: tuple[int, int]) -> dict[str, object]:
    return {
        "daily": [],
        "summaries": {
            str(size): {
                "mean_top_k_jaccard": 0.0,
                "retained_count": 0.0,
                "replaced_count": 0.0,
                "full_turnover_ratio": 0.0,
                "zero_turnover_ratio": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "one_way_turnover": 0.0,
                "holding_buckets": {"1d": 0, "2-5d": 0, "6-10d": 0, "11+d": 0},
            }
            for size in top_k
        },
        "exit_reasons": {
            "candidate_exit": 0.0,
            "rank_change": 0.0,
            "other_unavailable": 0.0,
            "untradable": 0.0,
            "forced_sell": 0.0,
            "corporate_action": 0.0,
            "missing_market_data": 0.0,
            "total": 0.0,
        },
    }
