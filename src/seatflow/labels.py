from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from seatflow.data.contracts import AdjustmentFactorEvidence, DailyBar, normalize_stock_id

FROZEN_ADJUSTMENT_POLICY = "frozen_adjusted_close_v1"
FROZEN_CALENDAR_POLICY = "frozen_trading_calendar_v1"
FIXTURE_ADJUSTMENT_POLICY = "raw_close_fixture_only"
FIXTURE_CALENDAR_POLICY = "observed_trading_days_fixture_only"
RESEARCH_HORIZONS = (1, 3, 5)
LABEL_V2_VERSION = "executable_excess_return_v2"
LABEL_V2_PRICE_POLICY = "next_trading_day_open"
LABEL_V2_ADJUSTMENT_POLICY = "actual_adj_factor_required_v1"
LABEL_V2_DISPOSITIONS = frozenset(
    {
        "eligible",
        "missing_price",
        "missing_adjustment_factor",
        "missing_market_status",
        "suspended",
        "st_restricted",
        "price_limit_blocked",
        "new_stock_restricted",
        "unobserved_horizon",
    }
)


@dataclass(frozen=True)
class LabelingConfig:
    horizons: tuple[int, ...]
    positive_return_threshold: Decimal = Decimal("0")
    adjustment_policy: str = FROZEN_ADJUSTMENT_POLICY
    calendar_policy: str = FROZEN_CALENDAR_POLICY

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("labeling horizons must not be empty")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("labeling horizons must be positive")
        if self.adjustment_policy not in {FROZEN_ADJUSTMENT_POLICY, FIXTURE_ADJUSTMENT_POLICY}:
            raise ValueError(
                "adjustment_policy must be frozen_adjusted_close_v1 for research "
                "or raw_close_fixture_only for fixtures"
            )
        if self.calendar_policy not in {FROZEN_CALENDAR_POLICY, FIXTURE_CALENDAR_POLICY}:
            raise ValueError(
                "calendar_policy must be frozen_trading_calendar_v1 for research "
                "or observed_trading_days_fixture_only for fixtures"
            )

    @property
    def research_ready(self) -> bool:
        return (
            self.adjustment_policy == FROZEN_ADJUSTMENT_POLICY
            and self.calendar_policy == FROZEN_CALENDAR_POLICY
            and tuple(sorted(set(self.horizons))) == RESEARCH_HORIZONS
        )

    def require_research_ready(self) -> None:
        if tuple(sorted(set(self.horizons))) != RESEARCH_HORIZONS:
            raise ValueError("research labeling requires 1/3/5 trading-day horizons")
        if self.adjustment_policy != FROZEN_ADJUSTMENT_POLICY:
            raise ValueError("research labeling requires frozen adjustment policy")
        if self.calendar_policy != FROZEN_CALENDAR_POLICY:
            raise ValueError("research labeling requires frozen trading calendar")


@dataclass(frozen=True)
class StockLabel:
    as_of_date: date
    cutoff_time: datetime
    target_stock_id: str
    horizon: int
    label_start_date: date
    label_end_date: date
    base_close: Decimal
    future_close: Decimal
    future_return: Decimal
    direction_label: int
    label_available_time: datetime
    adjustment_policy: str
    calendar_policy: str
    base_source_name: str
    future_source_name: str
    base_source_version: str | None = None
    future_source_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "cutoff_time": self.cutoff_time.isoformat(),
            "target_stock_id": self.target_stock_id,
            "horizon": self.horizon,
            "label_start_date": self.label_start_date.isoformat(),
            "label_end_date": self.label_end_date.isoformat(),
            "base_close": str(self.base_close),
            "future_close": str(self.future_close),
            "future_return": str(self.future_return),
            "direction_label": self.direction_label,
            "label_available_time": self.label_available_time.isoformat(),
            "adjustment_policy": self.adjustment_policy,
            "calendar_policy": self.calendar_policy,
            "base_source_name": self.base_source_name,
            "future_source_name": self.future_source_name,
            "base_source_version": self.base_source_version,
            "future_source_version": self.future_source_version,
        }


@dataclass(frozen=True)
class CandidateResidualLabel:
    as_of_date: date
    target_stock_id: str
    horizon: int
    stock_return: Decimal
    candidate_pool_return: Decimal
    residual_return: Decimal
    direction_label: int


@dataclass(frozen=True)
class CrossSectionalRankingLabel:
    as_of_date: date
    target_stock_id: str
    horizon: int
    residual_return: Decimal
    winsorized_residual: Decimal
    residual_zscore: float
    residual_rank_percentile: float
    direction_target: int


def build_cross_sectional_ranking_labels(
    labels: Sequence[CandidateResidualLabel],
) -> tuple[CrossSectionalRankingLabel, ...]:
    """Create date-local R3 targets without fitting cross-date statistics."""

    grouped: dict[tuple[date, int], list[CandidateResidualLabel]] = defaultdict(list)
    for label in labels:
        grouped[(label.as_of_date, label.horizon)].append(label)
    result: list[CrossSectionalRankingLabel] = []
    for _, day_labels in sorted(grouped.items()):
        ordered = sorted(day_labels, key=lambda item: item.target_stock_id)
        raw = [float(item.residual_return) for item in ordered]
        lower, upper = _linear_quantile(raw, 0.01), _linear_quantile(raw, 0.99)
        winsorized = [min(max(value, lower), upper) for value in raw]
        mean = sum(winsorized) / len(winsorized)
        variance = sum((value - mean) ** 2 for value in winsorized) / len(winsorized)
        standard_deviation = variance**0.5
        ranks = _average_numeric_ranks(winsorized)
        denominator = max(len(winsorized) - 1, 1)
        for item, value, rank in zip(ordered, winsorized, ranks, strict=True):
            result.append(
                CrossSectionalRankingLabel(
                    as_of_date=item.as_of_date,
                    target_stock_id=item.target_stock_id,
                    horizon=item.horizon,
                    residual_return=item.residual_return,
                    winsorized_residual=Decimal(str(value)),
                    residual_zscore=(
                        0.0 if standard_deviation == 0 else (value - mean) / standard_deviation
                    ),
                    residual_rank_percentile=(
                        0.5 if len(winsorized) == 1 else rank / denominator
                    ),
                    direction_target=int(item.residual_return > 0),
                )
            )
    return tuple(result)


def build_candidate_pool_residual_labels(
    labels: Sequence[StockLabel],
    *,
    horizon: int = 1,
    neutral_band: Decimal = Decimal("0"),
) -> tuple[CandidateResidualLabel, ...]:
    """Subtract each date's equal-weight candidate return from every stock return."""

    if horizon <= 0:
        raise ValueError("residual label horizon must be positive")
    if neutral_band < 0:
        raise ValueError("residual label neutral band must be non-negative")
    grouped: dict[date, list[StockLabel]] = defaultdict(list)
    for label in labels:
        if label.horizon == horizon:
            grouped[label.as_of_date].append(label)
    residuals: list[CandidateResidualLabel] = []
    for as_of_date, day_labels in sorted(grouped.items()):
        stock_ids = [label.target_stock_id for label in day_labels]
        if len(stock_ids) != len(set(stock_ids)):
            raise ValueError("candidate residual labels require one stock per date")
        pool_return = sum(
            (label.future_return for label in day_labels), start=Decimal("0")
        ) / Decimal(len(day_labels))
        for label in sorted(day_labels, key=lambda item: item.target_stock_id):
            residual = label.future_return - pool_return
            residuals.append(
                CandidateResidualLabel(
                    as_of_date=as_of_date,
                    target_stock_id=label.target_stock_id,
                    horizon=horizon,
                    stock_return=label.future_return,
                    candidate_pool_return=pool_return,
                    residual_return=residual,
                    direction_label=_three_way_direction(residual, neutral_band),
                )
            )
    return tuple(residuals)


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _average_numeric_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + end - 1) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = average
        start = end
    return ranks


@dataclass(frozen=True)
class ExecutablePriceEvidence:
    """A price and actual adjustment factor used by the V2 executable label."""

    trade_date: date
    price: Decimal
    adjustment_factor: Decimal | None
    available_time: datetime
    source_hash: str

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("executable label price must be positive")
        if self.adjustment_factor is not None and self.adjustment_factor <= 0:
            raise ValueError("adjustment factor must be positive when present")
        if not self.source_hash or len(self.source_hash) != 64:
            raise ValueError("price evidence requires a SHA-256 source hash")


def executable_open_evidence_from_daily_bar(
    bar: DailyBar,
    adjustment: AdjustmentFactorEvidence,
) -> ExecutablePriceEvidence:
    """Bind a real next-open price to its actual, source-hashed adjustment factor."""

    if bar.stock_id != adjustment.stock_id or bar.trade_date != adjustment.trade_date:
        raise ValueError("daily bar and adjustment factor must describe the same stock and date")
    if bar.available_time is None:
        raise ValueError("daily bar requires available_time for executable label evidence")
    payload = {
        "daily_bar": bar.to_dict(),
        "adjustment_factor": adjustment.to_dict(),
    }
    return ExecutablePriceEvidence(
        trade_date=bar.trade_date,
        price=bar.open_price,
        adjustment_factor=adjustment.adjustment_factor,
        available_time=bar.available_time,
        source_hash=sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


@dataclass(frozen=True)
class MarketDispositionEvidence:
    """Auditable reason why an otherwise scheduled label cannot be executed."""

    disposition: str
    source_hash: str

    def __post_init__(self) -> None:
        if self.disposition not in LABEL_V2_DISPOSITIONS - {"eligible"}:
            raise ValueError("unsupported non-eligible label disposition")
        if not self.source_hash or len(self.source_hash) != 64:
            raise ValueError("disposition evidence requires a SHA-256 source hash")


@dataclass(frozen=True)
class StockLabelV2:
    """Executable, benchmark-relative label with explicit provenance and disposition."""

    as_of_date: date
    cutoff_time: datetime
    target_stock_id: str
    horizon: int
    label_start_date: date
    label_end_date: date
    benchmark_id: str
    price_policy: str
    adjustment_policy: str
    calendar_policy: str
    label_available_time: datetime
    disposition: str
    source_hash: str
    stock_return: Decimal | None
    benchmark_return: Decimal | None
    excess_return: Decimal | None
    direction_label: int | None

    def __post_init__(self) -> None:
        if self.disposition not in LABEL_V2_DISPOSITIONS:
            raise ValueError("unsupported V2 label disposition")
        if self.price_policy != LABEL_V2_PRICE_POLICY:
            raise ValueError("unsupported V2 label price policy")
        if self.adjustment_policy != LABEL_V2_ADJUSTMENT_POLICY:
            raise ValueError("unsupported V2 adjustment policy")
        if not self.source_hash or len(self.source_hash) != 64:
            raise ValueError("V2 label requires a SHA-256 source hash")
        if self.disposition == "eligible":
            if None in (
                self.stock_return,
                self.benchmark_return,
                self.excess_return,
                self.direction_label,
            ):
                raise ValueError("eligible V2 labels require return and direction values")
            if self.direction_label not in {-1, 0, 1}:
                raise ValueError("V2 direction label must be -1, 0, or 1")
        elif any(
            value is not None
            for value in (
                self.stock_return,
                self.benchmark_return,
                self.excess_return,
                self.direction_label,
            )
        ):
            raise ValueError("ineligible V2 labels must not carry numeric targets")

    def to_dict(self) -> dict[str, object]:
        return {
            "label_version": LABEL_V2_VERSION,
            "as_of_date": self.as_of_date.isoformat(),
            "cutoff_time": self.cutoff_time.isoformat(),
            "target_stock_id": self.target_stock_id,
            "horizon": self.horizon,
            "label_start_date": self.label_start_date.isoformat(),
            "label_end_date": self.label_end_date.isoformat(),
            "benchmark_id": self.benchmark_id,
            "price_policy": self.price_policy,
            "adjustment_policy": self.adjustment_policy,
            "calendar_policy": self.calendar_policy,
            "label_available_time": self.label_available_time.isoformat(),
            "disposition": self.disposition,
            "source_hash": self.source_hash,
            "stock_return": None if self.stock_return is None else str(self.stock_return),
            "benchmark_return": (
                None if self.benchmark_return is None else str(self.benchmark_return)
            ),
            "excess_return": None if self.excess_return is None else str(self.excess_return),
            "direction_label": self.direction_label,
        }


def load_labeling_config(path: str | Path) -> LabelingConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    labeling = raw["labeling"]
    return LabelingConfig(
        horizons=tuple(int(item) for item in labeling["horizons"]),
        positive_return_threshold=Decimal(str(labeling.get("positive_return_threshold", "0"))),
        adjustment_policy=str(labeling.get("adjustment_policy", "raw_close_fixture_only")),
        calendar_policy=str(labeling.get("calendar_policy", "observed_trading_days_fixture_only")),
    )


def generate_future_return_labels(
    bars: Iterable[DailyBar],
    as_of_date: date,
    cutoff_time: datetime,
    target_stock_ids: Sequence[str],
    horizons: Sequence[int],
    positive_return_threshold: Decimal = Decimal("0"),
    adjustment_policy: str = FROZEN_ADJUSTMENT_POLICY,
    calendar_policy: str = FROZEN_CALENDAR_POLICY,
    trading_calendar: Sequence[date] | None = None,
) -> tuple[StockLabel, ...]:
    """Generate t+1-based future labels over explicit trading-day horizons."""

    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    _validate_policies(adjustment_policy, calendar_policy)

    by_stock = _group_bars(bars)
    calendar = _resolve_calendar(
        by_stock.values(),
        calendar_policy=calendar_policy,
        trading_calendar=trading_calendar,
    )
    labels: list[StockLabel] = []
    for raw_stock_id in target_stock_ids:
        stock_id = normalize_stock_id(raw_stock_id)
        ordered = by_stock.get(stock_id)
        if not ordered:
            raise ValueError(f"no bars found for target_stock_id={stock_id}")
        as_of_index = _index_for_date(ordered, as_of_date, stock_id)
        base_bar = ordered[as_of_index]
        _require_available_at_cutoff(base_bar, cutoff_time)
        if base_bar.close_price <= 0:
            raise ValueError(f"base close must be positive for {stock_id} on {as_of_date}")
        as_of_calendar_index = _calendar_index(calendar, as_of_date)

        for horizon in sorted(set(horizons)):
            future_calendar_index = as_of_calendar_index + horizon
            if future_calendar_index >= len(calendar):
                continue
            label_start_date = calendar[as_of_calendar_index + 1]
            label_end_date = calendar[future_calendar_index]
            future_bar = _bar_for_date(ordered, label_end_date, stock_id)
            if future_bar is None:
                continue
            if future_bar.trade_date <= as_of_date:
                raise ValueError("label window must start after as_of_date")
            if future_bar.available_time is None:
                raise ValueError("future label bars require available_time")
            _require_compatible_timezone(cutoff_time, future_bar.available_time)
            if future_bar.available_time <= cutoff_time:
                raise ValueError("future label availability must be after cutoff_time")
            future_return = (future_bar.close_price - base_bar.close_price) / base_bar.close_price
            labels.append(
                StockLabel(
                    as_of_date=as_of_date,
                    cutoff_time=cutoff_time,
                    target_stock_id=stock_id,
                    horizon=horizon,
                    label_start_date=label_start_date,
                    label_end_date=future_bar.trade_date,
                    base_close=base_bar.close_price,
                    future_close=future_bar.close_price,
                    future_return=future_return,
                    direction_label=1 if future_return > positive_return_threshold else 0,
                    label_available_time=future_bar.available_time,
                    adjustment_policy=adjustment_policy,
                    calendar_policy=calendar_policy,
                    base_source_name=base_bar.source_name,
                    future_source_name=future_bar.source_name,
                    base_source_version=base_bar.source_version,
                    future_source_version=future_bar.source_version,
                )
            )
    return tuple(labels)


def generate_executable_excess_return_labels_v2(
    *,
    as_of_date: date,
    cutoff_time: datetime,
    target_stock_ids: Sequence[str],
    horizons: Sequence[int],
    trading_calendar: Sequence[date],
    benchmark_by_stock: Mapping[str, str],
    stock_open_prices: Mapping[tuple[str, date], ExecutablePriceEvidence],
    benchmark_open_prices: Mapping[tuple[str, date], ExecutablePriceEvidence],
    dispositions: Mapping[tuple[str, date], MarketDispositionEvidence] | None = None,
    market_status_hashes: Mapping[str, str] | None = None,
    neutral_band: Decimal = Decimal("0"),
) -> tuple[StockLabelV2, ...]:
    """Create explicit T+1-open, benchmark-relative V2 labels.

    This function never manufactures an adjusted price. Missing prices or
    adjustment factors become a recorded non-eligible disposition. The caller
    supplies the frozen calendar and source-hashed market evidence.
    """

    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("V2 horizons must be positive")
    if neutral_band < 0:
        raise ValueError("V2 neutral band must be non-negative")
    calendar = tuple(sorted(set(trading_calendar)))
    dispositions = {} if dispositions is None else dispositions
    market_status_hashes = {} if market_status_hashes is None else market_status_hashes
    as_of_index = _calendar_index(calendar, as_of_date)
    labels: list[StockLabelV2] = []
    for raw_stock_id in target_stock_ids:
        stock_id = normalize_stock_id(raw_stock_id)
        benchmark_id = benchmark_by_stock.get(stock_id)
        if not benchmark_id:
            raise ValueError(f"V2 label requires benchmark for {stock_id}")
        market_status_hash = market_status_hashes.get(stock_id)
        if market_status_hash is not None and len(market_status_hash) != 64:
            raise ValueError("V2 market-status evidence must be a SHA-256 hash")
        for horizon in sorted(set(horizons)):
            start_index = as_of_index + 1
            end_index = start_index + horizon
            if end_index >= len(calendar):
                continue
            start_date = calendar[start_index]
            end_date = calendar[end_index]
            disposition = _v2_disposition(stock_id, start_date, end_date, dispositions)
            stock_start = stock_open_prices.get((stock_id, start_date))
            stock_end = stock_open_prices.get((stock_id, end_date))
            benchmark_start = benchmark_open_prices.get((benchmark_id, start_date))
            benchmark_end = benchmark_open_prices.get((benchmark_id, end_date))
            evidence = (stock_start, stock_end, benchmark_start, benchmark_end)
            if disposition is None:
                disposition = _missing_v2_disposition(evidence)
            hashes = [item.source_hash for item in evidence if item is not None]
            if market_status_hash is not None:
                hashes.append(market_status_hash)
            if disposition is not None:
                status_evidence = _v2_disposition_evidence(
                    stock_id, start_date, end_date, dispositions
                )
                if status_evidence is not None:
                    hashes.append(status_evidence.source_hash)
                labels.append(
                    _ineligible_v2_label(
                        as_of_date, cutoff_time, stock_id, horizon, start_date, end_date,
                        benchmark_id, disposition, hashes
                    )
                )
                continue
            assert all(item is not None and item.adjustment_factor is not None for item in evidence)
            typed_evidence = tuple(item for item in evidence if item is not None)
            available_time = max(item.available_time for item in typed_evidence)
            _require_compatible_timezone(cutoff_time, available_time)
            if available_time <= cutoff_time:
                raise ValueError("V2 label evidence must be available after cutoff_time")
            stock_return = _adjusted_return(stock_start, stock_end)
            benchmark_return = _adjusted_return(benchmark_start, benchmark_end)
            excess_return = stock_return - benchmark_return
            labels.append(
                StockLabelV2(
                    as_of_date=as_of_date,
                    cutoff_time=cutoff_time,
                    target_stock_id=stock_id,
                    horizon=horizon,
                    label_start_date=start_date,
                    label_end_date=end_date,
                    benchmark_id=benchmark_id,
                    price_policy=LABEL_V2_PRICE_POLICY,
                    adjustment_policy=LABEL_V2_ADJUSTMENT_POLICY,
                    calendar_policy=FROZEN_CALENDAR_POLICY,
                    label_available_time=available_time,
                    disposition="eligible",
                    source_hash=_v2_hash(hashes),
                    stock_return=stock_return,
                    benchmark_return=benchmark_return,
                    excess_return=excess_return,
                    direction_label=_three_way_direction(excess_return, neutral_band),
                )
            )
    return tuple(labels)


def _v2_disposition(
    stock_id: str,
    start_date: date,
    end_date: date,
    dispositions: Mapping[tuple[str, date], MarketDispositionEvidence],
) -> str | None:
    for trade_date in (start_date, end_date):
        evidence = dispositions.get((stock_id, trade_date))
        if evidence is not None:
            return evidence.disposition
    return None


def _v2_disposition_evidence(
    stock_id: str,
    start_date: date,
    end_date: date,
    dispositions: Mapping[tuple[str, date], MarketDispositionEvidence],
) -> MarketDispositionEvidence | None:
    for trade_date in (start_date, end_date):
        evidence = dispositions.get((stock_id, trade_date))
        if evidence is not None:
            return evidence
    return None


def _missing_v2_disposition(
    evidence: Sequence[ExecutablePriceEvidence | None],
) -> str | None:
    if any(item is None for item in evidence):
        return "missing_price"
    if any(item.adjustment_factor is None for item in evidence if item is not None):
        return "missing_adjustment_factor"
    return None


def _ineligible_v2_label(
    as_of_date: date,
    cutoff_time: datetime,
    stock_id: str,
    horizon: int,
    start_date: date,
    end_date: date,
    benchmark_id: str,
    disposition: str,
    hashes: Sequence[str],
) -> StockLabelV2:
    return StockLabelV2(
        as_of_date=as_of_date,
        cutoff_time=cutoff_time,
        target_stock_id=stock_id,
        horizon=horizon,
        label_start_date=start_date,
        label_end_date=end_date,
        benchmark_id=benchmark_id,
        price_policy=LABEL_V2_PRICE_POLICY,
        adjustment_policy=LABEL_V2_ADJUSTMENT_POLICY,
        calendar_policy=FROZEN_CALENDAR_POLICY,
        label_available_time=cutoff_time,
        disposition=disposition,
        source_hash=_v2_hash(hashes),
        stock_return=None,
        benchmark_return=None,
        excess_return=None,
        direction_label=None,
    )


def _adjusted_return(
    start: ExecutablePriceEvidence | None,
    end: ExecutablePriceEvidence | None,
) -> Decimal:
    if (
        start is None
        or end is None
        or start.adjustment_factor is None
        or end.adjustment_factor is None
    ):
        raise ValueError("V2 adjusted return requires price and adjustment evidence")
    adjusted_start = start.price * start.adjustment_factor
    adjusted_end = end.price * end.adjustment_factor
    return (adjusted_end - adjusted_start) / adjusted_start


def _three_way_direction(excess_return: Decimal, neutral_band: Decimal) -> int:
    if excess_return > neutral_band:
        return 1
    if excess_return < -neutral_band:
        return -1
    return 0


def _v2_hash(hashes: Sequence[str]) -> str:
    if not hashes:
        return sha256(b"missing-v2-evidence").hexdigest()
    return sha256("|".join(sorted(hashes)).encode("utf-8")).hexdigest()


def label_distribution(labels: Iterable[StockLabel]) -> dict[str, int]:
    counts: dict[str, int] = {"0": 0, "1": 0}
    for label in labels:
        counts[str(label.direction_label)] = counts.get(str(label.direction_label), 0) + 1
    return counts


def _group_bars(bars: Iterable[DailyBar]) -> Mapping[str, tuple[DailyBar, ...]]:
    grouped: dict[str, list[DailyBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.stock_id].append(bar)
    return {
        stock_id: tuple(sorted(items, key=lambda item: item.trade_date))
        for stock_id, items in grouped.items()
    }


def _index_for_date(bars: tuple[DailyBar, ...], as_of_date: date, stock_id: str) -> int:
    for index, bar in enumerate(bars):
        if bar.trade_date == as_of_date:
            return index
    raise ValueError(f"no as_of bar for target_stock_id={stock_id} on {as_of_date}")


def _require_available_at_cutoff(bar: DailyBar, cutoff_time: datetime) -> None:
    if bar.available_time is None:
        raise ValueError("feature bars require available_time")
    _require_compatible_timezone(cutoff_time, bar.available_time)
    if bar.available_time > cutoff_time:
        raise ValueError("feature bar available_time is after cutoff_time")


def _require_compatible_timezone(left: datetime, right: datetime) -> None:
    if (left.tzinfo is None) != (right.tzinfo is None):
        raise ValueError("datetime values must use compatible timezone awareness")


def _validate_policies(adjustment_policy: str, calendar_policy: str) -> None:
    LabelingConfig(
        horizons=(1,),
        adjustment_policy=adjustment_policy,
        calendar_policy=calendar_policy,
    )


def _resolve_calendar(
    grouped_bars: Iterable[tuple[DailyBar, ...]],
    *,
    calendar_policy: str,
    trading_calendar: Sequence[date] | None,
) -> tuple[date, ...]:
    if calendar_policy == FROZEN_CALENDAR_POLICY:
        if trading_calendar is None:
            raise ValueError("frozen trading calendar is required for research labels")
        calendar = tuple(sorted(set(trading_calendar)))
    else:
        observed = {bar.trade_date for bars in grouped_bars for bar in bars}
        calendar = tuple(sorted(observed))
    if not calendar:
        raise ValueError("trading calendar must not be empty")
    return calendar


def _calendar_index(calendar: tuple[date, ...], as_of_date: date) -> int:
    try:
        return calendar.index(as_of_date)
    except ValueError as error:
        message = f"as_of_date {as_of_date.isoformat()} is not in trading calendar"
        raise ValueError(message) from error


def _bar_for_date(
    bars: tuple[DailyBar, ...], trade_date: date, stock_id: str
) -> DailyBar | None:
    for bar in bars:
        if bar.trade_date == trade_date:
            return bar
    return None
