from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from typing import Literal, SupportsInt, cast

from seatflow.backtest.trading_rules import price_limit_pct
from seatflow.data.contracts import DailyBar, normalize_stock_id


class FailClosedBacktestError(ValueError):
    """Raised when a run lacks evidence required for an auditable backtest claim."""

    def __init__(
        self,
        message: str,
        *,
        report: BacktestReport | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.diagnostics = dict(diagnostics or {})


RunMode = Literal["fixture_smoke", "real"]


@dataclass(frozen=True)
class PredictionSignal:
    as_of_date: date
    stock_id: str
    score: Decimal
    available_time: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_id", normalize_stock_id(self.stock_id))

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "stock_id": self.stock_id,
            "score": str(self.score),
            "available_time": self.available_time.isoformat(),
        }


@dataclass(frozen=True)
class OosFold:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class BacktestConfig:
    run_id: str
    mode: RunMode = "fixture_smoke"
    initial_cash: Decimal = Decimal("1000000")
    top_n: int = 5
    minimum_score: Decimal | None = None
    commission_bps: Decimal = Decimal("3")
    stamp_tax_bps: Decimal = Decimal("5")
    slippage_bps: Decimal = Decimal("5")
    max_participation_rate: Decimal = Decimal("0.10")
    min_real_fill_rate: Decimal = Decimal("0.95")
    max_real_untradable_order_rate: Decimal = Decimal("0.05")
    limit_up_pct: Decimal = Decimal("0.10")
    limit_down_pct: Decimal = Decimal("0.10")
    minimum_lot_size: int = 100
    exclude_as_of_limit_up_signals: bool = False
    exclude_as_of_zero_volume_signals: bool = False
    min_as_of_close_price: Decimal | None = None
    excluded_stock_exchanges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.minimum_lot_size <= 0:
            raise ValueError("minimum_lot_size must be positive")
        if self.minimum_score is not None and not self.minimum_score.is_finite():
            raise ValueError("minimum_score must be finite when set")
        for field_name in (
            "commission_bps",
            "stamp_tax_bps",
            "slippage_bps",
            "max_participation_rate",
            "min_real_fill_rate",
            "max_real_untradable_order_rate",
            "limit_up_pct",
            "limit_down_pct",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.min_real_fill_rate > 1:
            raise ValueError("min_real_fill_rate must not exceed 1")
        if self.max_real_untradable_order_rate > 1:
            raise ValueError("max_real_untradable_order_rate must not exceed 1")
        if self.min_as_of_close_price is not None and self.min_as_of_close_price <= 0:
            raise ValueError("min_as_of_close_price must be positive when set")


@dataclass(frozen=True)
class ExecutionOrder:
    as_of_date: date
    execution_date: date
    stock_id: str
    side: Literal["buy", "sell"]
    requested_quantity: int
    reason: str
    target_notional: Decimal | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "execution_date": self.execution_date.isoformat(),
            "stock_id": self.stock_id,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "reason": self.reason,
            "target_notional": (
                str(self.target_notional) if self.target_notional is not None else None
            ),
        }


@dataclass(frozen=True)
class TradeFill:
    order: ExecutionOrder
    price: Decimal
    quantity: int
    commission: Decimal
    stamp_tax: Decimal
    blocked: bool
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order.to_dict(),
            "price": str(self.price),
            "quantity": self.quantity,
            "commission": str(self.commission),
            "stamp_tax": str(self.stamp_tax),
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class BacktestReport:
    run_id: str
    mode: RunMode
    claims: dict[str, bool]
    metrics: dict[str, object]
    sensitivity: dict[str, object]
    orders: tuple[ExecutionOrder, ...]
    fills: tuple[TradeFill, ...]
    audit: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "claims": dict(self.claims),
            "metrics": dict(self.metrics),
            "sensitivity": dict(self.sensitivity),
            "orders": [order.to_dict() for order in self.orders],
            "fills": [fill.to_dict() for fill in self.fills],
            "audit": dict(self.audit),
        }


@dataclass
class _Lot:
    quantity: int
    acquired_date: date


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(
        self,
        *,
        signals: Sequence[PredictionSignal],
        bars: Sequence[DailyBar],
        trading_calendar: Sequence[date],
        universe_by_date: Mapping[date, Sequence[str]],
        benchmark_returns: Mapping[date, Decimal],
        evaluation_manifest_sha256: str | None = None,
        evaluation_manifest_verified: bool = False,
        blind_manifest_sha256: str | None = None,
        blind_manifest_verified: bool = False,
        final_blind_manifest_sha256: str | None = None,
        final_blind_manifest_verified: bool = False,
        oos_folds: Sequence[OosFold] = (),
        confirmed_suspensions: Mapping[date, Sequence[str]] | None = None,
    ) -> BacktestReport:
        calendar = tuple(sorted(set(trading_calendar)))
        bar_map = {(bar.trade_date, bar.stock_id): bar for bar in bars}
        confirmed_suspension_map = _normalize_confirmed_suspensions(confirmed_suspensions)
        self._validate_inputs(
            signals=signals,
            bars=bars,
            calendar=calendar,
            universe_by_date=universe_by_date,
            benchmark_returns=benchmark_returns,
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            evaluation_manifest_verified=evaluation_manifest_verified,
            blind_manifest_sha256=blind_manifest_sha256,
            blind_manifest_verified=blind_manifest_verified,
            final_blind_manifest_sha256=final_blind_manifest_sha256,
            final_blind_manifest_verified=final_blind_manifest_verified,
        )

        cash = self.config.initial_cash
        lots: dict[str, list[_Lot]] = {}
        fills: list[TradeFill] = []
        orders: list[ExecutionOrder] = []
        pending_orders: dict[date, list[ExecutionOrder]] = {}
        nav_by_date: dict[date, Decimal] = {}
        cash_by_date: dict[date, Decimal] = {}
        position_value_by_date: dict[date, dict[str, Decimal]] = {}
        mtm_fallbacks: list[dict[str, object]] = []
        signals_by_date = _group_signals(signals)
        selection_days: list[dict[str, object]] = []
        selected_signal_count = 0
        abstention_day_count = 0
        exposure_day_count = 0

        for current_date in calendar:
            for order in pending_orders.pop(current_date, []):
                fill = self._execute(
                    order,
                    bar_map,
                    lots,
                    cash,
                    confirmed_suspension_map,
                )
                fills.append(fill)
                if fill.blocked:
                    continue
                gross = fill.price * Decimal(fill.quantity)
                if order.side == "buy":
                    total_cost = gross + fill.commission
                    if total_cost > cash:
                        raise FailClosedBacktestError(
                            "execution produced unaffordable buy fill"
                        )
                    cash -= total_cost
                else:
                    cash += gross - fill.commission - fill.stamp_tax

            mtm_value, position_values, fallback_records = _mark_to_market(
                lots,
                bar_map,
                current_date,
                confirmed_suspension_map,
            )
            mtm_fallbacks.extend(fallback_records)
            nav = cash + mtm_value
            daily_signals = signals_by_date.get(current_date, ())
            if daily_signals:
                ranked = self._rank_candidates(
                    current_date,
                    daily_signals,
                    universe_by_date,
                    bar_map,
                )
                selected = self._apply_selection_policy(ranked)
                selected_signal_count += len(selected)
                if not selected:
                    abstention_day_count += 1
                if selected:
                    exposure_day_count += 1
                selection_days.append(
                    {
                        "trade_date": current_date.isoformat(),
                        "eligible_signal_count": len(ranked),
                        "selected_signal_count": len(selected),
                        "abstained": bool(not selected),
                    }
                )
                execution_date = _next_trading_day(calendar, current_date)
                if execution_date is not None:
                    day_orders = self._build_orders(
                        as_of_date=current_date,
                        execution_date=execution_date,
                        selected=selected,
                        lots=lots,
                        position_values=position_values,
                        portfolio_value=nav,
                    )
                    orders.extend(day_orders)
                    pending_orders.setdefault(execution_date, []).extend(day_orders)
            nav_by_date[current_date] = nav
            cash_by_date[current_date] = cash
            position_value_by_date[current_date] = position_values

        metrics = _metrics(nav_by_date, benchmark_returns, oos_folds)
        audit_metrics = _audit_metrics(
            fills=fills,
            nav_by_date=nav_by_date,
            cash_by_date=cash_by_date,
            position_value_by_date=position_value_by_date,
        )
        negative_reasons = _negative_result_reasons(
            mode=self.config.mode,
            evaluation_manifest_sha256=evaluation_manifest_sha256,
            evaluation_manifest_verified=evaluation_manifest_verified,
            legacy_blind_manifest_sha256=blind_manifest_sha256,
            legacy_blind_manifest_verified=blind_manifest_verified,
            final_blind_manifest_sha256=final_blind_manifest_sha256,
            final_blind_manifest_verified=final_blind_manifest_verified,
        )
        execution_quality_failure = None
        if self.config.mode == "real":
            execution_quality_failure = _real_execution_quality_failure(
                audit_metrics=audit_metrics,
                min_fill_rate=self.config.min_real_fill_rate,
                max_untradable_order_rate=self.config.max_real_untradable_order_rate,
            )
        real_backtest_valid = (
            self.config.mode == "real"
            and final_blind_manifest_verified
            and bool(final_blind_manifest_sha256)
            and not negative_reasons
        )
        claims = {
            "engineering_smoke": self.config.mode == "fixture_smoke",
            "real_backtest_valid": real_backtest_valid,
            "negative_result": not real_backtest_valid,
            "uses_next_trading_day_execution": True,
            "enforces_t_plus_one": True,
            "uses_historical_as_of_universe": True,
        }
        audit: dict[str, object] = {
            "evaluation_manifest_sha256": evaluation_manifest_sha256 or blind_manifest_sha256,
            "evaluation_manifest_verified": evaluation_manifest_verified
            or (bool(blind_manifest_sha256) and blind_manifest_verified),
            "legacy_blind_manifest_sha256": blind_manifest_sha256,
            "legacy_blind_manifest_verified": blind_manifest_verified,
            "final_blind_manifest_sha256": final_blind_manifest_sha256,
            "final_blind_manifest_verified": final_blind_manifest_verified,
            "negative_result_reasons": negative_reasons,
            "calendar_start": calendar[0].isoformat() if calendar else None,
            "calendar_end": calendar[-1].isoformat() if calendar else None,
            "selection": {
                "minimum_score": (
                    str(self.config.minimum_score)
                    if self.config.minimum_score is not None
                    else None
                ),
                "threshold_mode": (
                    "abstain_below_minimum_score"
                    if self.config.minimum_score is not None
                    else "legacy_top_n"
                ),
                "selected_signal_count": selected_signal_count,
                "abstention_day_count": abstention_day_count,
                "exposure_day_count": exposure_day_count,
                "day_summaries": selection_days,
            },
            "costs": {
                "commission_bps": str(self.config.commission_bps),
                "stamp_tax_bps": str(self.config.stamp_tax_bps),
                "slippage_bps": str(self.config.slippage_bps),
                "max_participation_rate": str(self.config.max_participation_rate),
                "min_real_fill_rate": str(self.config.min_real_fill_rate),
                "max_real_untradable_order_rate": str(
                    self.config.max_real_untradable_order_rate
                ),
                "exclude_as_of_limit_up_signals": self.config.exclude_as_of_limit_up_signals,
                "exclude_as_of_zero_volume_signals": (
                    self.config.exclude_as_of_zero_volume_signals
                ),
                "min_as_of_close_price": (
                    str(self.config.min_as_of_close_price)
                    if self.config.min_as_of_close_price is not None
                    else None
                ),
                "excluded_stock_exchanges": list(self.config.excluded_stock_exchanges),
            },
            "mark_to_market_fallbacks": mtm_fallbacks,
            "execution": audit_metrics,
            "real_execution_quality": {
                "passed": execution_quality_failure is None,
                "failure_reason": execution_quality_failure,
                "min_real_fill_rate": str(self.config.min_real_fill_rate),
                "max_real_untradable_order_rate": str(
                    self.config.max_real_untradable_order_rate
                ),
            },
        }
        report = BacktestReport(
            run_id=self.config.run_id,
            mode=self.config.mode,
            claims=claims,
            metrics=metrics,
            sensitivity=_sensitivity_summary(metrics),
            orders=tuple(orders),
            fills=tuple(fills),
            audit=audit,
        )
        if execution_quality_failure is not None:
            raise FailClosedBacktestError(
                execution_quality_failure,
                report=report,
                diagnostics={
                    "execution": audit_metrics,
                    "real_execution_quality": audit["real_execution_quality"],
                },
            )
        return report

    def _validate_inputs(
        self,
        *,
        signals: Sequence[PredictionSignal],
        bars: Sequence[DailyBar],
        calendar: Sequence[date],
        universe_by_date: Mapping[date, Sequence[str]],
        benchmark_returns: Mapping[date, Decimal],
        evaluation_manifest_sha256: str | None,
        evaluation_manifest_verified: bool,
        blind_manifest_sha256: str | None,
        blind_manifest_verified: bool,
        final_blind_manifest_sha256: str | None,
        final_blind_manifest_verified: bool,
    ) -> None:
        if not calendar:
            raise FailClosedBacktestError("trading calendar is required")
        if not signals:
            raise FailClosedBacktestError("prediction signals are required")
        if not bars:
            raise FailClosedBacktestError("daily bars are required")
        if not benchmark_returns:
            raise FailClosedBacktestError("benchmark returns are required")
        has_verified_evaluation_manifest = bool(evaluation_manifest_sha256) and (
            evaluation_manifest_verified
        )
        has_verified_legacy_manifest = bool(blind_manifest_sha256) and blind_manifest_verified
        if self.config.mode == "real" and not (
            has_verified_evaluation_manifest or has_verified_legacy_manifest
        ):
            raise FailClosedBacktestError(
                "verified evaluation manifest is required for real backtest"
            )
        if final_blind_manifest_verified and not final_blind_manifest_sha256:
            raise FailClosedBacktestError(
                "final_blind_manifest_verified requires final_blind_manifest_sha256"
            )
        for signal in signals:
            if signal.as_of_date not in universe_by_date:
                raise FailClosedBacktestError(
                    f"historical as-of universe missing for {signal.as_of_date.isoformat()}"
                )
            universe = {normalize_stock_id(item) for item in universe_by_date[signal.as_of_date]}
            if signal.stock_id not in universe:
                raise FailClosedBacktestError(
                    f"signal stock {signal.stock_id} absent from as-of universe"
                )
            if signal.available_time.date() > signal.as_of_date:
                raise FailClosedBacktestError("prediction available_time is after as_of_date")
        if self.config.mode == "real":
            for bar in bars:
                if bar.available_time is None:
                    raise FailClosedBacktestError(
                        "daily bars require available_time for real backtest"
                    )
                if bar.source_version is None:
                    raise FailClosedBacktestError(
                        "daily bars require source_version for real backtest"
                    )
                if bar.pre_close is None:
                    raise FailClosedBacktestError("daily bars require pre_close for limit checks")
                if bar.volume_unit != "shares":
                    raise FailClosedBacktestError(
                        "daily bars require volume_unit='shares' for capacity"
                    )

    def _select(
        self,
        current_date: date,
        signals: Sequence[PredictionSignal],
        universe_by_date: Mapping[date, Sequence[str]],
        bar_map: Mapping[tuple[date, str], DailyBar],
    ) -> tuple[PredictionSignal, ...]:
        ranked = self._rank_candidates(current_date, signals, universe_by_date, bar_map)
        return self._apply_selection_policy(ranked)

    def _rank_candidates(
        self,
        current_date: date,
        signals: Sequence[PredictionSignal],
        universe_by_date: Mapping[date, Sequence[str]],
        bar_map: Mapping[tuple[date, str], DailyBar],
    ) -> tuple[PredictionSignal, ...]:
        universe = {normalize_stock_id(item) for item in universe_by_date[current_date]}
        return tuple(
            sorted(
                (
                    signal
                    for signal in signals
                    if signal.stock_id in universe
                    and self._passes_as_of_tradability_screen(signal, bar_map)
                ),
                key=lambda signal: (-signal.score, signal.stock_id),
            )
        )

    def _apply_selection_policy(
        self, ranked: Sequence[PredictionSignal]
    ) -> tuple[PredictionSignal, ...]:
        if self.config.minimum_score is None:
            return tuple(ranked[: self.config.top_n])
        filtered = tuple(
            signal for signal in ranked if signal.score >= self.config.minimum_score
        )
        return tuple(filtered[: self.config.top_n])

    def _passes_as_of_tradability_screen(
        self,
        signal: PredictionSignal,
        bar_map: Mapping[tuple[date, str], DailyBar],
    ) -> bool:
        if not (
            self.config.exclude_as_of_limit_up_signals
            or self.config.exclude_as_of_zero_volume_signals
            or self.config.min_as_of_close_price is not None
            or bool(self.config.excluded_stock_exchanges)
        ):
            return True
        if signal.stock_id.split(".")[-1] in self.config.excluded_stock_exchanges:
            return False
        bar = bar_map.get((signal.as_of_date, signal.stock_id))
        if bar is None:
            return False
        if (
            self.config.min_as_of_close_price is not None
            and bar.close_price < self.config.min_as_of_close_price
        ):
            return False
        if self.config.exclude_as_of_zero_volume_signals and bar.volume <= 0:
            return False
        if self.config.exclude_as_of_limit_up_signals and _is_close_limit_up(
            bar, self.config
        ):
            return False
        return True

    def _build_orders(
        self,
        *,
        as_of_date: date,
        execution_date: date,
        selected: Sequence[PredictionSignal],
        lots: Mapping[str, Sequence[_Lot]],
        position_values: Mapping[str, Decimal],
        portfolio_value: Decimal,
    ) -> tuple[ExecutionOrder, ...]:
        orders: list[ExecutionOrder] = []
        selected_ids = {signal.stock_id for signal in selected}
        for stock_id, stock_lots in sorted(lots.items()):
            quantity = sum(lot.quantity for lot in stock_lots if lot.quantity > 0)
            if quantity > 0 and stock_id not in selected_ids:
                orders.append(
                    ExecutionOrder(
                        as_of_date,
                        execution_date,
                        stock_id,
                        "sell",
                        quantity,
                        "rebalance",
                    )
                )
        if not selected:
            return tuple(orders)
        target_value = portfolio_value / Decimal(len(selected))
        for signal in selected:
            current_value = position_values.get(signal.stock_id, Decimal("0"))
            delta_value = target_value - current_value
            if delta_value <= 0:
                held_quantity = sum(
                    lot.quantity for lot in lots.get(signal.stock_id, ()) if lot.quantity > 0
                )
                if held_quantity <= 0 or current_value <= 0:
                    continue
                quantity = min(
                    held_quantity,
                    _round_lot(
                        abs(delta_value) / (current_value / Decimal(held_quantity)),
                        self.config.minimum_lot_size,
                    ),
                )
                if quantity > 0:
                    orders.append(
                        ExecutionOrder(
                            as_of_date,
                            execution_date,
                            signal.stock_id,
                            "sell",
                            quantity,
                            "target_weight_rebalance",
                        )
                    )
                continue
            orders.append(
                ExecutionOrder(
                    as_of_date,
                    execution_date,
                    signal.stock_id,
                    "buy",
                    0,
                    "target_weight_rebalance",
                    target_notional=delta_value,
                )
            )
        return tuple(orders)

    def _execute(
        self,
        order: ExecutionOrder,
        bar_map: Mapping[tuple[date, str], DailyBar],
        lots: dict[str, list[_Lot]],
        cash: Decimal,
        confirmed_suspensions: Mapping[date, frozenset[str]],
    ) -> TradeFill:
        bar = bar_map.get((order.execution_date, order.stock_id))
        zero = Decimal("0")
        if bar is None:
            if order.stock_id in confirmed_suspensions.get(order.execution_date, frozenset()):
                return _blocked_fill(order, zero, "confirmed_suspension_no_daily_bar")
            return _blocked_fill(order, zero, "missing_execution_bar")
        if bar.volume <= 0:
            return _blocked_fill(order, bar.open_price, "suspended_or_zero_volume")
        if order.side == "buy":
            if _is_limit_up(bar, self.config):
                return _blocked_fill(order, bar.open_price, "limit_up_buy_blocked")
            price = _buy_price(bar, self.config)
            quantity = _requested_buy_quantity(order, price, self.config)
            if quantity <= 0:
                return _blocked_fill(order, bar.open_price, "zero_lot_after_rounding")
            capacity = _execution_capacity(bar, self.config)
            if quantity > capacity:
                return _blocked_fill(order, bar.open_price, "capacity_exceeded")
            affordable_quantity = _round_lot(
                cash / _buy_cost_per_share(price, self.config),
                self.config.minimum_lot_size,
            )
            quantity = min(quantity, affordable_quantity)
            if quantity <= 0:
                return _blocked_fill(order, bar.open_price, "insufficient_cash_at_execution")
            lots.setdefault(order.stock_id, []).append(
                _Lot(quantity, order.execution_date)
            )
            return _fill(order, price, quantity, self.config, sell=False)
        if order.requested_quantity <= 0:
            return _blocked_fill(order, bar.open_price, "zero_lot_after_rounding")
        capacity = _execution_capacity(bar, self.config)
        if order.requested_quantity > capacity:
            return _blocked_fill(order, bar.open_price, "capacity_exceeded")
        if _is_limit_down(bar, self.config):
            return _blocked_fill(order, bar.open_price, "limit_down_sell_blocked")
        sellable = _sellable_quantity(lots.get(order.stock_id, []), order.execution_date)
        quantity = min(order.requested_quantity, sellable)
        if quantity <= 0:
            return _blocked_fill(order, bar.open_price, "t_plus_one_sell_blocked")
        _remove_lots(lots[order.stock_id], quantity, order.execution_date)
        return _fill(order, _sell_price(bar, self.config), quantity, self.config, sell=True)


def _group_signals(
    signals: Sequence[PredictionSignal],
) -> dict[date, tuple[PredictionSignal, ...]]:
    grouped: dict[date, list[PredictionSignal]] = {}
    for signal in sorted(signals, key=lambda item: (item.as_of_date, item.stock_id)):
        grouped.setdefault(signal.as_of_date, []).append(signal)
    return {key: tuple(value) for key, value in grouped.items()}


def _next_trading_day(calendar: Sequence[date], current: date) -> date | None:
    for item in calendar:
        if item > current:
            return item
    return None


def _round_lot(quantity: Decimal, lot_size: int) -> int:
    lots = (quantity / Decimal(lot_size)).to_integral_value(rounding=ROUND_FLOOR)
    return int(lots) * lot_size


def _requested_buy_quantity(
    order: ExecutionOrder,
    price: Decimal,
    config: BacktestConfig,
) -> int:
    if order.target_notional is None:
        return order.requested_quantity
    # Target notionals are fixed at t; only t+1 execution price determines shares.
    return _round_lot(
        order.target_notional / _buy_cost_per_share(price, config),
        config.minimum_lot_size,
    )


def _buy_cost_per_share(price: Decimal, config: BacktestConfig) -> Decimal:
    return price * (Decimal("1") + config.commission_bps / Decimal("10000"))


def _execution_capacity(bar: DailyBar, config: BacktestConfig) -> int:
    return int(
        (bar.volume * config.max_participation_rate).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )


def _buy_price(bar: DailyBar, config: BacktestConfig) -> Decimal:
    return bar.open_price * (Decimal("1") + config.slippage_bps / Decimal("10000"))


def _sell_price(bar: DailyBar, config: BacktestConfig) -> Decimal:
    return bar.open_price * (Decimal("1") - config.slippage_bps / Decimal("10000"))


def _is_limit_up(bar: DailyBar, config: BacktestConfig) -> bool:
    limit_pct = price_limit_pct(bar.stock_id, config.limit_up_pct)
    return (
        bar.pre_close is not None
        and bar.open_price >= bar.pre_close * (Decimal("1") + limit_pct)
    )


def _is_close_limit_up(bar: DailyBar, config: BacktestConfig) -> bool:
    limit_pct = price_limit_pct(bar.stock_id, config.limit_up_pct)
    return (
        bar.pre_close is not None
        and bar.close_price >= bar.pre_close * (Decimal("1") + limit_pct)
    )


def _is_limit_down(bar: DailyBar, config: BacktestConfig) -> bool:
    limit_pct = price_limit_pct(bar.stock_id, config.limit_down_pct)
    return (
        bar.pre_close is not None
        and bar.open_price <= bar.pre_close * (Decimal("1") - limit_pct)
    )


def _fill(
    order: ExecutionOrder,
    price: Decimal,
    quantity: int,
    config: BacktestConfig,
    *,
    sell: bool,
) -> TradeFill:
    notional = price * Decimal(quantity)
    commission = notional * config.commission_bps / Decimal("10000")
    stamp_tax = notional * config.stamp_tax_bps / Decimal("10000") if sell else Decimal("0")
    return TradeFill(order, price, quantity, commission, stamp_tax, blocked=False)


def _blocked_fill(order: ExecutionOrder, price: Decimal, reason: str) -> TradeFill:
    return TradeFill(
        order,
        price,
        0,
        Decimal("0"),
        Decimal("0"),
        blocked=True,
        blocked_reason=reason,
    )


def _sellable_quantity(lots: Sequence[_Lot], execution_date: date) -> int:
    return sum(lot.quantity for lot in lots if lot.acquired_date < execution_date)


def _remove_lots(lots: list[_Lot], quantity: int, execution_date: date) -> None:
    remaining = quantity
    for lot in lots:
        if remaining <= 0 or lot.acquired_date >= execution_date:
            continue
        sold = min(lot.quantity, remaining)
        lot.quantity -= sold
        remaining -= sold


def _mark_to_market(
    lots: Mapping[str, Sequence[_Lot]],
    bar_map: Mapping[tuple[date, str], DailyBar],
    current_date: date,
    confirmed_suspensions: Mapping[date, frozenset[str]],
) -> tuple[Decimal, dict[str, Decimal], list[dict[str, object]]]:
    value = Decimal("0")
    position_values: dict[str, Decimal] = {}
    fallback_records: list[dict[str, object]] = []
    for stock_id, stock_lots in lots.items():
        quantity = sum(lot.quantity for lot in stock_lots)
        if quantity <= 0:
            continue
        bar = bar_map.get((current_date, stock_id))
        if bar is None:
            if stock_id not in confirmed_suspensions.get(current_date, frozenset()):
                raise FailClosedBacktestError(
                    "held position is missing a current bar without confirmed suspension "
                    f"evidence: {stock_id} on {current_date.isoformat()}"
                )
            fallback = _latest_close_before(bar_map, current_date, stock_id)
            if fallback is None:
                raise FailClosedBacktestError(
                    "confirmed suspension has no prior close for held position: "
                    f"{stock_id} on {current_date.isoformat()}"
                )
            fallback_date, close_price = fallback
            position_value = Decimal(quantity) * close_price
            fallback_records.append(
                {
                    "trade_date": current_date.isoformat(),
                    "stock_id": stock_id,
                    "fallback_trade_date": fallback_date.isoformat(),
                    "fallback_close": str(close_price),
                    "quantity": quantity,
                    "reason": "confirmed_suspension_no_daily_bar_valued_at_previous_close",
                }
            )
        else:
            position_value = Decimal(quantity) * bar.close_price
        position_values[stock_id] = position_value
        value += position_value
    return value, position_values, fallback_records


def _latest_close_before(
    bar_map: Mapping[tuple[date, str], DailyBar],
    current_date: date,
    stock_id: str,
) -> tuple[date, Decimal] | None:
    candidates = [
        (trade_date, bar.close_price)
        for (trade_date, bar_stock_id), bar in bar_map.items()
        if bar_stock_id == stock_id and trade_date < current_date
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def _normalize_confirmed_suspensions(
    confirmed_suspensions: Mapping[date, Sequence[str]] | None,
) -> dict[date, frozenset[str]]:
    """Validate the exact date-stock evidence map used for missing-bar handling."""

    if confirmed_suspensions is None:
        return {}
    normalized: dict[date, frozenset[str]] = {}
    for trade_date, stock_ids in confirmed_suspensions.items():
        if isinstance(trade_date, datetime) or not isinstance(trade_date, date):
            raise ValueError("confirmed suspension trade dates must be dates")
        if isinstance(stock_ids, str):
            raise ValueError("confirmed suspension stock ids must be a sequence")
        normalized[trade_date] = frozenset(normalize_stock_id(stock_id) for stock_id in stock_ids)
    return normalized


def _audit_metrics(
    *,
    fills: Sequence[TradeFill],
    nav_by_date: Mapping[date, Decimal],
    cash_by_date: Mapping[date, Decimal],
    position_value_by_date: Mapping[date, Mapping[str, Decimal]],
) -> dict[str, object]:
    total_orders = len(fills)
    filled = [fill for fill in fills if not fill.blocked and fill.quantity > 0]
    blocked_reasons: dict[str, int] = {}
    for fill in fills:
        if fill.blocked and fill.blocked_reason is not None:
            blocked_reasons[fill.blocked_reason] = blocked_reasons.get(fill.blocked_reason, 0) + 1
    turnover = sum((fill.price * Decimal(fill.quantity) for fill in filled), Decimal("0"))
    last_day = max(nav_by_date) if nav_by_date else None
    if last_day is None:
        gross_exposure = Decimal("0")
        cash = Decimal("0")
        concentration = Decimal("0")
    else:
        positions = position_value_by_date.get(last_day, {})
        gross_exposure = sum(positions.values(), Decimal("0"))
        cash = cash_by_date.get(last_day, Decimal("0"))
        nav = nav_by_date[last_day]
        concentration = max((value / nav for value in positions.values()), default=Decimal("0"))
    fill_rate = Decimal(len(filled)) / Decimal(total_orders) if total_orders else Decimal("0")
    return {
        "order_count": total_orders,
        "filled_order_count": len(filled),
        "blocked_order_count": total_orders - len(filled),
        "untradable_order_rate": str(
            Decimal(total_orders - len(filled)) / Decimal(total_orders)
            if total_orders
            else Decimal("1")
        ),
        "blocked_reasons": blocked_reasons,
        "fill_rate": str(fill_rate),
        "turnover": str(turnover),
        "gross_exposure": str(gross_exposure),
        "cash": str(cash),
        "position_concentration": str(concentration),
    }


def _enforce_real_execution_quality(
    *,
    audit_metrics: Mapping[str, object],
    min_fill_rate: Decimal,
    max_untradable_order_rate: Decimal,
) -> None:
    failure = _real_execution_quality_failure(
        audit_metrics=audit_metrics,
        min_fill_rate=min_fill_rate,
        max_untradable_order_rate=max_untradable_order_rate,
    )
    if failure is not None:
        raise FailClosedBacktestError(failure)


def _real_execution_quality_failure(
    *,
    audit_metrics: Mapping[str, object],
    min_fill_rate: Decimal,
    max_untradable_order_rate: Decimal,
) -> str | None:
    order_count = int(cast(SupportsInt, audit_metrics["order_count"]))
    if order_count <= 0:
        return "real backtest generated no executable orders"
    blocked_reasons = audit_metrics["blocked_reasons"]
    if not isinstance(blocked_reasons, dict):
        return "real backtest execution audit is malformed"
    if blocked_reasons.get("missing_execution_bar", 0) > 0:
        return "real backtest has missing execution bar fills"
    fill_rate = Decimal(str(audit_metrics["fill_rate"]))
    if fill_rate < min_fill_rate:
        return f"real backtest fill_rate {fill_rate} is below required {min_fill_rate}"
    untradable_rate = Decimal(str(audit_metrics["untradable_order_rate"]))
    if untradable_rate > max_untradable_order_rate:
        return (
            "real backtest untradable_order_rate "
            f"{untradable_rate} exceeds allowed {max_untradable_order_rate}"
        )
    return None


def _negative_result_reasons(
    *,
    mode: RunMode,
    evaluation_manifest_sha256: str | None,
    evaluation_manifest_verified: bool,
    legacy_blind_manifest_sha256: str | None,
    legacy_blind_manifest_verified: bool,
    final_blind_manifest_sha256: str | None,
    final_blind_manifest_verified: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if mode != "real":
        reasons.append("fixture_smoke_is_not_real_backtest")
    if not (bool(evaluation_manifest_sha256) and evaluation_manifest_verified):
        if bool(legacy_blind_manifest_sha256) and legacy_blind_manifest_verified:
            reasons.append("legacy_blind_manifest_treated_as_evaluation_manifest_only")
        else:
            reasons.append("missing_verified_evaluation_manifest")
    if not (bool(final_blind_manifest_sha256) and final_blind_manifest_verified):
        reasons.append("missing_verified_final_blind_manifest")
    return tuple(reasons)


def _metrics(
    nav_by_date: Mapping[date, Decimal],
    benchmark_returns: Mapping[date, Decimal],
    oos_folds: Sequence[OosFold],
) -> dict[str, object]:
    if not nav_by_date:
        raise FailClosedBacktestError("no NAV observations")
    ordered = sorted(nav_by_date.items())
    first_nav = ordered[0][1]
    last_nav = ordered[-1][1]
    daily_returns: dict[date, Decimal] = {}
    previous = first_nav
    for day, nav in ordered[1:]:
        daily_returns[day] = (nav / previous) - Decimal("1") if previous != 0 else Decimal("0")
        previous = nav
    bench_total = Decimal("1")
    for day, value in benchmark_returns.items():
        if ordered[0][0] < day <= ordered[-1][0]:
            bench_total *= Decimal("1") + value
    fold_returns = {
        fold.name: _period_return(nav_by_date, fold.start, fold.end) for fold in oos_folds
    }
    worst_fold = min(fold_returns, key=lambda name: fold_returns[name]) if fold_returns else None
    return {
        "total_return": str((last_nav / first_nav) - Decimal("1")),
        "benchmark_total_return": str(bench_total - Decimal("1")),
        "excess_return": str(
            ((last_nav / first_nav) - Decimal("1")) - (bench_total - Decimal("1"))
        ),
        "max_drawdown": str(_max_drawdown(tuple(nav for _, nav in ordered))),
        "daily_return_count": len(daily_returns),
        "worst_oos_fold": worst_fold,
        "oos_fold_returns": {key: str(value) for key, value in fold_returns.items()},
    }


def _period_return(nav_by_date: Mapping[date, Decimal], start: date, end: date) -> Decimal:
    values = [(day, nav) for day, nav in sorted(nav_by_date.items()) if start <= day <= end]
    if len(values) < 2:
        return Decimal("0")
    return values[-1][1] / values[0][1] - Decimal("1")


def _max_drawdown(values: Sequence[Decimal]) -> Decimal:
    peak = values[0]
    worst = Decimal("0")
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - Decimal("1"))
    return worst


def _sensitivity_summary(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        "baseline_total_return": metrics["total_return"],
        "worst_oos_fold": metrics["worst_oos_fold"],
        "required_dimensions": [
            "commission_bps",
            "slippage_bps",
            "max_participation_rate",
            "top_n",
        ],
    }
