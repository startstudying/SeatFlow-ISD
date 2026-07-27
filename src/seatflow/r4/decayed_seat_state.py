"""Forward-only decayed seat-state propagation for R4."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date

TradingDate = date | str

MAXIMUM_EVENT_AGE_DAYS = 5
SEAT_SIGNAL_HALF_LIFE_DAYS = 2.0


@dataclass(frozen=True)
class SeatEvent:
    trade_date: TradingDate
    stock_id: str
    net_buy_amount: float
    seat_count: int
    institution_ratio: float
    seat_concentration: float
    seat_signal: float


@dataclass(frozen=True)
class DecayedSeatState:
    trade_date: TradingDate
    stock_id: str
    last_seat_event_date: TradingDate | None
    days_since_last_seat_event: int | None
    last_net_buy_amount: float | None
    last_seat_count: int | None
    last_institution_ratio: float | None
    last_seat_concentration: float | None
    last_seat_signal: float | None
    decayed_seat_signal: float
    has_new_seat_event: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_decayed_seat_states(
    *,
    trading_dates: Sequence[TradingDate],
    events: Iterable[SeatEvent],
    stock_ids: Iterable[str] | None = None,
    half_life_days: float = SEAT_SIGNAL_HALF_LIFE_DAYS,
    maximum_event_age_days: int = MAXIMUM_EVENT_AGE_DAYS,
) -> list[DecayedSeatState]:
    """Build point-in-time decayed seat states over explicit trading dates.

    The function only propagates state forward along the supplied trading
    calendar. Future events are never visible on earlier dates. If ``stock_ids``
    is provided, stocks without a prior event are emitted with empty seat fields
    and zero signal so non-seat features can still be joined downstream.
    """
    _validate_decay_config(
        half_life_days=half_life_days,
        maximum_event_age_days=maximum_event_age_days,
    )
    ordered_dates = _validate_trading_dates(trading_dates)
    date_to_index = {trading_date: index for index, trading_date in enumerate(ordered_dates)}
    events_by_date = _group_events_by_date(events, date_to_index)
    tracked_stock_ids = {str(stock_id) for stock_id in stock_ids or ()}
    latest: dict[str, SeatEvent] = {}
    states: list[DecayedSeatState] = []

    for trading_date in ordered_dates:
        todays_events = events_by_date.get(trading_date, {})
        tracked_stock_ids.update(todays_events)
        latest.update(todays_events)
        for stock_id in sorted(tracked_stock_ids):
            states.append(
                _state_for_stock_date(
                    trading_date=trading_date,
                    stock_id=stock_id,
                    latest=latest.get(stock_id),
                    has_new_event=stock_id in todays_events,
                    date_to_index=date_to_index,
                    half_life_days=half_life_days,
                    maximum_event_age_days=maximum_event_age_days,
                )
            )
    return states


def propagate_decayed_seat_state(
    *,
    events_by_date: Mapping[TradingDate, Sequence[SeatEvent]],
    universe_by_date: Mapping[TradingDate, Sequence[str]],
    trading_calendar: Sequence[TradingDate],
    half_life_days: float = SEAT_SIGNAL_HALF_LIFE_DAYS,
    maximum_event_age_days: int = MAXIMUM_EVENT_AGE_DAYS,
) -> tuple[DecayedSeatState, ...]:
    """Propagate latest seat event state for each day's scoreable universe."""
    events = [
        event
        for trading_date in trading_calendar
        for event in events_by_date.get(trading_date, ())
    ]
    stock_ids_by_date = {
        trading_date: tuple(universe_by_date.get(trading_date, ()))
        for trading_date in trading_calendar
    }
    states = _build_decayed_seat_states_by_daily_universe(
        trading_dates=trading_calendar,
        events=events,
        stock_ids_by_date=stock_ids_by_date,
        half_life_days=half_life_days,
        maximum_event_age_days=maximum_event_age_days,
    )
    return tuple(states)


def half_life_decay(age: int, *, half_life_days: float = SEAT_SIGNAL_HALF_LIFE_DAYS) -> float:
    """Return the half-life decay multiplier ``2 ** (-age / half_life)``."""
    return decay_multiplier(age, half_life_days=half_life_days)


def decay_multiplier(age: int, *, half_life_days: float = SEAT_SIGNAL_HALF_LIFE_DAYS) -> float:
    """Return the half-life decay multiplier ``2 ** (-age / half_life)``."""
    if age < 0:
        raise ValueError("age must be non-negative")
    if half_life_days <= 0.0:
        raise ValueError("half_life_days must be positive")
    return float(2.0 ** (-float(age) / half_life_days))


def states_by_date_stock(
    states: Iterable[DecayedSeatState],
) -> dict[tuple[TradingDate, str], DecayedSeatState]:
    """Index decayed states by ``(trade_date, stock_id)``."""
    return {(state.trade_date, state.stock_id): state for state in states}


def _build_decayed_seat_states_by_daily_universe(
    *,
    trading_dates: Sequence[TradingDate],
    events: Iterable[SeatEvent],
    stock_ids_by_date: Mapping[TradingDate, Sequence[str]],
    half_life_days: float,
    maximum_event_age_days: int,
) -> list[DecayedSeatState]:
    _validate_decay_config(
        half_life_days=half_life_days,
        maximum_event_age_days=maximum_event_age_days,
    )
    ordered_dates = _validate_trading_dates(trading_dates)
    date_to_index = {trading_date: index for index, trading_date in enumerate(ordered_dates)}
    events_by_date = _group_events_by_date(events, date_to_index)
    latest: dict[str, SeatEvent] = {}
    states: list[DecayedSeatState] = []

    for trading_date in ordered_dates:
        todays_events = events_by_date.get(trading_date, {})
        latest.update(todays_events)
        scoreable_stock_ids = sorted(
            {str(stock_id) for stock_id in stock_ids_by_date.get(trading_date, ())}
            | set(todays_events)
        )
        for stock_id in scoreable_stock_ids:
            states.append(
                _state_for_stock_date(
                    trading_date=trading_date,
                    stock_id=stock_id,
                    latest=latest.get(stock_id),
                    has_new_event=stock_id in todays_events,
                    date_to_index=date_to_index,
                    half_life_days=half_life_days,
                    maximum_event_age_days=maximum_event_age_days,
                )
            )
    return states


def _validate_decay_config(
    *,
    half_life_days: float,
    maximum_event_age_days: int,
) -> None:
    if half_life_days != SEAT_SIGNAL_HALF_LIFE_DAYS:
        raise ValueError("R4 v1 fixes seat_signal_half_life_days=2")
    if maximum_event_age_days != MAXIMUM_EVENT_AGE_DAYS:
        raise ValueError("R4 v1 fixes maximum_event_age_days=5")


def _validate_trading_dates(trading_dates: Sequence[TradingDate]) -> tuple[TradingDate, ...]:
    ordered_dates = tuple(trading_dates)
    if not ordered_dates:
        raise ValueError("trading_dates must not be empty")
    if len(set(ordered_dates)) != len(ordered_dates):
        raise ValueError("trading_dates must not contain duplicates")
    if tuple(sorted(ordered_dates)) != ordered_dates:
        raise ValueError("trading_dates must be sorted in ascending order")
    return ordered_dates


def _group_events_by_date(
    events: Iterable[SeatEvent],
    date_to_index: Mapping[TradingDate, int],
) -> dict[TradingDate, dict[str, SeatEvent]]:
    events_by_date: dict[TradingDate, dict[str, SeatEvent]] = {}
    for event in events:
        if event.trade_date not in date_to_index:
            raise ValueError("event trade_date must exist in trading_dates")
        if event.seat_count < 0:
            raise ValueError("seat_count must be non-negative")
        stock_id = str(event.stock_id)
        events_for_date = events_by_date.setdefault(event.trade_date, {})
        if stock_id in events_for_date:
            raise ValueError("duplicate seat event for stock_id and trade_date")
        events_for_date[stock_id] = SeatEvent(
            trade_date=event.trade_date,
            stock_id=stock_id,
            net_buy_amount=float(event.net_buy_amount),
            seat_count=event.seat_count,
            institution_ratio=float(event.institution_ratio),
            seat_concentration=float(event.seat_concentration),
            seat_signal=float(event.seat_signal),
        )
    return events_by_date


def _state_for_stock_date(
    *,
    trading_date: TradingDate,
    stock_id: str,
    latest: SeatEvent | None,
    has_new_event: bool,
    date_to_index: Mapping[TradingDate, int],
    half_life_days: float,
    maximum_event_age_days: int,
) -> DecayedSeatState:
    if latest is None:
        return _empty_state(trading_date, stock_id)

    age = date_to_index[trading_date] - date_to_index[latest.trade_date]
    if age < 0:
        raise ValueError("latest seat event cannot be after trading_date")
    decayed_signal = (
        0.0
        if age > maximum_event_age_days
        else latest.seat_signal * decay_multiplier(age, half_life_days=half_life_days)
    )
    return DecayedSeatState(
        trade_date=trading_date,
        stock_id=stock_id,
        last_seat_event_date=latest.trade_date,
        days_since_last_seat_event=age,
        last_net_buy_amount=latest.net_buy_amount,
        last_seat_count=latest.seat_count,
        last_institution_ratio=latest.institution_ratio,
        last_seat_concentration=latest.seat_concentration,
        last_seat_signal=latest.seat_signal,
        decayed_seat_signal=decayed_signal,
        has_new_seat_event=has_new_event,
    )


def _empty_state(trade_date: TradingDate, stock_id: str) -> DecayedSeatState:
    return DecayedSeatState(
        trade_date=trade_date,
        stock_id=stock_id,
        last_seat_event_date=None,
        days_since_last_seat_event=None,
        last_net_buy_amount=None,
        last_seat_count=None,
        last_institution_ratio=None,
        last_seat_concentration=None,
        last_seat_signal=None,
        decayed_seat_signal=0.0,
        has_new_seat_event=False,
    )
