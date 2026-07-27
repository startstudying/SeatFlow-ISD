"""R4 daily scoring eligibility for persistent candidates and holdings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

ALLOWED_UNSCOREABLE_REASONS = frozenset(
    {
        "long_suspended",
        "delisted",
        "critical_market_data_missing",
        "preregistered_risk_exclusion",
    }
)


@dataclass(frozen=True)
class ScoreableCandidate:
    trade_date: str
    stock_id: str
    is_current_event_candidate: bool
    is_recent_event_candidate: bool
    is_existing_holding: bool
    scoreable: bool
    unscoreable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_scoreable_daily_pool(
    *,
    trade_date: str,
    persistent_candidates: Sequence[str],
    previous_holdings: Sequence[str],
    current_event_candidates: Sequence[str],
    recent_event_candidates: Sequence[str],
    market_features: Mapping[str, Mapping[str, object]],
    unscoreable_reasons: Mapping[str, str] | None = None,
) -> tuple[ScoreableCandidate, ...]:
    """Guarantee previous holdings are represented unless explicitly unscoreable."""

    reasons = unscoreable_reasons or {}
    stock_ids = sorted(set(persistent_candidates) | set(previous_holdings))
    current = set(current_event_candidates)
    recent = set(recent_event_candidates)
    holdings = set(previous_holdings)
    rows: list[ScoreableCandidate] = []
    for stock_id in stock_ids:
        reason = reasons.get(stock_id)
        if reason is not None and reason not in ALLOWED_UNSCOREABLE_REASONS:
            raise ValueError(f"unregistered unscoreable reason for {stock_id}: {reason}")
        has_market_features = stock_id in market_features
        if not has_market_features and reason is None:
            raise ValueError(f"critical market data missing requires reason for {stock_id}")
        scoreable = has_market_features and reason is None
        rows.append(
            ScoreableCandidate(
                trade_date=trade_date,
                stock_id=stock_id,
                is_current_event_candidate=stock_id in current,
                is_recent_event_candidate=stock_id in recent,
                is_existing_holding=stock_id in holdings,
                scoreable=scoreable,
                unscoreable_reason=reason,
            )
        )
    return tuple(rows)

