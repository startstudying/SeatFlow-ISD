"""Time domain types for SeatFlow V3.

All timestamps are timezone-aware UTC. The distinction between
``event_time`` (when a fact occurred) and ``available_time`` (when the
system could first observe it) is the core anti-leakage mechanism.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Return the current UTC datetime with timezone info."""
    return datetime.now(timezone.utc)


def assert_aware(dt: datetime, field_name: str = "timestamp") -> None:
    """Raise if *dt* is naive (lacks timezone info)."""
    if dt.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware, got naive datetime")


def validate_available_after_event(
    event_time: datetime,
    available_time: datetime,
    *,
    allow_pre_announced: bool = False,
) -> None:
    """Validate the available_time/event_time ordering.

    By default ``available_time`` must be at or after ``event_time``.
    When *allow_pre_announced* is ``True`` (e.g. for scheduled corporate
    actions) the check is relaxed.
    """
    assert_aware(event_time, "event_time")
    assert_aware(available_time, "available_time")
    if not allow_pre_announced and available_time < event_time:
        raise ValueError(
            f"available_time ({available_time}) must be >= event_time ({event_time}) "
            f"unless the provider allows pre-announced records"
        )
