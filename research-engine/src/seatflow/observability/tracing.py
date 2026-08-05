"""Tracing utilities for SeatFlow V3 (Task 29)."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from seatflow.observability.metrics import set_trace_context


@dataclass(frozen=True)
class SpanContext:
    """Context for a single span in the trace tree."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


def new_trace() -> SpanContext:
    """Create a new top-level trace context."""
    return SpanContext(
        trace_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex[:16],
    )


@contextmanager
def start_span(
    name: str,
    parent: Optional[SpanContext] = None,
):
    """Start a new span, propagating trace context.

    Args:
        name: span name (operation being performed)
        parent: parent span context (None for root)
    """
    if parent is None:
        ctx = new_trace()
    else:
        ctx = SpanContext(
            trace_id=parent.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id,
        )

    set_trace_context(ctx.trace_id, ctx.span_id)
    start_time = time.time()
    try:
        yield ctx
    finally:
        elapsed = time.time() - start_time
        # In production, emit span to tracing backend
        _emit_span(name, ctx, elapsed)


def _emit_span(name: str, ctx: SpanContext, duration_seconds: float) -> None:
    """Emit a span event. Stub implementation."""
    # In production, this would use OpenTelemetry SDK
    pass


@contextmanager
def traced_operation(name: str, parent: Optional[SpanContext] = None):
    """Convenience wrapper for traced operations."""
    with start_span(name, parent=parent) as ctx:
        yield ctx
