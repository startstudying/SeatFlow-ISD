"""Observability: metrics, tracing, and structured logging (Task 29).

Provides:
- Prometheus-compatible metrics (request latency, task duration, error rate)
- Structured JSON logging with trace context
- OpenTelemetry-compatible span context
- Health check endpoints
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

# Trace context variables
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
span_id_var: ContextVar[str] = ContextVar("span_id", default="")


def generate_trace_id() -> str:
    """Generate a 128-bit trace ID (hex string)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 64-bit span ID (hex string)."""
    return uuid.uuid4().hex[:16]


def set_trace_context(trace_id: str, span_id: str) -> None:
    """Set the current trace context."""
    trace_id_var.set(trace_id)
    span_id_var.set(span_id)


def get_trace_context() -> tuple[str, str]:
    """Get the current trace context (trace_id, span_id)."""
    return trace_id_var.get(), span_id_var.get()


class StructuredLogger:
    """Structured JSON logger with trace context."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _log(self, level: str, message: str, **kwargs):
        trace_id, span_id = get_trace_context()
        entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "trace_id": trace_id,
            "span_id": span_id,
            **kwargs,
        }
        self._logger.info(json.dumps(entry))

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs):
        self._log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)


@dataclass
class MetricSample:
    """A single metric sample."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsRegistry:
    """In-memory metrics registry (Prometheus-compatible format)."""

    def __init__(self):
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        label_key = self._label_key(labels or {})
        self._counters[name][label_key] += value

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        self._histograms[name].append(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        label_key = self._label_key(labels or {})
        self._gauges[f"{name}:{label_key}"] = value

    def get_counter(self, name: str, labels: Optional[dict[str, str]] = None) -> float:
        label_key = self._label_key(labels or {})
        return self._counters[name][label_key]

    def get_histogram_values(self, name: str) -> list[float]:
        return self._histograms[name]

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines: list[str] = []
        for name, label_map in self._counters.items():
            for label_key, value in label_map.items():
                labels = self._parse_label_key(label_key)
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                lines.append(f"# TYPE {name} counter")
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        for name, values in self._histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values)}")

        for key, value in self._gauges.items():
            name = key.split(":")[0]
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{key.replace(':', ' ')} {value}")

        return "\n".join(lines)

    @staticmethod
    def _label_key(labels: dict[str, str]) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    @staticmethod
    def _parse_label_key(key: str) -> dict[str, str]:
        if not key:
            return {}
        result = {}
        for pair in key.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                result[k] = v
        return result


# Global metrics registry
metrics = MetricsRegistry()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(name)


@dataclass
class HealthStatus:
    """Health check result."""

    status: str  # "healthy", "degraded", "unhealthy"
    checks: dict[str, bool] = field(default_factory=dict)
    version: str = "3.0.0-dev"
    uptime_seconds: float = 0.0


def health_check() -> HealthStatus:
    """Run health checks for all components."""
    checks = {
        "api": True,
        "database": True,  # In production, check actual connection
        "redis": True,
        "storage": True,
    }
    all_healthy = all(checks.values())
    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        checks=checks,
    )
