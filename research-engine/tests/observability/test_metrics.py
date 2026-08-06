"""Tests for observability (Task 29)."""

import json
import pytest

from seatflow.observability.metrics import (
    MetricsRegistry,
    StructuredLogger,
    health_check,
    set_trace_context,
    get_trace_context,
    generate_trace_id,
)
from seatflow.observability.tracing import new_trace, start_span, traced_operation


class TestStructuredLogging:
    def test_log_contains_trace_id(self):
        """Every log entry must contain trace_id and span_id."""
        set_trace_context("test-trace-123", "test-span-456")
        logger = StructuredLogger("test")
        # Capture log output
        records: list[str] = []
        handler = logger._logger.handlers[0]
        original_emit = handler.emit
        handler.emit = lambda record: records.append(record.getMessage())
        try:
            logger.info("test message", extra_field="value")
        finally:
            handler.emit = original_emit

        assert len(records) == 1
        entry = json.loads(records[0])
        assert entry["trace_id"] == "test-trace-123"
        assert entry["span_id"] == "test-span-456"
        assert entry["message"] == "test message"
        assert entry["level"] == "INFO"
        assert entry["extra_field"] == "value"

    def test_log_levels(self):
        logger = StructuredLogger("test-levels")
        records: list[str] = []
        handler = logger._logger.handlers[0]
        handler.emit = lambda record: records.append(record.getMessage())

        logger.info("info msg")
        logger.warn("warn msg")
        logger.error("error msg")

        assert len(records) == 3
        levels = [json.loads(r)["level"] for r in records]
        assert levels == ["INFO", "WARN", "ERROR"]


class TestMetricsRegistry:
    def test_counter_increment(self):
        reg = MetricsRegistry()
        reg.increment("requests_total")
        assert reg.get_counter("requests_total") == 1.0

        reg.increment("requests_total", value=5)
        assert reg.get_counter("requests_total") == 6.0

    def test_counter_with_labels(self):
        reg = MetricsRegistry()
        reg.increment("http_requests", labels={"method": "GET", "status": "200"})
        reg.increment("http_requests", labels={"method": "GET", "status": "200"})
        reg.increment("http_requests", labels={"method": "POST", "status": "201"})

        assert reg.get_counter("http_requests", labels={"method": "GET", "status": "200"}) == 2.0
        assert reg.get_counter("http_requests", labels={"method": "POST", "status": "201"}) == 1.0

    def test_histogram_observe(self):
        reg = MetricsRegistry()
        reg.observe("request_duration_seconds", 0.1)
        reg.observe("request_duration_seconds", 0.2)
        reg.observe("request_duration_seconds", 0.3)

        values = reg.get_histogram_values("request_duration_seconds")
        assert len(values) == 3
        assert sum(values) == pytest.approx(0.6)

    def test_gauge_set(self):
        reg = MetricsRegistry()
        reg.set_gauge("active_connections", 42)
        reg.set_gauge("active_connections", 50)
        # Gauge should reflect latest value
        assert "active_connections:" in str(reg._gauges)

    def test_prometheus_export(self):
        reg = MetricsRegistry()
        reg.increment("requests_total")
        reg.observe("latency_seconds", 0.05)

        export = reg.export_prometheus()
        assert "requests_total" in export
        assert "latency_seconds_count" in export
        assert "latency_seconds_sum" in export


class TestTracing:
    def test_new_trace_generates_ids(self):
        ctx = new_trace()
        assert len(ctx.trace_id) == 32  # 128-bit hex
        assert len(ctx.span_id) == 16   # 64-bit hex
        assert ctx.parent_span_id is None

    def test_span_context_propagation(self):
        with start_span("operation-1") as ctx1:
            assert ctx1.trace_id is not None
            assert ctx1.span_id is not None

            with start_span("operation-2", parent=ctx1) as ctx2:
                # Child span should share trace_id
                assert ctx2.trace_id == ctx1.trace_id
                assert ctx2.span_id != ctx1.span_id
                assert ctx2.parent_span_id == ctx1.span_id

    def test_traced_operation(self):
        with traced_operation("test-op") as ctx:
            trace_id, span_id = get_trace_context()
            assert trace_id == ctx.trace_id
            assert span_id == ctx.span_id


class TestHealthCheck:
    def test_health_check_returns_status(self):
        result = health_check()
        assert result.status in ("healthy", "degraded", "unhealthy")
        assert "api" in result.checks
        assert "database" in result.checks
        assert result.version

    def test_trace_id_generation(self):
        id1 = generate_trace_id()
        id2 = generate_trace_id()
        assert id1 != id2
        assert len(id1) == 32
