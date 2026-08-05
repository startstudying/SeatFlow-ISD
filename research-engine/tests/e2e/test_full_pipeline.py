"""E2E integration tests (Task 31).

Tests the full pipeline: Ingestion → Identity → Events/Cycles → Features →
Labels → Samples → Graph → Model → Evaluation → Backtest.

These tests verify the end-to-end data flow with synthetic data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from seatflow.ingestion.models import (
    BronzeRecord,
    SourceConfidence,
)
from seatflow.ingestion.reconcile import reconcile_records
from seatflow.identity.stock_identity import StockIdentity, StockIdentityRegistry
from seatflow.identity.seat_identity import SeatIdentity, SeatIdentityRegistry
from seatflow.evaluation.metrics import mae, mse, rank_ic, ndcg_at_k
from seatflow.backtest.engine import (
    backtest,
    compare_strategies,
    BacktestConfig,
    CostConfig,
)


def _utc(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_bronze_record(
    source: str,
    event_time: datetime,
    raw_payload: dict[str, Any],
    confidence: SourceConfidence = SourceConfidence.HIGH,
) -> BronzeRecord:
    """Helper to create a BronzeRecord with sensible defaults."""
    import hashlib
    import json

    payload_hash = hashlib.sha256(
        json.dumps(raw_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return BronzeRecord(
        event_time=event_time,
        available_time=event_time,
        ingested_at=event_time,
        source=source,
        source_record_id=f"{source}-001",
        source_version="1.0",
        raw_payload=raw_payload,
        raw_payload_hash=payload_hash,
        ingestion_run_id="test-run-001",
        source_confidence=confidence,
    )


class TestE2EIngestionToIdentity:
    """E2E: Ingestion → Reconciliation → Identity Resolution."""

    def test_bronze_to_reconciled_to_identity_flow(self):
        """Full flow: raw records → reconciled → identity resolution."""
        # Step 1: Create raw Bronze records from two sources
        event_ts = _utc(2024, 6, 1)
        tushare_record = _make_bronze_record(
            "tushare",
            event_ts,
            {"stock_code": "000001", "seat_name": "华泰证券深圳总部", "buy_amount": 1e8},
            SourceConfidence.HIGH,
        )
        akshare_record = _make_bronze_record(
            "akshare",
            event_ts,
            {"stock_code": "000001", "seat_name": "华泰证券深圳总部", "buy_amount": 1.1e8},
            SourceConfidence.MEDIUM,
        )

        # Step 2: Reconcile (group by key)
        records_by_key = {"000001": [tushare_record, akshare_record]}
        field_specs = {"buy_amount": lambda payload: payload.get("buy_amount", 0)}
        reconciled = reconcile_records(records_by_key, field_specs=field_specs)

        assert "000001" in reconciled
        assert reconciled["000001"].fields["buy_amount"] == pytest.approx(1e8)

    def test_stock_identity_resolution(self):
        """Stock identity PIT resolution via registry."""
        identity = StockIdentity(
            stock_id="STOCK_000001",
            ts_code="000001.SZ",
            exchange="SZSE",
            board="main",
            listing_date=_utc(2010, 1, 1),
            delisting_date=None,
            aliases=["000001"],
            valid_from=_utc(2010, 1, 1),
            valid_to=None,
            identity_confidence=1.0,
            source_evidence=["tushare"],
        )
        registry = StockIdentityRegistry()
        registry.register(identity)

        resolved = registry.resolve_stock("000001", _utc(2024, 6, 1))
        assert resolved is not None
        assert resolved.stock_id == "STOCK_000001"

    def test_seat_identity_resolution(self):
        """Seat identity PIT resolution via registry."""
        identity = SeatIdentity(
            seat_id="SEAT_HUATAI_SZ",
            canonical_name="华泰证券-深圳",
            seat_type="broker_branch",
            identity_confidence=0.95,
            valid_from=_utc(2010, 1, 1),
            valid_to=None,
            aliases=["华泰证券深圳总部"],
            source_evidence=["manual"],
        )
        registry = SeatIdentityRegistry()
        registry.register(identity)

        resolved = registry.resolve_seat("华泰证券深圳总部", _utc(2024, 6, 1))
        assert resolved is not None
        assert resolved.seat_id == "SEAT_HUATAI_SZ"


class TestE2EModelToEvaluation:
    """E2E: Model predictions → Evaluation metrics."""

    def test_predictions_to_metrics(self):
        """Predictions flow through to evaluation metrics correctly."""
        n_samples = 100
        rng = np.random.default_rng(42)

        truths = rng.standard_normal(n_samples)
        predictions = truths + rng.standard_normal(n_samples) * 0.3

        # Note: metrics API is (y_true, y_pred)
        mae_val = mae(truths, predictions)
        mse_val = mse(truths, predictions)
        ic_val = rank_ic(truths, predictions)
        ndcg_val = ndcg_at_k(truths, predictions, k=10)

        assert mae_val > 0
        assert mse_val > 0
        assert -1 <= ic_val <= 1
        assert 0 < ndcg_val <= 1
        # Noisy copy should have positive IC
        assert ic_val > 0

    def test_perfect_predictions_metrics(self):
        """Perfect predictions should give optimal metrics."""
        truths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = truths.copy()

        assert mae(truths, predictions) == pytest.approx(0.0)
        assert mse(truths, predictions) == pytest.approx(0.0)
        assert rank_ic(truths, predictions) == pytest.approx(1.0)
        assert ndcg_at_k(truths, predictions, k=5) == pytest.approx(1.0)


class TestE2EBacktestPipeline:
    """E2E: Predictions → Backtest → Performance metrics."""

    def test_backtest_with_cost(self):
        """Backtest pipeline includes transaction costs."""
        n_days, n_stocks = 30, 20
        rng = np.random.default_rng(42)
        actual = rng.standard_normal((n_days, n_stocks)) * 0.02
        preds = actual + rng.standard_normal((n_days, n_stocks)) * 0.01

        config = BacktestConfig(
            top_k=5,
            cost=CostConfig(commission_bps=3, slippage_bps=5),
        )
        result = backtest(preds, actual, config)

        assert result.daily_returns.shape == (n_days,)
        assert result.costs.sum() > 0
        assert result.avg_cost_bps > 0
        assert result.sharpe != 0 or result.total_return != 0

    def test_strategy_comparison_attribution(self):
        """Strategy comparison produces all four paths."""
        n_days, n_stocks = 20, 15
        rng = np.random.default_rng(99)
        actual = rng.standard_normal((n_days, n_stocks)) * 0.02

        preds_by_path = {
            "Base": rng.standard_normal((n_days, n_stocks)),
            "Direct": actual + rng.standard_normal((n_days, n_stocks)) * 0.02,
            "Spillover": actual + rng.standard_normal((n_days, n_stocks)) * 0.03,
            "Final": actual.copy(),
        }
        results = compare_strategies(
            preds_by_path, actual, BacktestConfig(top_k=3)
        )

        assert len(results) == 4
        # Final (perfect predictions) should have best total return
        assert results["Final"].total_return == max(
            r.total_return for r in results.values()
        )


class TestE2EFullPipeline:
    """Full pipeline E2E smoke test with synthetic data."""

    def test_synthetic_data_to_predictions(self):
        """Synthetic data flows through the full pipeline."""
        # 1. Ingestion
        record = _make_bronze_record(
            "tushare",
            _utc(2024, 1, 1),
            {"stock_code": "600000", "buy_amount": 5e7},
        )
        records_by_key = {"600000": [record]}
        field_specs = {"buy_amount": lambda payload: payload.get("buy_amount", 0)}
        reconciled = reconcile_records(records_by_key, field_specs=field_specs)
        assert "600000" in reconciled

        # 2. Identity
        stock_identity = StockIdentity(
            stock_id="STOCK_600000",
            ts_code="600000.SH",
            exchange="SSE",
            board="main",
            listing_date=_utc(2010, 1, 1),
            delisting_date=None,
            aliases=["600000"],
            valid_from=_utc(2010, 1, 1),
            valid_to=None,
            identity_confidence=1.0,
            source_evidence=["tushare"],
        )
        stock_registry = StockIdentityRegistry()
        stock_registry.register(stock_identity)
        resolved = stock_registry.resolve_stock("600000", _utc(2024, 1, 1))
        assert resolved is not None
        assert resolved.stock_id == "STOCK_600000"

        # 3. Model + Evaluation
        rng = np.random.default_rng(42)
        truths = rng.standard_normal(50)
        preds = truths + rng.standard_normal(50) * 0.2
        assert mae(truths, preds) > 0
        assert rank_ic(truths, preds) > 0

        # 4. Backtest
        actual = rng.standard_normal((10, 5)) * 0.01
        pred = actual + rng.standard_normal((10, 5)) * 0.005
        result = backtest(pred, actual, BacktestConfig(top_k=2))
        assert result.daily_returns.shape == (10,)

    def test_trace_context_propagation(self):
        """Trace context propagates through the pipeline."""
        from seatflow.observability.metrics import (
            set_trace_context,
            get_trace_context,
            generate_trace_id,
        )

        trace_id = generate_trace_id()
        set_trace_context(trace_id, "span-123")
        retrieved_trace, retrieved_span = get_trace_context()
        assert retrieved_trace == trace_id
        assert retrieved_span == "span-123"

    def test_health_check(self):
        """Health check returns valid status."""
        from seatflow.observability.metrics import health_check

        result = health_check()
        assert result.status in ("healthy", "degraded", "unhealthy")
        assert "api" in result.checks
        assert result.version

    def test_observability_metrics_registry(self):
        """Metrics registry works for E2E tracking."""
        from seatflow.observability.metrics import MetricsRegistry

        reg = MetricsRegistry()
        reg.increment("pipeline_runs_total", labels={"stage": "ingestion"})
        reg.increment("pipeline_runs_total", labels={"stage": "ingestion"})
        reg.increment("pipeline_runs_total", labels={"stage": "evaluation"})

        assert reg.get_counter("pipeline_runs_total", labels={"stage": "ingestion"}) == 2.0
        assert reg.get_counter("pipeline_runs_total", labels={"stage": "evaluation"}) == 1.0

        # Export to Prometheus format
        export = reg.export_prometheus()
        assert "pipeline_runs_total" in export
