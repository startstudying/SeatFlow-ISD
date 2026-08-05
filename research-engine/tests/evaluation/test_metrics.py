"""Tests for evaluation metrics (Task 18)."""

import math
import numpy as np
import pytest

from seatflow.evaluation.metrics import (
    mae, mse, rmse, huber, auc, brier,
    pearson_ic, rank_ic, icir, ndcg_at_k, hit_at_k,
    edge_aupr, recall_at_k, ece, aggregate_metric,
)


class TestRegressionMetrics:
    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        assert mae(y_true, y_pred) == pytest.approx(0.333, abs=0.01)

    def test_mse(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 2.0])
        assert mse(y_true, y_pred) == pytest.approx(0.5)

    def test_rmse(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 2.0])
        assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(0.5))

    def test_huber(self):
        y_true = np.array([0.0])
        y_pred = np.array([0.5])
        # delta=1.0: quadratic for |diff| < delta
        # 0.5 * 0.5^2 = 0.125
        assert huber(y_true, y_pred, delta=1.0) == pytest.approx(0.125)

    def test_huber_large_error(self):
        y_true = np.array([0.0])
        y_pred = np.array([3.0])
        # delta=1.0: linear for |diff| > delta
        # 0.5 * 1^2 + 1 * 2 = 2.5
        assert huber(y_true, y_pred, delta=1.0) == pytest.approx(2.5)


class TestClassificationMetrics:
    def test_auc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert auc(y_true, y_score) == pytest.approx(1.0)

    def test_auc_random(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        assert auc(y_true, y_score) == pytest.approx(0.5)

    def test_brier_perfect(self):
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.0, 1.0])
        assert brier(y_true, y_prob) == pytest.approx(0.0)

    def test_brier_worst(self):
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([1.0, 0.0])
        assert brier(y_true, y_prob) == pytest.approx(1.0)


class TestRankingMetrics:
    def test_pearson_ic_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.0, 4.0, 6.0, 8.0])
        assert pearson_ic(y_true, y_pred) == pytest.approx(1.0)

    def test_pearson_ic_anticorrelated(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert pearson_ic(y_true, y_pred) == pytest.approx(-1.0)

    def test_rank_ic_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        assert rank_ic(y_true, y_pred) == pytest.approx(1.0)

    def test_icir(self):
        ic_values = [0.1, 0.2, 0.15, 0.05]
        expected = np.mean(ic_values) / np.std(ic_values)
        assert icir(ic_values) == pytest.approx(expected, abs=0.01)

    def test_icir_single_value(self):
        assert icir([0.1]) == 0.0

    def test_ndcg_perfect(self):
        y_true = np.array([3.0, 2.0, 1.0, 0.0])
        y_pred = np.array([3.0, 2.0, 1.0, 0.0])
        assert ndcg_at_k(y_true, y_pred, k=4) == pytest.approx(1.0)

    def test_ndcg_k_larger_than_n(self):
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([1.0, 0.0])
        assert ndcg_at_k(y_true, y_pred, k=10) == pytest.approx(1.0)

    def test_hit_at_k_perfect(self):
        y_true = np.array([4.0, 3.0, 2.0, 1.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert hit_at_k(y_true, y_pred, k=2) == pytest.approx(1.0)

    def test_hit_at_k_partial(self):
        y_true = np.array([4.0, 3.0, 2.0, 1.0])
        y_pred = np.array([4.0, 1.0, 3.0, 2.0])
        # top-2 pred: {0, 3}, top-2 true: {0, 1}
        # overlap: {0} -> 1/2
        assert hit_at_k(y_true, y_pred, k=2) == pytest.approx(0.5)


class TestEdgeMetrics:
    def test_edge_aupr_perfect(self):
        edge_true = np.array([0, 0, 1, 1])
        edge_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert edge_aupr(edge_true, edge_score) == pytest.approx(1.0, abs=0.01)

    def test_recall_at_k(self):
        edge_true = np.array([0, 0, 1, 1])
        edge_score = np.array([0.1, 0.2, 0.8, 0.9])
        # top-2 by score: indices 3, 2 -> both true positives
        assert recall_at_k(edge_true, edge_score, k=2) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        edge_true = np.array([1, 0, 1, 0])
        edge_score = np.array([0.9, 0.8, 0.1, 0.0])
        # top-2 by score: indices 0, 1 -> 1 true positive out of 2 total
        assert recall_at_k(edge_true, edge_score, k=2) == pytest.approx(0.5)


class TestCalibrationMetrics:
    def test_ece_perfect(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)

    def test_ece_worst(self):
        y_true = np.array([0.0, 0.0])
        y_prob = np.array([1.0, 1.0])
        assert ece(y_true, y_prob) == pytest.approx(1.0)


class TestAggregation:
    def test_aggregate_metric(self):
        values = [0.1, 0.2, 0.3]
        result = aggregate_metric(values)
        assert result["mean"] == pytest.approx(0.2)
        assert result["min"] == pytest.approx(0.1)
        assert result["max"] == pytest.approx(0.3)

    def test_aggregate_empty(self):
        result = aggregate_metric([])
        assert result["mean"] == 0.0
