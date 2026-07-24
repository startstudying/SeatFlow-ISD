import unittest
from datetime import date

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.evaluation.v2_metrics import (
    V2EvaluationRow,
    evaluate_v2_rows,
    v2_rows_from_prediction_snapshot,
)
from seatflow.graph.tensorize import TensorizedGraph


class V2MetricsTest(unittest.TestCase):
    def test_reports_daily_ranking_and_top_k_metrics(self) -> None:
        rows = (
            V2EvaluationRow("2024-01-02", 1, 0.03, (0.1, 0.1, 0.8)),
            V2EvaluationRow("2024-01-02", -1, -0.02, (0.8, 0.1, 0.1)),
            V2EvaluationRow("2024-01-03", 0, 0.0, (0.1, 0.8, 0.1)),
        )
        result = evaluate_v2_rows(rows, top_k=2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertGreater(result["ndcg_at_k"], 0.0)

    def test_rejects_legacy_label_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires executable"):
            V2EvaluationRow("2024-01-02", 1, 0.01, (0.1, 0.1, 0.8), "future_return_binary_v1")

    def test_converts_only_three_way_v2_logits(self) -> None:
        sample = TensorizedWindowSample(
            sample_id="v2", source_hash="a" * 64, target_stock_id="000001.SZ",
            target_node_index=0, graph=TensorizedGraph(data=None, metadata={}),
            labels={1: 1}, returns={1: 0.02}, as_of_date=date(2024, 1, 2),
            label_schema_version="executable_excess_return_v2",
        )
        rows = v2_rows_from_prediction_snapshot(
            (sample,), {"v2": {"direction_logits": {"1": [-2.0, 0.0, 2.0]}}}, horizon=1
        )
        self.assertGreater(rows[0].probabilities[2], 0.8)
