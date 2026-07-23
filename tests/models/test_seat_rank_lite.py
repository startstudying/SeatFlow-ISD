from __future__ import annotations

import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import PyGUnavailableError, tensorize_hetero_graph
from seatflow.models.seat_rank_lite import (
    SeatRankLiteConfig,
    build_seat_rank_lite,
    count_parameters,
    validate_parameter_count,
)


class SeatRankLiteTest(unittest.TestCase):
    def test_four_heads_have_one_value_per_stock(self) -> None:
        try:
            graph = _tensorized_graph()
        except PyGUnavailableError as error:
            self.skipTest(str(error))
        model = build_seat_rank_lite(SeatRankLiteConfig(), graph)
        outputs = model(graph.data)

        self.assertEqual(outputs["rank_score"].shape, (2,))
        self.assertEqual(outputs["residual_return"].shape, (2,))
        self.assertEqual(outputs["direction_logit"].shape, (2,))
        self.assertEqual(outputs["uncertainty"].shape, (2,))
        self.assertTrue(bool((outputs["uncertainty"] > 0).all()))
        self.assertFalse(outputs["model_valid"])
        self.assertLess(count_parameters(model), 3_000_000)
        report = validate_parameter_count(model)
        self.assertFalse(report["recommended_limit_exceeded"])
        for layer in model.layers:
            self.assertTrue(layer.last_edge_gate_shapes)
            for rows, width in layer.last_edge_gate_shapes.values():
                self.assertGreater(rows, 0)
                self.assertEqual(width, 2)
                self.assertNotEqual(width, model.config.hidden_dim)

    def test_cuda_forward_uses_the_gpu_when_available(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        graph = _tensorized_graph()
        model = build_seat_rank_lite(SeatRankLiteConfig(), graph).to("cuda")
        data = graph.data.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(data)

        self.assertEqual(outputs["rank_score"].device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(outputs["rank_score"]).all()))

    def test_parameter_hard_limit_fails_closed(self) -> None:
        import torch

        oversized = torch.nn.Linear(2_500_001, 2, bias=False)
        with self.assertRaisesRegex(ValueError, "hard limit exceeded"):
            validate_parameter_count(oversized)


def _tensorized_graph():  # type: ignore[no-untyped-def]
    snapshot = DynamicGraphBuilder(
        decay_lambda=0.1,
        max_edge_age_days=20,
        enable_seat_cooccurrence=False,
        enable_concept_graph=False,
        add_reverse_relations=True,
        aggregate_duplicate_edges=True,
    ).build(
        (
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id="seat-a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=100.0,
                industry="bank",
            ),
            DragonTigerRecord(
                trade_date=date(2026, 1, 1),
                seat_id="seat-b",
                stock_id="000002.SZ",
                side=TradeSide.SELL,
                amount=50.0,
                industry="tech",
            ),
        ),
        date(2026, 1, 1),
    )
    return tensorize_hetero_graph(snapshot)


if __name__ == "__main__":
    unittest.main()
