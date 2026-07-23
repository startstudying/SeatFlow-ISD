from __future__ import annotations

import unittest
from datetime import date
from math import log1p

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import PyGUnavailableError, tensorize_hetero_graph
from seatflow.graph_transforms import fit_apply_fold_graph_transform


class EdgeTransformAppliedOnceTest(unittest.TestCase):
    def test_amount_is_signed_log_once_then_robust_scaled(self) -> None:
        try:
            train_graph = _graph(100.0, date(2026, 1, 1))
            test_graph = _graph(1_000_000.0, date(2026, 1, 2))
        except PyGUnavailableError as error:
            self.skipTest(str(error))

        edge_type = ("seat", "buys", "stock")
        schema = train_graph.metadata["edge_attr_schema"]["seat::buys::stock"]
        amount_index = schema.index("amount")
        self.assertAlmostEqual(
            float(train_graph.data[edge_type].edge_attr[0, amount_index]),
            log1p(100.0),
            places=5,
        )

        transformed = fit_apply_fold_graph_transform(
            (_sample("train", train_graph),),
            (_sample("test", test_graph),),
        )
        block_state = transformed.state.blocks["edge::seat::buys::stock"]
        stats = block_state.statistics["amount"]
        self.assertEqual(stats.transform, "robust_standardize")
        self.assertNotAlmostEqual(stats.mean, log1p(log1p(100.0)), places=5)
        self.assertEqual(block_state.fitted_sample_count, 1)


def _graph(amount: float, as_of: date):  # type: ignore[no-untyped-def]
    snapshot = DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
        (
            DragonTigerRecord(
                trade_date=as_of,
                seat_id="seat-a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=amount,
            ),
        ),
        as_of,
    )
    return tensorize_hetero_graph(snapshot)


def _sample(sample_id: str, graph):  # type: ignore[no-untyped-def]
    return TensorizedWindowSample(
        sample_id=sample_id,
        source_hash="a" * 64,
        target_stock_id="000001.SZ",
        target_node_index=0,
        graph=graph,
        labels={1: 1},
        returns={1: 0.01},
    )


if __name__ == "__main__":
    unittest.main()
