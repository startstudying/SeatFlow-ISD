from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date

from seatflow.datasets.pyg import TensorizedWindowSample
from seatflow.graph.tensorize import TensorizedGraph
from seatflow.return_transforms import (
    ReturnTransformError,
    fit_return_transform,
    return_transform_diagnostics,
)


class ReturnTransformTest(unittest.TestCase):
    def test_fit_inverse_is_deterministic_and_training_only(self) -> None:
        train = (_sample("train-a", -0.02), _sample("train-b", 0.02), _sample("train-c", 0.04))
        state = fit_return_transform(train)
        self.assertEqual(state.inverse_transform(1, state.transform(1, 0.03)), 0.03)
        self.assertEqual(state.statistics[1].count, 3)
        self.assertNotEqual(state.transform(1, 1.0), state.transform(1, 0.03))
        diagnostics = return_transform_diagnostics((_sample("test", 1.0),), state)
        self.assertEqual(len(str(diagnostics["diagnostics_hash"])), 64)

    def test_zero_variance_nonfinite_and_horizon_mismatch_fail_closed(self) -> None:
        with self.assertRaisesRegex(ReturnTransformError, "zero variance"):
            fit_return_transform((_sample("a", 0.01), _sample("b", 0.01)))
        with self.assertRaisesRegex(ReturnTransformError, "return horizons"):
            fit_return_transform(
                (_sample("a", 0.01), replace(_sample("b", 0.02), returns={3: 0.02}))
            )
        with self.assertRaisesRegex(ReturnTransformError, "finite"):
            fit_return_transform((_sample("a", float("nan")), _sample("b", 0.02)))


def _sample(sample_id: str, value: float) -> TensorizedWindowSample:
    return TensorizedWindowSample(
        sample_id=sample_id,
        source_hash="a" * 64,
        target_stock_id="000001.SZ",
        target_node_index=0,
        graph=TensorizedGraph(data=None, metadata={}),
        labels={1: int(value > 0)},
        returns={1: value},
        as_of_date=date(2026, 1, 1),
    )
