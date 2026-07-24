from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import PyGUnavailableError, tensorize_hetero_graph
from seatflow.graph.types import HeteroGraphSnapshot
from seatflow.models.seat_hgt import (
    DTSeatHGTConfig,
    PyGModelUnavailableError,
    build_dt_seat_hgt,
    dt_seat_hgt_checkpoint_payload,
)
from seatflow.training.runner import run_pyg_shape_smoke


class DTSeatHGTModelTest(unittest.TestCase):
    def test_dtseathgt_multi_task_heads_shape_or_clear_skip(self) -> None:
        graph = _graph()
        try:
            tensorized = tensorize_hetero_graph(graph)
            model = build_dt_seat_hgt(
                DTSeatHGTConfig(
                    hidden_dim=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    horizons=(1, 3, 5),
                ),
                tensorized,
            )
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        outputs = model(tensorized.data)

        self.assertFalse(outputs["model_valid"])
        self.assertEqual(set(outputs["direction_logits"]), {1, 3, 5})
        self.assertEqual(set(outputs["return_predictions"]), {1, 3, 5})
        for horizon in (1, 3, 5):
            self.assertEqual(list(outputs["direction_logits"][horizon].shape), [1])
            self.assertEqual(list(outputs["return_predictions"][horizon].shape), [1])

    def test_dtseathgt_three_class_direction_heads(self) -> None:
        try:
            tensorized = tensorize_hetero_graph(_graph())
            model = build_dt_seat_hgt(
                DTSeatHGTConfig(
                    hidden_dim=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    horizons=(1, 3, 5),
                    direction_classes=3,
                ),
                tensorized,
            )
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        outputs = model(tensorized.data)
        for horizon in (1, 3, 5):
            self.assertEqual(list(outputs["direction_logits"][horizon].shape), [1, 3])

    def test_dtseathgt_fails_closed_when_edge_weight_is_missing(self) -> None:
        graph = _graph()
        try:
            tensorized = tensorize_hetero_graph(graph)
            model = build_dt_seat_hgt(
                DTSeatHGTConfig(
                    hidden_dim=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    horizons=(1, 3, 5),
                ),
                tensorized,
            )
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        edge_type = tensorized.data.edge_types[0]
        del tensorized.data[edge_type].edge_weight

        with self.assertRaisesRegex(ValueError, "edge_weight missing"):
            model(tensorized.data)

    def test_dtseathgt_consumes_relation_edge_attr_beyond_edge_weight(self) -> None:
        graph = _graph()
        try:
            import torch

            tensorized = tensorize_hetero_graph(graph)
            _attach_edge_attr(tensorized.data, torch)
            torch.manual_seed(101)
            model = build_dt_seat_hgt(
                DTSeatHGTConfig(
                    hidden_dim=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    horizons=(1, 3, 5),
                ),
                tensorized,
            )
        except (ModuleNotFoundError, PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        model.eval()
        with torch.no_grad():
            baseline = model(tensorized.data)["direction_logits"][1].clone()
            for edge_type in tensorized.data.edge_types:
                tensorized.data[edge_type].edge_attr[:, 1] += 10.0
            perturbed = model(tensorized.data)["direction_logits"][1].clone()

        self.assertFalse(torch.allclose(baseline, perturbed))

    def test_dtseathgt_fails_closed_when_edge_attr_schema_is_missing_or_wrong(self) -> None:
        graph = _graph()
        try:
            import torch

            tensorized = tensorize_hetero_graph(graph)
            _attach_edge_attr(tensorized.data, torch)
            model = build_dt_seat_hgt(
                DTSeatHGTConfig(
                    hidden_dim=16,
                    num_heads=2,
                    num_layers=1,
                    dropout=0.0,
                    horizons=(1, 3, 5),
                ),
                tensorized,
            )
        except (ModuleNotFoundError, PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        edge_type = tensorized.data.edge_types[0]
        original = tensorized.data[edge_type].edge_attr
        del tensorized.data[edge_type].edge_attr
        with self.assertRaisesRegex(ValueError, "edge_attr missing"):
            model(tensorized.data)

        tensorized.data[edge_type].edge_attr = original[:, :1]
        with self.assertRaisesRegex(ValueError, "edge_attr width mismatch"):
            model(tensorized.data)

    def test_pyg_shape_smoke_writes_lineage_when_optional_stack_exists(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            try:
                result = run_pyg_shape_smoke(
                    Path("configs/default.toml"),
                    Path(temporary_directory),
                    run_id="pyg-shape-test",
                )
            except PyGModelUnavailableError as error:
                self.skipTest(str(error))

            self.assertFalse(result.model_valid)
            self.assertIn("model_architecture", result.hashes)
            lines = result.lineage_path.read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[-1])
            record = cast(dict[str, object], record)
            self.assertEqual(record["run_type"], "pyg_shape_smoke")
            self.assertFalse(record["model_valid"])
            hashes = cast(dict[str, str], record["hashes"])
            self.assertIn("model_architecture", hashes)

    def test_checkpoint_payload_never_marks_dtseathgt_valid(self) -> None:
        payload = dt_seat_hgt_checkpoint_payload(
            DTSeatHGTConfig(
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                horizons=(1,),
            )
        )

        self.assertFalse(payload["model_valid"])
        risk_flags = cast(list[str], payload["risk_flags"])
        self.assertIn("no_real_training_or_backtest", risk_flags)


def _graph() -> HeteroGraphSnapshot:
    return DynamicGraphBuilder(decay_lambda=0.1, max_edge_age_days=20).build(
        (
            DragonTigerRecord(
                trade_date=date(2026, 1, 15),
                seat_id="sample_institution_a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=12_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="sample_daily_deviation",
            ),
            DragonTigerRecord(
                trade_date=date(2026, 1, 14),
                seat_id="sample_arbitrage_b",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=4_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="sample_daily_deviation",
            ),
        ),
        as_of_date=date(2026, 1, 15),
    )


def _attach_edge_attr(data: Any, torch: Any) -> None:
    for edge_type in data.edge_types:
        edge_weight = data[edge_type].edge_weight
        side_or_static = (
            torch.ones_like(edge_weight)
            if edge_type[1] == "buys"
            else -torch.ones_like(edge_weight)
            if edge_type[1] == "sells"
            else torch.zeros_like(edge_weight)
        )
        data[edge_type].edge_attr = torch.stack(
            (torch.log1p(torch.clamp(edge_weight, min=0.0)), side_or_static),
            dim=-1,
        )


if __name__ == "__main__":
    unittest.main()
