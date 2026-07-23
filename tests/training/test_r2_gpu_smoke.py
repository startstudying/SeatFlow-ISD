from __future__ import annotations

import unittest
from datetime import date
from tempfile import TemporaryDirectory

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.graph.tensorize import tensorize_hetero_graph
from seatflow.models.seat_rank_lite import SeatRankLiteConfig
from seatflow.training.r2_runner import (
    R2GraphBatch,
    R2TrainingConfig,
    train_seat_rank_lite_seed,
)


class R2GpuSmokeTest(unittest.TestCase):
    def test_two_serial_seeds_train_and_release_gpu_memory(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the R2 GPU smoke test")
        batch = _batch()
        model_config = SeatRankLiteConfig()
        training_config = R2TrainingConfig(
            epochs=2,
            gradient_accumulation_steps=1,
            device="cuda",
        )
        with TemporaryDirectory() as temporary_directory:
            first = train_seat_rank_lite_seed(
                (batch,),
                model_config,
                training_config,
                temporary_directory,
                fold_id=0,
                seed=2026,
                validation_batches=(batch,),
            )
            second = train_seat_rank_lite_seed(
                (batch,),
                model_config,
                training_config,
                temporary_directory,
                fold_id=0,
                seed=2027,
                validation_batches=(batch,),
            )
            state = torch.load(first.checkpoint_path, weights_only=True)

        self.assertEqual(first.device, "cuda")
        self.assertEqual(first.epochs_completed, 2)
        self.assertTrue(all(row["finite"] for row in first.step_metrics))
        self.assertLess(first.peak_reserved_bytes, training_config.hard_memory_bytes)
        self.assertLess(second.peak_reserved_bytes, training_config.hard_memory_bytes)
        self.assertTrue(all(tensor.device.type == "cpu" for tensor in state.values()))
        self.assertLessEqual(
            second.allocated_after_release_bytes,
            second.peak_allocated_bytes,
        )


def _batch() -> R2GraphBatch:
    import torch

    records = tuple(
        DragonTigerRecord(
            trade_date=date(2026, 1, 1),
            seat_id=f"seat-{index}",
            stock_id=f"{index + 1:06d}.SZ",
            side=TradeSide.BUY if index % 2 == 0 else TradeSide.SELL,
            amount=100.0 + index,
            industry="industry-a" if index < 2 else "industry-b",
        )
        for index in range(6)
    )
    snapshot = DynamicGraphBuilder(
        decay_lambda=0.1,
        max_edge_age_days=20,
        enable_seat_cooccurrence=False,
        enable_concept_graph=False,
        add_reverse_relations=True,
        aggregate_duplicate_edges=True,
    ).build(records, date(2026, 1, 1))
    graph = tensorize_hetero_graph(snapshot)
    return R2GraphBatch(
        graph=graph,
        residual_targets=torch.tensor([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]),
        direction_targets=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        prediction_date="2026-01-01",
    )


if __name__ == "__main__":
    unittest.main()
