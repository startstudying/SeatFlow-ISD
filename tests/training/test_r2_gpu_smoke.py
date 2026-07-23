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

    def test_l2_objective_uses_cuda_within_r3_memory_limit(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the R3 L2 GPU smoke test")
        batch = _batch()
        batch = R2GraphBatch(
            graph=batch.graph,
            residual_targets=batch.residual_targets,
            direction_targets=batch.direction_targets,
            prediction_date=batch.prediction_date,
            standardized_residual_targets=torch.tensor(
                [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
            ),
            ranking_targets=torch.linspace(0.0, 1.0, 6),
        )
        with TemporaryDirectory() as temporary_directory:
            result = train_seat_rank_lite_seed(
                (batch,),
                SeatRankLiteConfig(),
                R2TrainingConfig(
                    epochs=1,
                    gradient_accumulation_steps=1,
                    device="cuda",
                    objective="L2",
                    hard_memory_bytes=int(7.2 * 1024**3),
                ),
                temporary_directory,
                fold_id=0,
                seed=2026,
            )

        self.assertEqual(result.device, "cuda")
        self.assertLess(result.peak_reserved_bytes, int(7.2 * 1024**3))

    def test_candidate_node_indices_align_real_graph_outputs(self) -> None:
        import torch

        batch = _batch()
        selected = R2GraphBatch(
            graph=batch.graph,
            residual_targets=batch.residual_targets[:3],
            direction_targets=batch.direction_targets[:3],
            prediction_date=batch.prediction_date,
            target_node_indices=torch.tensor([0, 2, 4]),
            target_stock_ids=("000001.SZ", "000003.SZ", "000005.SZ"),
        )
        with TemporaryDirectory() as temporary_directory:
            result = train_seat_rank_lite_seed(
                (selected,),
                SeatRankLiteConfig(),
                R2TrainingConfig(
                    epochs=1,
                    gradient_accumulation_steps=1,
                    device="cpu",
                    mixed_precision=False,
                ),
                temporary_directory,
                fold_id=0,
                seed=2026,
                prediction_batches=(selected,),
            )
            predictions = __import__("json").loads(
                (
                    result.checkpoint_path.parent / "predictions.json"
                ).read_text(encoding="utf-8")
            )

        snapshot = predictions["2026-01-01"]
        self.assertEqual(snapshot["stock_ids"], list(selected.target_stock_ids))
        self.assertEqual(len(snapshot["rank_score"]), 3)


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
