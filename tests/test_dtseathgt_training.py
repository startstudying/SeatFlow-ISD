from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import (
    EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
    TensorizedWindowSample,
    tensorize_window_sample,
)
from seatflow.datasets.windows import build_window_samples
from seatflow.graph.tensorize import PyGUnavailableError
from seatflow.models.seat_hgt import (
    DTSeatHGTConfig,
    PyGModelUnavailableError,
    build_dt_seat_hgt,
)
from seatflow.training.dtseathgt import (
    DTSeatHGTTrainingConfig,
    _hashes,
    _save_training_progress,
    _set_seed,
    prediction_snapshot_for_samples,
    restore_dt_seat_hgt_seed,
    train_dt_seat_hgt,
)


class DTSeatHGTTrainingTest(unittest.TestCase):
    def test_training_writes_recoverable_checkpoint_and_ablation_evidence(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        with TemporaryDirectory() as temporary_directory:
            config = _training_config(seeds=(11, 13))
            first = train_dt_seat_hgt(samples, config, temporary_directory, run_id="first")
            second = train_dt_seat_hgt(samples, config, temporary_directory, run_id="second")

            self.assertFalse(first.model_valid)
            self.assertTrue(first.checkpoint_path.exists())
            self.assertEqual(first.hashes["config"], second.hashes["config"])
            self.assertEqual(first.hashes["sample"], second.hashes["sample"])
            self.assertEqual(first.metrics["seed_runs"], second.metrics["seed_runs"])
            self.assertIn("checkpoint", first.hashes)

            import torch

            payload = torch.load(first.checkpoint_path, weights_only=False)
            self.assertFalse(payload["model_valid"])
            self.assertEqual({"11", "13"}, set(payload["state_by_seed"]))
            self.assertIn("model_state_dict", payload["state_by_seed"]["11"])
            self.assertIn("optimizer_state_dict", payload["state_by_seed"]["11"])
            self.assertIn("model_state", payload["state_by_seed"]["11"]["hashes"])
            self.assertIn("optimizer_state", payload["state_by_seed"]["11"]["hashes"])
            self.assertIn("runtime_data_scope", payload["state_by_seed"]["11"])
            self.assertEqual(
                samples[0].source_hash,
                payload["state_by_seed"]["11"]["runtime_data_scope"]["sample_source_hashes"][
                    samples[0].sample_id
                ],
            )
            self.assertEqual({"config", "sample", "label", "schema"}, set(payload["hashes"]))
            self.assertEqual(payload["effective_training_config"]["epochs"], config.epochs)
            self.assertEqual(
                payload["effective_training_config"]["gradient_clip_norm"],
                config.gradient_clip_norm,
            )
            self.assertEqual(len(payload["loss_history"]["11"]), config.epochs)
            self.assertIn("stock", payload["feature_schema"])
            seed_runs = payload["seed_runs"]
            self.assertEqual(len(seed_runs), 2)
            self.assertEqual(seed_runs[0]["gradient_clip_norm"], config.gradient_clip_norm)
            self.assertIsNotNone(seed_runs[0]["last_gradient_norm"])
            ablation = seed_runs[0]["edge_weight_ablation"]
            self.assertTrue(ablation["edge_weights_consumed"])

    def test_restore_helper_reproduces_checkpoint_prediction_snapshot(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        with TemporaryDirectory() as temporary_directory:
            config = _training_config(seeds=(17,))
            result = train_dt_seat_hgt(samples, config, temporary_directory, run_id="restore")
            restored = restore_dt_seat_hgt_seed(result.checkpoint_path, samples[0], seed=17)
            restored_snapshot = prediction_snapshot_for_samples(
                restored.model,
                (samples[0],),
                config.model.horizons,
            )
            state_by_seed = cast(dict[str, dict[str, object]], restored.checkpoint["state_by_seed"])
            checkpoint_snapshot = state_by_seed["17"]["prediction_snapshot"]
            self.assertEqual(checkpoint_snapshot, restored_snapshot)
            self.assertFalse(restored.checkpoint["model_valid"])

    def test_resume_reuses_only_a_matching_completed_checkpoint(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        with TemporaryDirectory() as temporary_directory:
            config = _training_config(seeds=(19,))
            first = train_dt_seat_hgt(
                samples,
                config,
                temporary_directory,
                run_id="resume-existing",
            )
            checkpoint_bytes = first.checkpoint_path.read_bytes()
            resumed = train_dt_seat_hgt(
                samples,
                config,
                temporary_directory,
                run_id="resume-existing",
                resume_existing=True,
            )

            self.assertTrue(resumed.metrics["resumed"])
            self.assertEqual(resumed.checkpoint_path.read_bytes(), checkpoint_bytes)

            incompatible = DTSeatHGTTrainingConfig(
                model=config.model,
                learning_rate=config.learning_rate,
                epochs=2,
                seeds=config.seeds,
            )
            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                train_dt_seat_hgt(
                    samples,
                    incompatible,
                    temporary_directory,
                    run_id="resume-existing",
                    resume_existing=True,
                )

    def test_validation_monitoring_is_logged_and_bound_to_checkpoint_identity(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        with TemporaryDirectory() as temporary_directory:
            config = replace(_training_config(), epochs=2, scheduler_patience=1)
            result = train_dt_seat_hgt(
                samples,
                config,
                temporary_directory,
                run_id="validation",
                validation_samples=samples,
            )

            import torch

            payload = torch.load(result.checkpoint_path, weights_only=False)
            self.assertIn("validation_sample", payload["hashes"])
            self.assertIn("validation_label", payload["hashes"])
            epoch_rows = payload["seed_runs"][0]["epochs"]
            self.assertTrue(all(row["validation_loss"] is not None for row in epoch_rows))
            self.assertTrue(all("learning_rate" in row for row in epoch_rows))

            with self.assertRaisesRegex(ValueError, "hashes do not match"):
                train_dt_seat_hgt(
                    samples,
                    config,
                    temporary_directory,
                    run_id="validation",
                    resume_existing=True,
                )

    def test_resume_recovers_a_matching_in_progress_epoch_checkpoint(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))

        with TemporaryDirectory() as temporary_directory:
            import torch

            config = replace(_training_config(), epochs=2)
            _set_seed(config.seeds[0], torch)
            model = build_dt_seat_hgt(config.model, samples[0].graph)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
            progress_path = Path(temporary_directory) / "partial" / "dtseathgt.in_progress.pt"
            _save_training_progress(
                torch,
                progress_path,
                "partial",
                samples,
                config,
                _hashes(samples, config),
                (),
                {},
                {
                    "seed": config.seeds[0],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "losses": [],
                    "epochs": [],
                    "best_model_state": None,
                    "best_epoch": 0,
                    "best_monitor": float("inf"),
                    "stale_epochs": 0,
                },
            )

            result = train_dt_seat_hgt(
                samples, config, temporary_directory, run_id="partial", resume_existing=True
            )

            self.assertTrue(result.checkpoint_path.exists())
            self.assertFalse(progress_path.exists())

    def test_training_fails_closed_on_empty_samples(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "at least one tensorized sample"):
                train_dt_seat_hgt((), _training_config(), temporary_directory)

    def test_training_fails_closed_on_missing_target_mapping(self) -> None:
        try:
            sample = _tensorized_samples()[0]
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))
        broken = type(sample)(
            sample_id=sample.sample_id,
            source_hash=sample.source_hash,
            target_stock_id=sample.target_stock_id,
            target_node_index=99,
            graph=sample.graph,
            labels=sample.labels,
            returns=sample.returns,
        )
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "target_node_index is out of range"):
                train_dt_seat_hgt((broken,), _training_config(), temporary_directory)

    def test_training_rejects_executable_v2_labels_until_multiclass_head_exists(self) -> None:
        try:
            sample = _tensorized_samples()[0]
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))
        v2_sample = replace(
            sample,
            labels={1: -1, 3: 0, 5: 1},
            label_schema_version=EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        )

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "direction head and tensorized label schema"):
                train_dt_seat_hgt((v2_sample,), _training_config(), temporary_directory)

    def test_training_accepts_v2_labels_with_three_class_direction_head(self) -> None:
        try:
            sample = _tensorized_samples()[0]
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))
        v2_sample = replace(
            sample,
            labels={1: -1, 3: 0, 5: 1},
            label_schema_version=EXECUTABLE_EXCESS_RETURN_LABEL_SCHEMA,
        )
        config = replace(
            _training_config(),
            model=replace(_training_config().model, direction_classes=3),
        )

        with TemporaryDirectory() as temporary_directory:
            result = train_dt_seat_hgt((v2_sample,), config, temporary_directory)

        self.assertEqual(result.status, "completed")
        self.assertFalse(result.model_valid)


def _training_config(seeds: tuple[int, ...] = (7,)) -> DTSeatHGTTrainingConfig:
    return DTSeatHGTTrainingConfig(
        model=DTSeatHGTConfig(
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            horizons=(1, 3, 5),
        ),
        learning_rate=0.001,
        epochs=1,
        seeds=seeds,
    )


def _tensorized_samples() -> tuple[TensorizedWindowSample, ...]:
    as_of = date(2026, 1, 15)
    cutoff = datetime(2026, 1, 15, 18, 0, 0)
    samples = build_window_samples(
        records=(
            DragonTigerRecord(
                trade_date=as_of,
                seat_id="seat_a",
                stock_id="000001.SZ",
                side=TradeSide.BUY,
                amount=10_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="daily_deviation",
                available_time=cutoff - timedelta(hours=1),
            ),
            DragonTigerRecord(
                trade_date=as_of - timedelta(days=1),
                seat_id="seat_b",
                stock_id="000001.SZ",
                side=TradeSide.SELL,
                amount=3_000_000,
                industry="banking",
                concepts=("large_cap",),
                event_reason="daily_deviation",
                available_time=cutoff - timedelta(days=1),
            ),
        ),
        bars=tuple(
            DailyBar(
                stock_id="000001.SZ",
                trade_date=as_of + timedelta(days=offset),
                open_price=Decimal("10.0") + Decimal(str(offset)) / Decimal("10"),
                high_price=Decimal("10.1") + Decimal(str(offset)) / Decimal("10"),
                low_price=Decimal("9.9") + Decimal(str(offset)) / Decimal("10"),
                close_price=Decimal("10.0") + Decimal(str(offset)) / Decimal("10"),
                volume=Decimal("1000000"),
                amount=Decimal("10000000"),
                source_name="fixture",
                available_time=cutoff + timedelta(days=offset),
            )
            for offset in range(0, 6)
        ),
        as_of_dates=(as_of,),
        target_stock_ids=("000001.SZ",),
        horizons=(1, 3, 5),
        cutoff_for_date=lambda _: cutoff,
        decay_lambda=0.1,
        max_edge_age_days=20,
    )
    return tuple(tensorize_window_sample(sample) for sample in samples)


if __name__ == "__main__":
    unittest.main()
