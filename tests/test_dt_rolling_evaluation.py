from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from seatflow.data.contracts import DailyBar
from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.datasets.pyg import TensorizedWindowSample, tensorize_window_samples
from seatflow.datasets.windows import build_window_samples
from seatflow.evaluation.dt_rolling import (
    _scores,
    build_dt_rolling_folds,
    run_dt_seathgt_rolling_evaluation,
)
from seatflow.graph.tensorize import PyGUnavailableError, TensorizedGraph
from seatflow.graph_transforms import GraphTransformError, fit_apply_fold_graph_transform
from seatflow.models.seat_hgt import DTSeatHGTConfig, PyGModelUnavailableError
from seatflow.training.dtseathgt import DTSeatHGTTrainingConfig

START = date(2026, 1, 1)


class DTRollingEvaluationTest(unittest.TestCase):
    def test_builds_purged_embargoed_folds_from_tensorized_sample_metadata(self) -> None:
        samples = tuple(_sample(index) for index in range(18))

        folds = build_dt_rolling_folds(
            samples,
            selection_horizon=1,
            n_folds=3,
            min_train_dates=12,
            purge_days=2,
            embargo_days=1,
        )

        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertTrue(fold.train)
            self.assertTrue(fold.test)
            self.assertTrue(
                all(
                    sample.as_of_date is not None
                    and sample.as_of_date < fold.test_start - timedelta(days=1)
                    for sample in fold.train
                )
            )
            self.assertTrue(
                all(
                    sample.label_end_dates[1] < fold.test_start - timedelta(days=2)
                    for sample in fold.train
                )
            )
            self.assertTrue(
                all(
                    sample.as_of_date is not None
                    and sample.as_of_date < fold.test_start - timedelta(days=5)
                    for sample in fold.train
                )
            )

    def test_requires_five_unique_seeds_before_training(self) -> None:
        config = DTSeatHGTTrainingConfig(
            model=DTSeatHGTConfig(
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                horizons=(1, 3, 5),
            )
        )
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "exactly 5 unique seeds"):
                run_dt_seathgt_rolling_evaluation(
                    (_sample(0),),
                    config,
                    temporary_directory,
                    seeds=(1, 2, 3),
                )

    def test_rolling_uses_model_max_horizon_for_purge_and_embargo(self) -> None:
        config = DTSeatHGTTrainingConfig(
            model=DTSeatHGTConfig(
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                horizons=(1, 3, 5),
            )
        )
        with (
            patch("seatflow.evaluation.dt_rolling.build_dt_rolling_folds") as build_folds,
            patch("seatflow.evaluation.dt_rolling.run_dt_seathgt_folds_evaluation"),
        ):
            build_folds.return_value = ()
            with TemporaryDirectory() as temporary_directory:
                run_dt_seathgt_rolling_evaluation(
                    (_sample(0),), config, temporary_directory, diagnostic_e1=True, seeds=(0,)
                )
        self.assertEqual(build_folds.call_args.kwargs["purge_days"], 5)
        self.assertEqual(build_folds.call_args.kwargs["embargo_days"], 5)

    def test_rolling_rejects_v2_until_dedicated_evaluator_exists(self) -> None:
        config = DTSeatHGTTrainingConfig(
            model=DTSeatHGTConfig(
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                horizons=(1, 3, 5),
                direction_classes=3,
            )
        )
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "dedicated V2 rolling evaluator"):
                run_dt_seathgt_rolling_evaluation(
                    (_sample(0),), config, temporary_directory, diagnostic_e1=True, seeds=(0,)
                )

    def test_missing_sample_time_metadata_fails_closed(self) -> None:
        broken = TensorizedWindowSample(
            sample_id="broken",
            source_hash="a" * 64,
            target_stock_id="000001.SZ",
            target_node_index=0,
            graph=TensorizedGraph(data=None, metadata={}),
            labels={1: 1},
            returns={1: 0.01},
        )

        with self.assertRaisesRegex(ValueError, "missing as_of_date"):
            build_dt_rolling_folds((broken,), selection_horizon=1)

    def test_scores_expose_direction_probability_and_return_head_selection(self) -> None:
        sample = _sample(1)

        scores = _scores(
            (sample,),
            {
                sample.sample_id: {
                    "direction_logits": {"1": 0.0},
                    "return_predictions": {"1": -0.025},
                }
            },
            horizon=1,
            seed=7,
            fold_id="fold-x",
        )

        self.assertEqual(scores[0]["score"], 0.5)
        self.assertEqual(scores[0]["direction_probability"], 0.5)
        self.assertEqual(scores[0]["expected_return"], -0.025)
        self.assertEqual(scores[0]["selection_score"], -0.025)
        self.assertEqual(scores[0]["selection_score_policy"], "return_head_expected_return")

        with self.assertRaisesRegex(ValueError, "return head"):
            _scores(
                (sample,),
                {sample.sample_id: {"direction_logits": {"1": 0.0}}},
                horizon=1,
                seed=7,
                fold_id="fold-x",
            )

    def test_runs_all_five_seed_three_fold_train_restore_predict_cycles(self) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))
        config = DTSeatHGTTrainingConfig(
            model=DTSeatHGTConfig(
                hidden_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                horizons=(1, 3, 5),
            ),
            epochs=1,
        )
        with TemporaryDirectory() as temporary_directory:
            evaluation = run_dt_seathgt_rolling_evaluation(
                samples,
                config,
                temporary_directory,
                min_train_dates=12,
            )
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "transforms"
                    / "fold_0"
                    / "feature_transform.json"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "diagnostics"
                    / "fold_0"
                    / "feature_distribution.json"
                ).exists()
            )

        self.assertEqual(len(evaluation.folds), 3)
        self.assertEqual(len(evaluation.results), 15)
        self.assertEqual(len(evaluation.ablations), 45)
        self.assertEqual(
            {str(row["group"]) for row in evaluation.ablations},
            {"seat_profile", "time_edges", "direction_edges"},
        )
        self.assertTrue(
            all("mean_abs_score_delta" in row for row in evaluation.ablations)
        )
        self.assertEqual({result.seed for result in evaluation.results}, {0, 1, 2, 3, 4})
        self.assertTrue(all(result.scores for result in evaluation.results))
        rows = cast(list[object], evaluation.to_dict()["rows"])
        self.assertEqual(len(rows), 15)

    def test_fold_transform_rejects_shared_train_test_graph_and_preserves_input_graphs(
        self,
    ) -> None:
        try:
            samples = _tensorized_samples()
        except (PyGUnavailableError, PyGModelUnavailableError) as error:
            self.skipTest(str(error))
        original = samples[0].graph.data["stock"].x.clone()
        with self.assertRaisesRegex(GraphTransformError, "shared"):
            fit_apply_fold_graph_transform((samples[0],), (samples[0],))
        transformed = fit_apply_fold_graph_transform(samples[:3], samples[3:4])
        self.assertEqual(samples[0].graph.data["stock"].x.tolist(), original.tolist())
        self.assertIsNot(transformed.train[0].graph.data, samples[0].graph.data)
        self.assertEqual(len(str(transformed.diagnostics["diagnostics_hash"])), 64)


def _sample(index: int) -> TensorizedWindowSample:
    as_of = START + timedelta(days=index)
    return TensorizedWindowSample(
        sample_id=f"sample-{index:03d}",
        source_hash="a" * 64,
        target_stock_id="000001.SZ",
        target_node_index=0,
        graph=TensorizedGraph(data=None, metadata={}),
        labels={1: index % 2, 3: index % 2, 5: index % 2},
        returns={1: 0.01 if index % 2 else -0.01, 3: 0.02, 5: 0.03},
        as_of_date=as_of,
        label_end_dates={
            1: as_of + timedelta(days=1),
            3: as_of + timedelta(days=3),
            5: as_of + timedelta(days=5),
        },
    )


def _tensorized_samples() -> tuple[TensorizedWindowSample, ...]:
    calendar = tuple(START + timedelta(days=index) for index in range(22))
    records = tuple(
        DragonTigerRecord(
            trade_date=trade_date,
            seat_id=f"seat-{index % 3}",
            stock_id="000001.SZ",
            side=TradeSide.BUY if index % 2 else TradeSide.SELL,
            amount=1_000_000.0 + index,
            available_time=datetime.combine(trade_date, datetime.min.time()),
        )
        for index, trade_date in enumerate(calendar)
    )
    bars = tuple(
        DailyBar(
            trade_date=trade_date,
            stock_id="000001.SZ",
            open_price=Decimal("10") + Decimal(index),
            high_price=Decimal("11") + Decimal(index),
            low_price=Decimal("9") + Decimal(index),
            close_price=Decimal("10") + Decimal(index),
            volume=Decimal("100000"),
            amount=Decimal("1000000"),
            source_name="fixture",
            source_version="fixture-v1",
            available_time=datetime.combine(trade_date, datetime.min.time()),
            volume_unit="shares",
            amount_unit="cny",
        )
        for index, trade_date in enumerate(calendar)
    )
    windows = build_window_samples(
        records,
        bars,
        as_of_dates=calendar[:16],
        target_stock_ids=("000001.SZ",),
        horizons=(1, 3, 5),
        cutoff_for_date=lambda trade_date: datetime.combine(trade_date, datetime.min.time()),
        decay_lambda=0.1,
        max_edge_age_days=20,
        adjustment_policy="frozen_adjusted_close_v1",
        calendar_policy="frozen_trading_calendar_v1",
        trading_calendar=calendar,
    )
    return tensorize_window_samples(windows)


if __name__ == "__main__":
    unittest.main()
