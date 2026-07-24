import unittest
from datetime import date, timedelta

from seatflow.evaluation.rolling import DEFAULT_SEEDS, build_rolling_folds, run_rolling_evaluation
from seatflow.models.strong_baselines import BASELINE_NAMES
from seatflow.research import FEATURE_NAMES, ResearchSample

START = date(2026, 1, 1)
SEAT_FEATURE_NAMES = FEATURE_NAMES + ("seat_profile_net_share",)


def _sample(index: int, stock_id: str = "000001.SZ") -> ResearchSample:
    label = index % 2
    signed = float(index if label else -index)
    label_return = 0.02 if label else -0.02
    return ResearchSample(
        sample_id=f"{stock_id}:{index:03d}",
        as_of_date=START + timedelta(days=index),
        stock_id=stock_id,
        feature_names=SEAT_FEATURE_NAMES,
        features=(
            max(signed, 0.0),
            max(-signed, 0.0),
            signed,
            1.0,
            label_return,
            label_return,
            label_return,
            label_return,
            1.0 + label_return,
            abs(label_return),
            abs(label_return) / 2.0,
            0.7 if label else -0.7,
        ),
        label_return=label_return,
        label=label,
        label_start_date=START + timedelta(days=index + 1),
        label_end_date=START + timedelta(days=index + 2),
    )


class RollingEvaluationTest(unittest.TestCase):
    def test_builds_at_least_three_purged_embargoed_time_folds(self) -> None:
        samples = tuple(_sample(index) for index in range(18))

        folds = build_rolling_folds(
            samples,
            n_folds=3,
            min_train_dates=6,
            purge_days=2,
            embargo_days=1,
        )

        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertTrue(fold.train)
            self.assertTrue(fold.test)
            self.assertLess(max(sample.as_of_date for sample in fold.train), fold.test_start)
            self.assertTrue(
                all(
                    sample.label_end_date < fold.test_start - timedelta(days=2)
                    for sample in fold.train
                )
            )
            self.assertTrue(
                all(
                    sample.as_of_date < fold.test_start - timedelta(days=1)
                    for sample in fold.train
                )
            )
            self.assertTrue(
                all(fold.test_start <= sample.as_of_date <= fold.test_end for sample in fold.test)
            )

    def test_runs_five_seeds_four_baselines_and_aggregates_without_selection(self) -> None:
        samples = tuple(_sample(index) for index in range(24))

        evaluation = run_rolling_evaluation(
            samples,
            n_folds=3,
            min_train_dates=9,
            top_k=2,
            epochs=20,
            learning_rate=0.05,
        )

        self.assertEqual(evaluation.seeds, DEFAULT_SEEDS)
        self.assertEqual(len(evaluation.folds), 3)
        self.assertEqual(
            len(evaluation.results),
            len(DEFAULT_SEEDS) * 3 * len(BASELINE_NAMES),
        )
        self.assertEqual(
            {aggregate.baseline_name for aggregate in evaluation.aggregates},
            set(BASELINE_NAMES),
        )
        for aggregate in evaluation.aggregates:
            self.assertEqual(aggregate.seed_count, 5)
            self.assertEqual(aggregate.fold_count, 3)
            self.assertEqual(aggregate.run_count, 15)
            self.assertTrue(aggregate.raw_scores)
            self.assertIsNotNone(aggregate.mean_auc)
            self.assertIsNotNone(aggregate.worst_fold_auc)
            self.assertIn("worst_fold_top_k_mean_return", aggregate.to_dict())
            self.assertIn("raw_scores", aggregate.to_dict())

        payload = evaluation.to_dict()
        self.assertIn("scores", payload["results"][0]["baseline"])

    def test_invalid_seed_count_and_empty_folds_fail_closed(self) -> None:
        samples = tuple(_sample(index) for index in range(8))

        with self.assertRaisesRegex(ValueError, "exactly 5 seeds"):
            run_rolling_evaluation(samples, seeds=(1, 2, 3), epochs=1)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_rolling_folds(samples, n_folds=2)
        with self.assertRaisesRegex(ValueError, "not enough unique dates"):
            build_rolling_folds(samples[:3], n_folds=3, min_train_dates=2)


if __name__ == "__main__":
    unittest.main()
