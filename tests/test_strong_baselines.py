import unittest
from datetime import date, timedelta

from seatflow.models.strong_baselines import BASELINE_NAMES, run_strong_baseline
from seatflow.research import FEATURE_NAMES, ResearchSample

START = date(2026, 1, 1)
SEAT_FEATURE_NAMES = FEATURE_NAMES + ("seat_profile_net_share",)


def _sample(
    index: int, stock_id: str = "000001.SZ", as_of_index: int | None = None
) -> ResearchSample:
    label = index % 2
    event_strength = float(index if label else -index)
    price_return = 0.03 if label else -0.03
    date_index = index if as_of_index is None else as_of_index
    return ResearchSample(
        sample_id=f"{stock_id}:sample_{index:03d}",
        as_of_date=START + timedelta(days=date_index),
        stock_id=stock_id,
        feature_names=SEAT_FEATURE_NAMES,
        features=(
            max(event_strength, 0.0),
            max(-event_strength, 0.0),
            event_strength,
            1.0,
            price_return,
            price_return,
            price_return,
            price_return,
            1.0 + price_return,
            abs(price_return),
            abs(price_return) / 2.0,
            0.8 if label else -0.8,
        ),
        label_return=price_return,
        label=label,
        label_start_date=START + timedelta(days=date_index + 1),
        label_end_date=START + timedelta(days=date_index + 2),
    )


class StrongBaselinesTest(unittest.TestCase):
    def test_all_required_baselines_emit_raw_scores_and_metrics(self) -> None:
        samples = tuple(_sample(index) for index in range(20))
        train = samples[:12]
        test = samples[12:]

        for baseline_name in BASELINE_NAMES:
            result = run_strong_baseline(
                baseline_name,
                train,
                test,
                seed=3,
                top_k=3,
                epochs=20,
                learning_rate=0.05,
            )

            self.assertEqual(result.baseline_name, baseline_name)
            self.assertEqual(result.seed, 3)
            self.assertEqual(len(result.scores), len(test))
            self.assertEqual(result.metrics.sample_count, len(test))
            self.assertIsNotNone(result.metrics.auc)
            self.assertIn("rank_ic", result.metrics.to_dict())
            self.assertIn("top_k_mean_return", result.to_dict())
            self.assertEqual(
                [score.sample_id for score in result.scores],
                [sample.sample_id for sample in test],
            )

    def test_same_seed_is_deterministic_without_best_seed_selection(self) -> None:
        samples = tuple(_sample(index) for index in range(20))

        first = run_strong_baseline(
            "event_seat_profile_logistic",
            samples[:12],
            samples[12:],
            seed=7,
            epochs=30,
        )
        second = run_strong_baseline(
            "event_seat_profile_logistic",
            samples[:12],
            samples[12:],
            seed=7,
            epochs=30,
        )

        self.assertEqual(first.to_dict(), second.to_dict())

    def test_top_k_mean_return_is_cross_sectional_by_as_of_date(self) -> None:
        train = tuple(_sample(index) for index in range(6))
        low_day_one = _sample(8, "000001.SZ")
        high_day_one = _sample(9, "000002.SZ", as_of_index=8)
        low_day_two = _sample(10, "000001.SZ")
        high_day_two = _sample(11, "000002.SZ", as_of_index=10)

        result = run_strong_baseline(
            "event_only_logistic",
            train,
            (low_day_one, high_day_one, low_day_two, high_day_two),
            seed=4,
            top_k=1,
            epochs=40,
            learning_rate=0.05,
        )

        expected_daily_mean = (high_day_one.label_return + high_day_two.label_return) / 2.0
        self.assertAlmostEqual(result.top_k_mean_return, expected_daily_mean)

    def test_empty_partitions_fail_closed(self) -> None:
        sample = _sample(1)

        with self.assertRaisesRegex(ValueError, "train_samples must not be empty"):
            run_strong_baseline("naive_event_mean", (), (sample,), seed=0)
        with self.assertRaisesRegex(ValueError, "test_samples must not be empty"):
            run_strong_baseline("naive_event_mean", (sample,), (), seed=0)

    def test_event_seat_profile_baseline_requires_explicit_seat_features(self) -> None:
        sample = _sample(1)
        without_seat = ResearchSample(
            sample_id=sample.sample_id,
            as_of_date=sample.as_of_date,
            stock_id=sample.stock_id,
            feature_names=FEATURE_NAMES,
            features=sample.features[: len(FEATURE_NAMES)],
            label_return=sample.label_return,
            label=sample.label,
            label_start_date=sample.label_start_date,
            label_end_date=sample.label_end_date,
        )

        with self.assertRaisesRegex(ValueError, "requires at least one seat-profile feature"):
            run_strong_baseline(
                "event_seat_profile_logistic",
                (without_seat,),
                (without_seat,),
                seed=0,
                epochs=1,
            )


if __name__ == "__main__":
    unittest.main()
