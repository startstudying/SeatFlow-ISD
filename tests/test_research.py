import unittest
from datetime import date, timedelta
from decimal import Decimal

from seatflow.data.contracts import DailyBar, DragonTigerSummary
from seatflow.research import (
    FEATURE_NAMES,
    build_research_samples,
    evaluate_predictions,
    predict_proba,
    samples_to_matrix,
    time_split_samples,
    train_l2_logistic,
)

START = date(2026, 1, 1)


def _day(offset: int) -> date:
    return START + timedelta(days=offset)


def _bar(offset: int, stock_id: str = "000001.SZ", open_delta: str = "0") -> DailyBar:
    open_price = Decimal("10") + Decimal(offset) + Decimal(open_delta)
    close_price = open_price + Decimal("0.5")
    return DailyBar(
        trade_date=_day(offset),
        stock_id=stock_id,
        open_price=open_price,
        high_price=close_price + Decimal("0.5"),
        low_price=open_price - Decimal("0.5"),
        close_price=close_price,
        volume=Decimal("1000") + Decimal(offset * 10),
        amount=Decimal("10000") + Decimal(offset * 100),
        source_name="fixture",
    )


def _event(offset: int, stock_id: str = "000001.SZ", net: str = "25") -> DragonTigerSummary:
    return DragonTigerSummary(
        trade_date=_day(offset),
        stock_id=stock_id,
        stock_name="fixture",
        buy_amount=Decimal("100"),
        sell_amount=Decimal("75"),
        net_amount=Decimal(net),
        reason="fixture",
        source_name="fixture",
    )


class ResearchCoreTest(unittest.TestCase):
    def test_features_use_as_of_and_trailing_data_only(self) -> None:
        bars = tuple(_bar(offset) for offset in range(25))
        samples = build_research_samples((_event(20),), bars)

        changed_future = list(bars)
        changed_future[21] = _bar(21, open_delta="1000")
        changed_future[22] = _bar(22, open_delta="2000")
        changed_samples = build_research_samples((_event(20),), tuple(changed_future))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].feature_names, FEATURE_NAMES)
        self.assertEqual(samples[0].features, changed_samples[0].features)
        self.assertNotEqual(samples[0].label_return, changed_samples[0].label_return)

    def test_label_is_next_open_to_second_next_open(self) -> None:
        bars = tuple(_bar(offset) for offset in range(24))

        sample = build_research_samples((_event(20),), bars)[0]

        expected = float(bars[22].open_price / bars[21].open_price - Decimal("1"))
        self.assertEqual(sample.label_start_date, bars[21].trade_date)
        self.assertEqual(sample.label_end_date, bars[22].trade_date)
        self.assertAlmostEqual(sample.label_return, expected)

    def test_requires_twenty_day_lookback(self) -> None:
        bars = tuple(_bar(offset) for offset in range(22))

        samples = build_research_samples((_event(18),), bars)

        self.assertEqual(samples, ())

    def test_same_seed_reproduces_training_and_predictions(self) -> None:
        samples = build_research_samples(
            tuple(_event(offset) for offset in range(20, 36)), _bars(40)
        )
        dataset = samples_to_matrix(samples)

        model_a = train_l2_logistic(dataset, seed=7, epochs=20, learning_rate=0.05)
        model_b = train_l2_logistic(dataset, seed=7, epochs=20, learning_rate=0.05)
        probabilities_a = predict_proba(model_a, dataset.x)
        probabilities_b = predict_proba(model_b, dataset.x)

        self.assertEqual(model_a.weights, model_b.weights)
        self.assertEqual(model_a.mean, model_b.mean)
        self.assertEqual(probabilities_a, probabilities_b)

    def test_time_split_uses_as_of_date_order(self) -> None:
        samples = build_research_samples(
            tuple(_event(offset) for offset in range(20, 36)), _bars(40)
        )

        split = time_split_samples(samples, train_end=_day(24), valid_end=_day(30))

        self.assertTrue(all(sample.as_of_date <= _day(24) for sample in split.train))
        self.assertTrue(all(_day(24) < sample.as_of_date <= _day(30) for sample in split.valid))
        self.assertTrue(all(sample.as_of_date > _day(30) for sample in split.test))
        self.assertEqual(len(split.train) + len(split.valid) + len(split.test), len(samples))

    def test_time_split_purges_and_embargoes_boundary_samples(self) -> None:
        samples = build_research_samples(
            tuple(_event(offset) for offset in range(20, 45)), _bars(50)
        )

        split = time_split_samples(
            samples,
            train_end=_day(28),
            valid_end=_day(36),
            purge_days=2,
            embargo_days=2,
        )

        self.assertTrue(all(sample.as_of_date <= _day(26) for sample in split.train))
        self.assertTrue(all(sample.label_end_date <= _day(28) for sample in split.train))
        self.assertTrue(all(_day(30) < sample.as_of_date <= _day(34) for sample in split.valid))
        self.assertTrue(all(sample.label_end_date <= _day(36) for sample in split.valid))
        self.assertTrue(all(sample.as_of_date > _day(38) for sample in split.test))

    def test_metrics_are_json_serializable_and_include_rank_ic_auc(self) -> None:
        metrics = evaluate_predictions(
            probabilities=(0.1, 0.9, 0.8, 0.2),
            labels=(0, 1, 1, 0),
            label_returns=(-0.02, 0.04, 0.03, -0.01),
        )

        self.assertEqual(metrics.to_dict()["accuracy"], 1.0)
        self.assertEqual(metrics.to_dict()["auc"], 1.0)
        self.assertEqual(metrics.to_dict()["rank_ic"], 1.0)


def _bars(count: int) -> tuple[DailyBar, ...]:
    return tuple(_bar(offset) for offset in range(count))


if __name__ == "__main__":
    unittest.main()
