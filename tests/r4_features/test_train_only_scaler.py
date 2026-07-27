from __future__ import annotations

import unittest

from seatflow.r4_features import (
    MISSING_SEAT_EVENT_FEATURE,
    SEAT_AGGREGATE_FEATURES,
    SeatAggregateFeatureError,
    TrainOnlyStandardScaler,
)


class R4TrainOnlyScalerTest(unittest.TestCase):
    def test_fit_and_transform_are_separate_and_training_only(self) -> None:
        scaler = TrainOnlyStandardScaler()
        train = (_row(100.0, 0.0), _row(300.0, 1.0))
        test = (_row(1_000_000.0, 0.0),)

        with self.assertRaisesRegex(RuntimeError, "before transform"):
            scaler.transform(test)

        scaler.fit(train)
        transformed_train = scaler.transform(train)
        transformed_test = scaler.transform(test)

        self.assertAlmostEqual(transformed_train[0]["seat_buy_amount"], -1.0)
        self.assertAlmostEqual(transformed_train[1]["seat_buy_amount"], 1.0)
        self.assertAlmostEqual(transformed_test[0]["seat_buy_amount"], 9998.0)

        with self.assertRaisesRegex(RuntimeError, "already been fitted"):
            scaler.fit(test)

    def test_double_transform_and_schema_mismatches_fail_closed(self) -> None:
        scaler = TrainOnlyStandardScaler()
        rows = (_row(100.0, 0.0), _row(300.0, 1.0))
        scaler.fit(rows)
        transformed = scaler.transform(rows)

        with self.assertRaisesRegex(RuntimeError, "already been transformed"):
            scaler.transform(transformed)

        extra_hash = dict(rows[0])
        extra_hash["seat_id_hash"] = 1.0
        with self.assertRaisesRegex(SeatAggregateFeatureError, "schema"):
            scaler.transform((extra_hash,))

        bad_marker = dict(rows[0])
        bad_marker[MISSING_SEAT_EVENT_FEATURE] = 0.5
        with self.assertRaisesRegex(SeatAggregateFeatureError, "0/1 marker"):
            scaler.transform((bad_marker,))


def _row(seat_buy_amount: float, missing_marker: float) -> dict[str, float]:
    row = {feature: 0.0 for feature in SEAT_AGGREGATE_FEATURES}
    row["seat_buy_amount"] = seat_buy_amount
    row["seat_net_amount"] = seat_buy_amount
    row["seat_count"] = 1.0 if missing_marker == 0.0 else 0.0
    row[MISSING_SEAT_EVENT_FEATURE] = missing_marker
    return row


if __name__ == "__main__":
    unittest.main()
