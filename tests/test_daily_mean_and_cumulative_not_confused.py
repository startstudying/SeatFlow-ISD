from __future__ import annotations

import unittest

from seatflow.evaluation.return_units import ReturnAggregationMethod, ReturnUnit, return_units


class ReturnUnitsAggregationTest(unittest.TestCase):
    def test_daily_mean_vs_sum_vs_compound_are_not_confused(self) -> None:
        values = [0.01, 0.02]
        mean_payload = return_units(
            values,
            unit=ReturnUnit.DECIMAL,
            aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
        )
        sum_payload = return_units(
            values,
            unit=ReturnUnit.DECIMAL,
            aggregation_method=ReturnAggregationMethod.SUM,
        )
        compound_payload = return_units(
            values,
            unit=ReturnUnit.DECIMAL,
            aggregation_method=ReturnAggregationMethod.COMPOUND,
        )

        self.assertAlmostEqual(mean_payload.decimal, 0.015)
        self.assertAlmostEqual(sum_payload.decimal, 0.03)
        self.assertAlmostEqual(compound_payload.decimal, 0.0302)
        self.assertGreater(compound_payload.decimal, sum_payload.decimal)
        self.assertNotEqual(mean_payload.decimal, sum_payload.decimal)


if __name__ == "__main__":
    unittest.main()
