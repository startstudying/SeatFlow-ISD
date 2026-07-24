from __future__ import annotations

import unittest

from seatflow.evaluation.return_units import ReturnAggregationMethod, ReturnUnit, return_units


class ReturnUnitsBasisPointConversionTest(unittest.TestCase):
    def test_percent_and_bps_metadata_are_deterministic(self) -> None:
        payload = return_units(
            (100.0,),
            unit=ReturnUnit.BPS,
            aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
        )

        self.assertAlmostEqual(payload.decimal, 0.01)
        self.assertAlmostEqual(payload.percent, 1.0)
        self.assertAlmostEqual(payload.bps, 100.0)

        payload2 = return_units(
            (1.0,),
            unit=ReturnUnit.PERCENT,
            aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
        )
        self.assertAlmostEqual(payload2.decimal, 0.01)
        self.assertAlmostEqual(payload2.percent, 1.0)


if __name__ == '__main__':
    unittest.main()
