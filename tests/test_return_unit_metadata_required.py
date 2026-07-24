from __future__ import annotations

import unittest

from seatflow.evaluation.return_units import ReturnAggregationMethod, ReturnUnit, return_units


class ReturnUnitMetadataRequiredTest(unittest.TestCase):
    def test_return_units_include_all_required_fields(self) -> None:
        payload = return_units(
            (0.01, -0.005, 0.003),
            unit=ReturnUnit.DECIMAL,
            aggregation_method=ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN,
        )
        row = payload.to_dict()

        self.assertEqual(
            set(row.keys()),
            {
                "decimal",
                "percent",
                "bps",
                "aggregation_method",
                "number_of_days",
            },
        )
        self.assertEqual(row["number_of_days"], 3)
        self.assertEqual(
            row["aggregation_method"],
            ReturnAggregationMethod.DAILY_ARITHMETIC_MEAN.value,
        )


if __name__ == "__main__":
    unittest.main()
