from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from seatflow.labels import (
    CandidateResidualLabel,
    build_cross_sectional_ranking_labels,
)


class CrossSectionalRankingLabelsTest(unittest.TestCase):
    def test_statistics_are_date_local_and_rank_is_monotonic(self) -> None:
        labels = tuple(
            CandidateResidualLabel(
                as_of_date=day,
                target_stock_id=f"{index:06d}.SZ",
                horizon=1,
                stock_return=Decimal(str(value)),
                candidate_pool_return=Decimal("0"),
                residual_return=Decimal(str(value)),
                direction_label=1 if value > 0 else -1,
            )
            for day, values in (
                (date(2026, 1, 1), [-100, -1, 0, 1, 100]),
                (date(2026, 1, 2), [900, 999, 1000, 1001, 1100]),
            )
            for index, value in enumerate(values)
        )

        result = build_cross_sectional_ranking_labels(labels)
        first = [row for row in result if row.as_of_date == date(2026, 1, 1)]
        second = [row for row in result if row.as_of_date == date(2026, 1, 2)]

        self.assertAlmostEqual(sum(row.residual_zscore for row in first), 0.0)
        self.assertAlmostEqual(sum(row.residual_zscore for row in second), 0.0)
        self.assertEqual(
            [row.residual_rank_percentile for row in first],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
        self.assertEqual([row.direction_target for row in first], [0, 0, 0, 1, 1])

    def test_constant_date_has_neutral_targets(self) -> None:
        labels = tuple(
            CandidateResidualLabel(
                as_of_date=date(2026, 1, 1),
                target_stock_id=f"{index:06d}.SZ",
                horizon=1,
                stock_return=Decimal("1"),
                candidate_pool_return=Decimal("0"),
                residual_return=Decimal("1"),
                direction_label=1,
            )
            for index in range(3)
        )
        result = build_cross_sectional_ranking_labels(labels)
        self.assertTrue(all(row.residual_zscore == 0.0 for row in result))
        self.assertTrue(all(row.residual_rank_percentile == 0.5 for row in result))


if __name__ == "__main__":
    unittest.main()

