from __future__ import annotations

import unittest

from seatflow.evaluation.realistic_costs import (
    RealisticCostComponents,
    effective_side_bps,
    split_turnover_costs,
)


class RealisticCostComponentsTest(unittest.TestCase):
    def test_effective_side_bps_matches_default_breakdown(self) -> None:
        buy_bps, sell_bps = effective_side_bps()
        self.assertAlmostEqual(buy_bps, 8.0)
        self.assertAlmostEqual(sell_bps, 18.0)

    def test_split_turnover_costs_decomposes(self) -> None:
        components = split_turnover_costs(buy_turnover=1.0, sell_turnover=1.0)
        self.assertIsInstance(components, RealisticCostComponents)
        self.assertAlmostEqual(components.turnover, 1.0)
        self.assertAlmostEqual(components.buy_turnover, 1.0)
        self.assertAlmostEqual(components.sell_turnover, 1.0)
        self.assertGreaterEqual(components.commission, 60.0)
        self.assertAlmostEqual(
            components.total,
            components.commission
            + components.stamp
            + components.slippage
            + components.min_commission,
        )

    def test_cost_scales_with_turnover_multiplier(self) -> None:
        base = split_turnover_costs(buy_turnover=0.1, sell_turnover=0.2)
        boosted = split_turnover_costs(buy_turnover=0.2, sell_turnover=0.4)
        self.assertAlmostEqual(boosted.total, base.total * 2.0)

    def test_minimum_commission_only_when_turnover_exists(self) -> None:
        no_turnover = split_turnover_costs(buy_turnover=0.0, sell_turnover=0.0)
        with_turnover = split_turnover_costs(buy_turnover=0.01, sell_turnover=0.0)
        self.assertEqual(no_turnover.total, 0.0)
        self.assertGreater(with_turnover.min_commission, 0.0)


if __name__ == "__main__":
    unittest.main()
