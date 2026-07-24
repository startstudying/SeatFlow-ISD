from __future__ import annotations

import unittest

from seatflow.evaluation.cost_aware_strategy import (
    simulate_p0,
    simulate_p1,
    simulate_p2,
    simulate_p3,
)
from seatflow.evaluation.realistic_costs import RealisticCost


class CostAwareStrategyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = {
            "2026-01-01": [
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "A",
                    "score": 9.0,
                    "expected_return": 0.06,
                },
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "B",
                    "score": 8.0,
                    "expected_return": 0.03,
                },
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "C",
                    "score": 7.0,
                    "expected_return": 0.05,
                },
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "D",
                    "score": 6.0,
                    "expected_return": 0.01,
                },
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "E",
                    "score": 5.0,
                    "expected_return": -0.01,
                },
                {
                    "prediction_date": "2026-01-01",
                    "stock_id": "F",
                    "score": 4.0,
                    "expected_return": -0.02,
                },
            ],
            "2026-01-02": [
                {
                    "prediction_date": "2026-01-02",
                    "stock_id": "C",
                    "score": 10.0,
                    "expected_return": 0.12,
                },
                {
                    "prediction_date": "2026-01-02",
                    "stock_id": "A",
                    "score": 9.2,
                    "expected_return": 0.07,
                },
                {
                    "prediction_date": "2026-01-02",
                    "stock_id": "B",
                    "score": 8.4,
                    "expected_return": 0.02,
                },
                {
                    "prediction_date": "2026-01-02",
                    "stock_id": "F",
                    "score": 7.1,
                    "expected_return": 0.0,
                },
                {
                    "prediction_date": "2026-01-02",
                    "stock_id": "D",
                    "score": 6.4,
                    "expected_return": 0.03,
                },
            ],
            "2026-01-03": [
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "C",
                    "score": 9.5,
                    "expected_return": 0.09,
                },
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "A",
                    "score": 8.6,
                    "expected_return": 0.08,
                },
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "B",
                    "score": 7.9,
                    "expected_return": 0.02,
                },
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "D",
                    "score": 6.8,
                    "expected_return": 0.03,
                },
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "F",
                    "score": 4.0,
                    "expected_return": 0.01,
                },
                {
                    "prediction_date": "2026-01-03",
                    "stock_id": "E",
                    "score": 3.0,
                    "expected_return": 0.04,
                },
            ],
        }

    def test_p0_is_strict_top5_ranking_each_day(self) -> None:
        transitions = simulate_p0(self.rows, top_k=2)
        self.assertEqual(transitions[0].candidate, ("A", "B"))
        self.assertEqual(transitions[1].candidate, ("C", "A"))
        self.assertEqual(transitions[2].candidate, ("C", "A"))

    def test_p1_forced_exit_and_single_voluntary_replacement(self) -> None:
        rows = {
            "d1": [
                {"prediction_date": "d1", "stock_id": "A", "score": 3.0, "expected_return": 0.1},
                {"prediction_date": "d1", "stock_id": "B", "score": 2.0, "expected_return": 0.1},
                {"prediction_date": "d1", "stock_id": "C", "score": 1.0, "expected_return": 0.1},
            ],
            "d2": [
                {"prediction_date": "d2", "stock_id": "A", "score": 5.0, "expected_return": 0.2},
                {"prediction_date": "d2", "stock_id": "D", "score": 4.5, "expected_return": 0.2},
                {"prediction_date": "d2", "stock_id": "E", "score": 4.0, "expected_return": 0.2},
            ],
            "d3": [
                {"prediction_date": "d3", "stock_id": "A", "score": 5.1, "expected_return": 0.2},
                {"prediction_date": "d3", "stock_id": "D", "score": 4.8, "expected_return": 0.2},
                {"prediction_date": "d3", "stock_id": "F", "score": 4.0, "expected_return": 0.2},
            ],
        }
        transitions = simulate_p1(rows, top_k=2, candidate_exit_rule="A")
        candidate_sizes = [len(t.candidate) for t in transitions]
        self.assertEqual(candidate_sizes, [2, 2, 2])
        self.assertEqual(transitions[1].candidate, ("A", "D"))
        self.assertEqual(transitions[2].candidate, ("A", "D"))

    def test_p2_holds_within_top10_universe(self) -> None:
        transitions = simulate_p2(self.rows, buy_rank=2, hold_rank=4)
        self.assertEqual(transitions[1].path, "p2")
        self.assertEqual(transitions[1].candidate, ("A", "B"))
        self.assertEqual(transitions[0].candidate, ("A", "B"))

    def test_p3_switches_only_if_net_improves(self) -> None:
        expensive: dict[str, list[dict[str, object]]] = {
            "d1": [
                {"prediction_date": "d1", "stock_id": "A", "score": 1.0, "expected_return": 0.10},
                {"prediction_date": "d1", "stock_id": "B", "score": 0.9, "expected_return": 0.01},
                {"prediction_date": "d1", "stock_id": "C", "score": 0.8, "expected_return": 0.01},
            ],
            "d2": [
                {"prediction_date": "d2", "stock_id": "C", "score": 5.0, "expected_return": 0.10},
                {"prediction_date": "d2", "stock_id": "D", "score": 4.8, "expected_return": 0.09},
                {"prediction_date": "d2", "stock_id": "A", "score": 4.7, "expected_return": 0.10},
                {"prediction_date": "d2", "stock_id": "E", "score": 4.0, "expected_return": 0.01},
            ],
        }
        transitions = simulate_p3(
            expensive,
            buy_rank=2,
            hold_rank=3,
            switch_margin=0.5,
            cost=RealisticCost(commission=300.0),
        )
        self.assertEqual(transitions[0].candidate, ("A", "B"))
        self.assertEqual(transitions[1].reason, "no_switch")
        self.assertEqual(transitions[1].candidate, ("A", "C"))


if __name__ == "__main__":
    unittest.main()
