import unittest

from seatflow.evaluation.ablations import evaluate_ablation_matrix


class AblationsTest(unittest.TestCase):
    def test_required_ablation_groups_pass_when_contribution_is_stable(self) -> None:
        matrix = evaluate_ablation_matrix(
            [
                {
                    "group": group,
                    "fold": f"roll-{index}",
                    "auc_contribution": 0.001,
                    "rank_ic_contribution": 0.002,
                }
                for group in ("seat_profile", "time_edges", "direction_edges")
                for index in range(3)
            ]
        )

        self.assertTrue(matrix.passed)
        self.assertEqual(matrix.blocked_reasons, ())

    def test_missing_required_group_is_negative_ablation_result(self) -> None:
        matrix = evaluate_ablation_matrix(
            [
                {
                    "group": "seat_profile",
                    "fold": "roll-0",
                    "auc_contribution": 0.001,
                    "rank_ic_contribution": 0.002,
                }
            ]
        )

        self.assertFalse(matrix.passed)
        self.assertIn("missing_ablation:time_edges", matrix.blocked_reasons)
        self.assertIn("missing_ablation:direction_edges", matrix.blocked_reasons)

    def test_unstable_direction_edge_contribution_fails_closed(self) -> None:
        matrix = evaluate_ablation_matrix(
            {
                "seat_profile": [
                    {"fold": "a", "auc_contribution": 0.1, "rank_ic_contribution": 0.1},
                    {"fold": "b", "auc_contribution": 0.1, "rank_ic_contribution": 0.1},
                ],
                "time_edges": [
                    {"fold": "a", "auc_contribution": 0.1, "rank_ic_contribution": 0.1},
                    {"fold": "b", "auc_contribution": 0.1, "rank_ic_contribution": 0.1},
                ],
                "direction_edges": [
                    {"fold": "a", "auc_contribution": -0.1, "rank_ic_contribution": 0.1},
                    {"fold": "b", "auc_contribution": 0.1, "rank_ic_contribution": -0.1},
                    {"fold": "c", "auc_contribution": 0.1, "rank_ic_contribution": 0.1},
                ],
            }
        )

        self.assertFalse(matrix.passed)
        self.assertIn("ablation_not_stable:direction_edges", matrix.blocked_reasons)


if __name__ == "__main__":
    unittest.main()
