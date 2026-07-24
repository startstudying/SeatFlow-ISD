import unittest
from datetime import date

from seatflow.data.records import DragonTigerRecord, TradeSide
from seatflow.graph.builder import DynamicGraphBuilder
from seatflow.models.contracts import ModelInput
from seatflow.services.predictor import HeuristicSmokePredictor


class HeuristicSmokePredictorTest(unittest.TestCase):
    def test_buy_dominance_produces_probability_above_half(self) -> None:
        as_of_date = date(2026, 1, 15)
        records = (
            DragonTigerRecord(
                trade_date=as_of_date,
                seat_id="buyer",
                stock_id="600000.SH",
                side=TradeSide.BUY,
                amount=200.0,
            ),
            DragonTigerRecord(
                trade_date=as_of_date,
                seat_id="seller",
                stock_id="600000.SH",
                side=TradeSide.SELL,
                amount=50.0,
            ),
        )
        graph = DynamicGraphBuilder(0.1, 20).build(records, as_of_date)
        predictor = HeuristicSmokePredictor((1, 3, 5))

        result = predictor.predict(ModelInput(graph, "600000.SH"))

        self.assertGreater(result.rise_probability, 0.5)
        self.assertEqual(set(result.expected_returns), {1, 3, 5})
        self.assertIn("smoke_baseline_not_for_trading", result.risk_flags)


if __name__ == "__main__":
    unittest.main()
