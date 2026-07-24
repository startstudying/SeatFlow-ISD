from __future__ import annotations

from dataclasses import asdict, dataclass

from seatflow.graph.types import HeteroGraphSnapshot
from seatflow.models.contracts import ModelInput, PredictionResult
from seatflow.services.predictor import HeuristicSmokePredictor


@dataclass(frozen=True)
class EngineeringSmokeBaseline:
    horizons: tuple[int, ...]

    @property
    def name(self) -> str:
        return "engineering_smoke_flow_baseline"

    def predict(self, graph: HeteroGraphSnapshot, target_stock_id: str) -> PredictionResult:
        predictor = HeuristicSmokePredictor(self.horizons)
        return predictor.predict(ModelInput(graph=graph, target_stock_id=target_stock_id))

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_name": self.name,
            "horizons": list(self.horizons),
            "model_valid": False,
            "risk_flags": [
                "engineering_smoke_only",
                "synthetic_sample_data_only",
                "not_for_research_claims_or_trading",
            ],
        }


def prediction_to_dict(prediction: PredictionResult) -> dict[str, object]:
    return asdict(prediction)
