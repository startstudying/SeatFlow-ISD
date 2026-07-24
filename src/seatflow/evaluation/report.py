from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    model_valid: bool
    metrics: dict[str, object]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model_valid": self.model_valid,
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
        }
