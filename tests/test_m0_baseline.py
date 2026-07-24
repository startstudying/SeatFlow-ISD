from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class M0BaselineTest(unittest.TestCase):
    def test_mlflow_tracking_extra_is_optional(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]

        self.assertIn("tracking", extras)
        self.assertTrue(
            any(requirement.startswith("mlflow>=") for requirement in extras["tracking"])
        )
        self.assertNotIn("mlflow", "\n".join(pyproject["project"].get("dependencies", [])))

    def test_environment_matrix_documents_optional_pyg_and_quality_gate(self) -> None:
        matrix = Path("docs/environment-matrix.md").read_text(encoding="utf-8")

        self.assertIn("seatflow[ml]", matrix)
        self.assertIn("seatflow[tracking]", matrix)
        self.assertIn("scripts\\quality.ps1", matrix)
        self.assertIn("Sample/synthetic preflight output is engineering smoke only.", matrix)

    def test_gitignore_excludes_sensitive_and_generated_outputs(self) -> None:
        ignore = Path(".gitignore").read_text(encoding="utf-8")

        for pattern in (
            ".env.*",
            "secrets/",
            "tokens/",
            "data/raw/*",
            "data/processed/*",
            "artifacts/*",
            "checkpoints/*",
            "logs/*",
            "mlruns/*",
        ):
            self.assertIn(pattern, ignore)

    def test_quality_and_doctor_scripts_exist(self) -> None:
        quality = Path("scripts/quality.ps1")
        doctor = Path("scripts/doctor.ps1")

        self.assertTrue(quality.exists())
        self.assertTrue(doctor.exists())
        self.assertIn("mypy src tests", quality.read_text(encoding="utf-8"))
        self.assertIn("pyg_cpu_smoke", doctor.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
