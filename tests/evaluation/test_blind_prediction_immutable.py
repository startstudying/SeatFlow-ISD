from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.evaluation.blind import append_blind_prediction


class BlindPredictionImmutableTest(unittest.TestCase):
    def test_existing_prediction_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            first = _append(path, "000001.SZ")
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                _append(path, "000001.SZ")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(first["file_sha256"]), 64)

    def test_new_stock_appends_without_changing_existing_prefix(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.jsonl"
            _append(path, "000001.SZ")
            prefix = path.read_bytes()
            _append(path, "000002.SZ")

            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)


def _append(path: Path, stock_id: str) -> dict[str, object]:
    return append_blind_prediction(
        path,
        prediction_date="2026-08-01",
        stock_id=stock_id,
        predicted_rank=1.0,
        predicted_residual_return=0.01,
        model_hash="a" * 64,
        config_hash="b" * 64,
        data_cutoff="2026-07-31",
        generated_at="2026-08-01T18:00:00+08:00",
    )


if __name__ == "__main__":
    unittest.main()
