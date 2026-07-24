import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seatflow.config import load_config


class ConfigTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config(Path("configs/default.toml"))

        self.assertEqual(config.name, "dt-seathgt")
        self.assertEqual(config.graph.prediction_horizons, (1, 3, 5))
        self.assertGreater(config.model.hidden_dim, 0)
        self.assertEqual(config.data.provider, "sample")
        self.assertLess(config.training.train_end, config.training.valid_start)

    def test_rejects_data_paths_with_different_parents(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.toml"
            path.write_text(
                """
[project]
name = "bad"
seed = 1
[data]
raw_dir = "one/raw"
processed_dir = "two/processed"
lookback_days = 1
provider = "sample"
available_after_close = "18:00"
[graph]
decay_lambda = 0.1
max_edge_age_days = 1
prediction_horizons = [1]
[model]
hidden_dim = 1
num_heads = 1
num_layers = 1
dropout = 0.0
seat_profile_types = 1
[training]
train_start = "2021-01-01"
train_end = "2021-01-02"
valid_start = "2021-01-03"
valid_end = "2021-01-04"
test_start = "2021-01-05"
test_end = "2021-01-06"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "share the same parent"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
