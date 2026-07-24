from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from seatflow.runtime_env import load_workspace_env


class RuntimeEnvTest(unittest.TestCase):
    def test_loads_allowed_provider_settings_without_overriding_process_env(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / ".env"
            path.write_text(
                """
# Local developer settings.
TUSHARE_TOKEN=rotated-token
TUSHARE_ALLOW_INSECURE_HTTP=1
SEATFLOW_DATA_PROVIDER=akshare
UNRELATED_VALUE=ignored
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TUSHARE_TOKEN": "process-token"}, clear=True):
                loaded = load_workspace_env(path)

                self.assertEqual(os.environ["TUSHARE_TOKEN"], "process-token")
                self.assertEqual(os.environ["TUSHARE_ALLOW_INSECURE_HTTP"], "1")
                self.assertEqual(os.environ["SEATFLOW_DATA_PROVIDER"], "akshare")
                self.assertNotIn("UNRELATED_VALUE", os.environ)
                self.assertEqual(
                    loaded, ("TUSHARE_ALLOW_INSECURE_HTTP", "SEATFLOW_DATA_PROVIDER")
                )

    def test_rejects_malformed_allowed_assignment(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / ".env"
            path.write_text("TUSHARE_TOKEN\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                load_workspace_env(path)


if __name__ == "__main__":
    unittest.main()
