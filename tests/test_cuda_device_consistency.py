from __future__ import annotations

import unittest

from seatflow.runtime.device import (
    move_graph_to_device,
    report_cuda_memory,
    resolve_amp_dtype,
    resolve_device,
)


class DeviceConsistencyTest(unittest.TestCase):
    def test_auto_device_and_graph_are_consistent(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        device = resolve_device("auto")
        graph = torch.tensor([1.0])
        moved = move_graph_to_device(graph, device)

        self.assertEqual(moved.device.type, device.type)
        self.assertIsNone(resolve_amp_dtype(device=torch.device("cpu")))

    def test_cpu_memory_report_is_zero(self) -> None:
        report = report_cuda_memory("cpu")

        self.assertFalse(report.available)
        self.assertEqual(report.allocated_bytes, 0)
        self.assertEqual(report.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
