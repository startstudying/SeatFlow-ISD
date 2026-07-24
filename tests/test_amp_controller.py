from __future__ import annotations

import unittest

from seatflow.training.amp import build_amp_controller


class AmpControllerTest(unittest.TestCase):
    def test_cpu_path_uses_full_precision_and_unscaled_training(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = build_amp_controller(torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)

        with controller.autocast():
            loss = model(torch.ones((1, 2))).square().mean()
        controller.backward(loss)
        controller.unscale_(optimizer)
        controller.step(optimizer)

        self.assertFalse(controller.enabled)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_explicitly_disabled_amp_is_a_noop(self) -> None:
        import torch

        controller = build_amp_controller(
            torch.device("cpu"),
            mixed_precision=False,
            preferred_dtype="fp16",
        )

        self.assertFalse(controller.enabled)


if __name__ == "__main__":
    unittest.main()
