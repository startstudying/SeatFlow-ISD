"""Mixed-precision control shared by R2 training and inference."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from seatflow.runtime.device import resolve_amp_dtype


@dataclass
class AmpController:
    device: Any
    dtype: Any | None
    scaler: Any

    @property
    def enabled(self) -> bool:
        return self.dtype is not None

    def autocast(self) -> Any:
        if not self.enabled:
            return nullcontext()
        import torch

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=True,
        )

    def backward(self, loss: Any) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: Any) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: Any) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()


def build_amp_controller(
    device: Any,
    *,
    mixed_precision: bool = True,
    preferred_dtype: str = "fp16",
) -> AmpController:
    import torch

    dtype = resolve_amp_dtype(
        preferred_dtype,
        device=device,
        mixed_precision=mixed_precision,
    )
    fp16_enabled = dtype == torch.float16
    try:
        scaler_type = torch.amp.GradScaler  # type: ignore[attr-defined,unused-ignore]
        scaler = scaler_type("cuda", enabled=fp16_enabled, init_scale=1024.0)
    except (AttributeError, TypeError):  # pragma: no cover - compatibility with older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=fp16_enabled, init_scale=1024.0)
    return AmpController(device=torch.device(device), dtype=dtype, scaler=scaler)
