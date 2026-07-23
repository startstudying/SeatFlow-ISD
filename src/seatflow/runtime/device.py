"""Centralized PyTorch device placement and CUDA memory management."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CudaMemoryReport:
    device: str
    available: bool
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_dict(self) -> dict[str, str | bool | int]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal deployments
        raise RuntimeError("PyTorch is required for device management") from exc
    return torch


def resolve_device(requested: str = "auto") -> Any:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    torch = _torch()
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_amp_dtype(
    preferred_dtype: str = "fp16",
    *,
    device: Any | None = None,
    mixed_precision: bool = True,
) -> Any | None:
    """Return a supported autocast dtype, or ``None`` when AMP is disabled."""

    if not mixed_precision:
        return None
    torch = _torch()
    resolved_device = resolve_device() if device is None else torch.device(device)
    if resolved_device.type != "cuda":
        return None
    normalized = preferred_dtype.strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested but is not supported by this CUDA device")
        return torch.bfloat16
    raise ValueError(f"unsupported mixed-precision dtype: {preferred_dtype}")


def move_graph_to_device(graph: Any, device: Any, *, non_blocking: bool = False) -> Any:
    """Move a graph/tensor container to one device through its standard ``to`` API."""

    torch = _torch()
    resolved_device = torch.device(device)
    to_method = getattr(graph, "to", None)
    if not callable(to_method):
        raise TypeError("graph object must provide a callable to(device) method")
    try:
        return to_method(resolved_device, non_blocking=non_blocking)
    except TypeError:
        return to_method(resolved_device)


def report_cuda_memory(device: Any | None = None, *, reset_peak: bool = False) -> CudaMemoryReport:
    """Capture current and peak CUDA allocator usage without synchronizing training."""

    torch = _torch()
    resolved_device = resolve_device() if device is None else torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        return CudaMemoryReport(str(resolved_device), False, 0, 0, 0, 0)
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(resolved_device)
    return CudaMemoryReport(
        device=str(resolved_device),
        available=True,
        allocated_bytes=int(torch.cuda.memory_allocated(resolved_device)),
        reserved_bytes=int(torch.cuda.memory_reserved(resolved_device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(resolved_device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(resolved_device)),
    )


def release_cuda_objects(*objects: Any, device: Any | None = None) -> CudaMemoryReport:
    """Release owned references, collect Python garbage, and empty the CUDA cache."""

    del objects
    gc.collect()
    torch = _torch()
    resolved_device = resolve_device() if device is None else torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report_cuda_memory(resolved_device)
