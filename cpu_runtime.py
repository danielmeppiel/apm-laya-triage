"""One CPU precision policy shared by single-issue and full-corpus inference."""

from __future__ import annotations

from typing import Any


def accelerate_cpu(agent: Any, precision: str) -> tuple[int, str | None]:
    """Quantize encoder Linear layers only; the fused decision head requires FP32 weights."""
    if agent.device.type != "cpu":
        raise ValueError("This comparison requires an actual CPU runtime.")
    if precision == "fp32":
        return 0, None
    if precision != "int8":
        raise ValueError("Unsupported CPU precision.")
    import torch
    from torch.ao.nn.quantized.dynamic import Linear
    from torch.ao.quantization import quantize_dynamic

    engines = torch.backends.quantized.supported_engines
    engine = next(
        (name for name in ("x86", "fbgemm", "qnnpack") if name in engines), None
    )
    if engine is None:
        raise ValueError("No supported INT8 CPU backend; choose FP32 on this machine.")
    torch.backends.quantized.engine = engine
    quantize_dynamic(
        agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    count = sum(isinstance(module, Linear) for module in agent.model.encoder.modules())
    if not count:
        raise ValueError("INT8 requested but no Linear layers were quantized.")
    return count, engine
