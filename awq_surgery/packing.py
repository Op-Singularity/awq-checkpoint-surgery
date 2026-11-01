"""int4 pack-quantized packing, exactly as compressed-tensors stores it.

The layout, verified byte-for-byte against a real AWQ checkpoint:

    a weight [out, in] is quantised in groups of 32 along ``in``
    each group shares one scale (bf16, symmetric, values -8..7)
    8 consecutive values pack into one int32, **low nibble first**:

        word = v0 | v1<<4 | v2<<8 | ... | v7<<28

Negative values are stored as two's complement nibbles: -8 becomes 0x8, -1
becomes 0xF. The packed tensor therefore has shape ``[out, in // 8]`` and dtype
``int32``, and a companion ``weight_scale`` of shape ``[out, in // 32]``.

``verify_packing_against_checkpoint`` in the tests re-packs a real tensor and
compares the bytes; that is the only way to be sure about nibble order.
"""
from __future__ import annotations

import torch

GROUP_SIZE = 32
PACK = 8  # values per int32
QMAX = 8.0  # symmetric int4: scale = max(|w|) / 8 uses the full -8..7 range


def unpack_int4(packed: torch.Tensor, high_first: bool = False) -> torch.Tensor:
    """[out, in//8] int32 -> [out, in] int8 in -8..7."""
    if packed.dim() != 2:
        raise ValueError(f"expected 2-D packed tensor, got {tuple(packed.shape)}")
    word = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    if high_first:
        shifts = torch.flip(shifts, dims=[0])
    nibbles = (word.unsqueeze(-1) >> shifts) & 0xF  # [out, in//8, 8]
    flat = nibbles.reshape(packed.shape[0], -1)
    return torch.where(flat >= 8, flat - 16, flat).to(torch.int8)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """[out, in] int8 in -8..7 -> [out, in//8] int32, low nibble first."""
    if values.dim() != 2:
        raise ValueError(f"expected 2-D tensor, got {tuple(values.shape)}")
    out, inn = values.shape
    if inn % PACK:
        raise ValueError(f"in_features {inn} is not a multiple of {PACK}")
    nibbles = (values.to(torch.int64) & 0xF).reshape(out, inn // PACK, PACK)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return (nibbles << shifts).sum(dim=-1).to(torch.int32)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Rebuild a float weight from packed int4 plus per-group scales."""
    quantised = unpack_int4(packed).to(torch.float32)
    expanded = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    return quantised * expanded[:, : quantised.shape[1]]


def quantize_symmetric(weight: torch.Tensor, group_size: int = GROUP_SIZE, qmax: float = QMAX):
    """[out, in] float -> (packed int32, scales bf16, original shape int64)."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")
    out, inn = weight.shape
    if inn % group_size:
        raise ValueError(f"in_features {inn} is not a multiple of group_size {group_size}")

    groups = weight.to(torch.float32).reshape(out, inn // group_size, group_size)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax / qmax, min=1e-8)
    quantised = torch.clamp(torch.round(groups / scale), -qmax, qmax - 1).to(torch.int8)
    packed = pack_int4(quantised.reshape(out, inn))
    scales = scale.squeeze(-1).to(torch.bfloat16)
    return packed, scales, torch.tensor([out, inn], dtype=torch.int64)


def quant_error(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> float:
    """Relative reconstruction error of a quantise/dequantise round trip."""
    packed, scales, _ = quantize_symmetric(weight, group_size)
    rebuilt = dequantize(packed, scales, group_size)
    return float((rebuilt - weight.to(torch.float32)).norm() / weight.to(torch.float32).norm())
