"""Tests for the int4 packing, including a byte-exact check against a real checkpoint.

The unit tests need only torch. The checkpoint test is skipped unless you point it
at a 4-bit AWQ model:

    AWQ_CHECKPOINT=~/models/qwen3.5-9b-awq python -m pytest tests/ -v

That test is the important one: it takes a tensor the AWQ tooling wrote, unpacks
it, re-packs it, and demands identical bytes. If the nibble order were wrong it
would fail here instead of producing quietly wrong logits later.
"""
from __future__ import annotations

import os

import pytest
import torch

from awq_surgery.packing import (
    dequantize,
    pack_int4,
    quant_error,
    quantize_symmetric,
    unpack_int4,
)


def test_pack_unpack_round_trip():
    values = torch.randint(-8, 8, (16, 128), dtype=torch.int8)
    packed = pack_int4(values)
    assert packed.dtype == torch.int32
    assert tuple(packed.shape) == (16, 16)
    assert torch.equal(unpack_int4(packed), values)


def test_low_nibble_first():
    """The first value must land in the least significant nibble."""
    values = torch.zeros((1, 8), dtype=torch.int8)
    values[0, 0] = 1
    assert pack_int4(values)[0, 0].item() == 0x00000001

    values = torch.zeros((1, 8), dtype=torch.int8)
    values[0, 7] = 1
    assert pack_int4(values)[0, 0].item() == 0x10000000

    values = torch.zeros((1, 8), dtype=torch.int8)
    values[0, 1] = -1  # 0xF in the second nibble
    assert pack_int4(values)[0, 0].item() == 0x000000F0


def test_negative_range_is_full():
    values = torch.full((1, 32), -8, dtype=torch.int8)
    assert unpack_int4(pack_int4(values)).min().item() == -8

    values = torch.full((1, 32), 7, dtype=torch.int8)
    assert unpack_int4(pack_int4(values)).max().item() == 7


def test_quantize_error_is_small():
    """~9 % relative error is what symmetric int4 on Gaussian data gives.

    The interesting property is not an absolute threshold but that a smaller group
    is better: more scales per tensor, less error.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 256) * 0.02

    coarse = quant_error(weight, group_size=128)
    fine = quant_error(weight, group_size=32)

    assert fine < 0.12, f"int4 group-32 error larger than expected: {fine:.3f}"
    assert fine <= coarse, f"group 32 ({fine:.3f}) should beat group 128 ({coarse:.3f})"


def test_quantize_shapes_and_dtypes():
    weight = torch.randn(32, 128)
    packed, scales, shape = quantize_symmetric(weight)
    assert tuple(packed.shape) == (32, 16)
    assert tuple(scales.shape) == (32, 4)
    assert scales.dtype == torch.bfloat16
    assert shape.tolist() == [32, 128]
    rebuilt = dequantize(packed, scales)
    assert tuple(rebuilt.shape) == (32, 128)


def test_rejects_unaligned_in_features():
    with pytest.raises(ValueError):
        quantize_symmetric(torch.randn(8, 100))  # 100 % 32 != 0


CHECKPOINT = os.environ.get("AWQ_CHECKPOINT")
CHECKPOINT_TENSOR = os.environ.get("AWQ_TENSOR", "")


def _find_quantised_tensor(root: str) -> str | None:
    """First `*.weight_packed` in the checkpoint, or None."""
    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    import json

    names = [n for n in json.load(open(index_path))["weight_map"] if n.endswith(".weight_packed")]
    return sorted(names)[0] if names else None


@pytest.mark.skipif(not CHECKPOINT, reason="set AWQ_CHECKPOINT to a 4-bit checkpoint to run this")
def test_repack_matches_real_checkpoint_bytes():
    import json

    from safetensors import safe_open

    root = os.path.expanduser(CHECKPOINT)
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    name = CHECKPOINT_TENSOR or _find_quantised_tensor(root)
    assert name, "no .weight_packed tensor found"

    with safe_open(os.path.join(root, index["weight_map"][name]), framework="pt") as handle:
        packed = handle.get_tensor(name)

    repacked = pack_int4(unpack_int4(packed))
    assert torch.equal(repacked, packed), (
        f"{name}: re-packing produced different bytes — the nibble order is wrong "
        f"for this checkpoint"
    )
