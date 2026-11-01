"""End-to-end test of the `inspect` command on a checkpoint built on the fly.

No download, no GPU: the fixture writes a two-tensor compressed-tensors-style
checkpoint to a temporary directory and the CLI has to describe it correctly.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def make_checkpoint(root: Path) -> None:
    """Minimal but structurally realistic AWQ checkpoint."""
    out, inn = 64, 128
    # one quantised linear: packed int4 + per-group bf16 scales + original shape
    packed = torch.zeros((out, inn // 8), dtype=torch.int32)
    scales = torch.zeros((out, inn // 32), dtype=torch.bfloat16)
    shape = torch.tensor([out, inn], dtype=torch.int64)

    tensors = {
        "model.layers.0.mlp.down_proj.weight_packed": packed,
        "model.layers.0.mlp.down_proj.weight_scale": scales,
        "model.layers.0.mlp.down_proj.weight_shape": shape,
        # vocabulary kept in BF16, which is the point of the whole tool
        "lm_head.weight": torch.zeros((256, inn), dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.zeros((256, inn), dtype=torch.bfloat16),
        # a vision tower that the tool should recognise and report separately
        "model.visual.blocks.0.attn.proj.weight": torch.zeros((16, 16), dtype=torch.bfloat16),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})

    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "vision_config": {"depth": 1},
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "strategy": "group",
                        "symmetric": True,
                    },
                }
            },
            "ignore": ["model.visual.blocks.0.attn.proj", "lm_head"],
        },
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))


def test_inspect_reports_the_composition(tmp_path: Path) -> None:
    make_checkpoint(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "awq_surgery.surgery", "inspect", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout

    # the four groups the tool distinguishes
    assert "body (quantised)" in out
    assert "lm_head" in out
    assert "embedding" in out
    assert "vision" in out

    # and the conclusion a reader needs: the vocabulary is not quantised
    assert "vocabulary ignored: True" in out
    assert "4-bit group 32" in out
    assert "tied embeds  : False" in out


def test_inspect_help_does_not_need_a_checkpoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "awq_surgery.surgery", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "inspect" in result.stdout
    assert "strip" in result.stdout
