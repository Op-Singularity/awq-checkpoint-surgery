"""Read a compressed-tensors checkpoint without loading the weights into VRAM.

Everything here works off the safetensors headers and the config, so it is fast
even for a 20 GB model and needs no GPU.
"""
from __future__ import annotations

import json
import os
import struct
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int
    shard: str


def read_shard_header(path: str) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def shard_files(root: str) -> list[str]:
    index_path = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        return sorted(set(index["weight_map"].values()))
    return sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))


def scan(root: str) -> dict[str, TensorInfo]:
    out: dict[str, TensorInfo] = {}
    for shard in shard_files(root):
        for name, meta in read_shard_header(os.path.join(root, shard)).items():
            start, end = meta.get("data_offsets", [0, 0])
            out[name] = TensorInfo(name, meta.get("dtype", "?"), meta.get("shape", []), end - start, shard)
    return out


def classify(name: str) -> str:
    low = name.lower()
    if ".visual." in low or low.startswith("visual.") or "vision" in low:
        return "vision"
    if "embed_tokens" in low:
        return "embedding"
    if "lm_head" in low:
        return "lm_head"
    if low.endswith((".weight_packed", ".weight_scale", ".weight_shape", ".qweight", ".scales")):
        return "body (quantised)"
    if "layers." in low or low.startswith("mtp."):
        return "body"
    return "other"


def inventory(root: str) -> tuple[dict[str, list[TensorInfo]], int]:
    tensors = scan(root)
    groups: dict[str, list[TensorInfo]] = defaultdict(list)
    for info in tensors.values():
        groups[classify(info.name)].append(info)
    return dict(groups), sum(t.nbytes for t in tensors.values())


def quant_scheme(root: str) -> dict:
    """The parts of quantization_config that decide what is quantised."""
    config = json.load(open(os.path.join(root, "config.json")))
    quant = config.get("quantization_config") or {}
    groups = {}
    for key, value in (quant.get("config_groups") or {}).items():
        weights = value.get("weights") or {}
        groups[key] = {
            "num_bits": weights.get("num_bits"),
            "group_size": weights.get("group_size"),
            "symmetric": weights.get("symmetric"),
            "strategy": weights.get("strategy"),
            "targets": value.get("targets"),
        }
    return {
        "quant_method": quant.get("quant_method"),
        "format": quant.get("format"),
        "groups": groups,
        "ignore": quant.get("ignore") or [],
        "tied": config.get("tie_word_embeddings"),
        "architectures": config.get("architectures"),
        "has_vision_config": "vision_config" in config,
    }


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"
