"""Checkpoint surgery: remove what you do not need, quantise what is worth it.

Two operations, both aimed at the same problem: an AWQ checkpoint leaves the
vocabulary and the vision tower in BF16, and on a small card those are the parts
that decide whether the model fits.

    # what is in here, and where the bytes went
    python -m awq_surgery.surgery inspect ~/models/qwen3.5-9b-awq

    # vision tower out, lm_head to int4, config retargeted to the text-only class
    python -m awq_surgery.surgery strip --src ~/models/qwen3.5-9b-awq \
        --dst ~/models/qwen3.5-9b-text --drop-mtp --quantize-lm-head --causal

The output is a normal compressed-tensors checkpoint: same config layout, same
shard format, lm_head moved out of the ignore list so the 4-bit scheme applies.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .inspect import human, inventory, quant_scheme, shard_files
from .packing import quantize_symmetric

SHARD_BYTES = 4 * 1024**3
COPY_SUFFIXES = (".json", ".txt", ".model", ".py")
SKIP_COPY = {"config.json", "model.safetensors.index.json"}


def cmd_inspect(args) -> int:
    groups, total = inventory(args.src)
    scheme = quant_scheme(args.src)
    print(f"{args.src}\n  {sum(len(v) for v in groups.values())} tensors, {human(total)}\n")
    print(f"  {'group':<20}{'tensors':>8}{'size':>12}{'share':>8}")
    for name, infos in sorted(groups.items(), key=lambda kv: -sum(i.nbytes for i in kv[1])):
        size = sum(i.nbytes for i in infos)
        print(f"  {name:<20}{len(infos):>8}{human(size):>12}{size / total * 100:>7.1f}%")

    print(f"\n  quant format : {scheme['quant_method']} / {scheme['format']}")
    for key, value in scheme["groups"].items():
        print(f"  {key}: {value['num_bits']}-bit group {value['group_size']} "
              f"symmetric={value['symmetric']} targets={value['targets']}")
    ignored = scheme["ignore"]
    vocab_ignored = [x for x in ignored if "lm_head" in x]
    print(f"  ignore list  : {len(ignored)} entries, vocabulary ignored: {bool(vocab_ignored)}")
    print(f"  tied embeds  : {scheme['tied']}")
    print(f"  architectures: {scheme['architectures']}")
    print(f"  vision_config: {scheme['has_vision_config']}")
    return 0


def cmd_strip(args) -> int:
    src, dst = args.src, args.dst
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("--dst must differ from --src")

    keep: dict[str, torch.Tensor] = {}
    dropped = {"vision": 0, "mtp": 0}
    for shard in shard_files(src):
        with safe_open(os.path.join(src, shard), framework="pt") as handle:
            for name in handle.keys():
                if args.drop_vision and (".visual." in name or name.startswith("visual.")):
                    dropped["vision"] += 1
                    continue
                if args.drop_mtp and name.startswith("mtp."):
                    dropped["mtp"] += 1
                    continue
                keep[name] = handle.get_tensor(name)

    print(f"kept {len(keep)} tensors, dropped vision={dropped['vision']} mtp={dropped['mtp']}")

    if args.quantize_lm_head and "lm_head.weight" in keep:
        weight = keep.pop("lm_head.weight")
        packed, scales, shape = quantize_symmetric(weight)
        keep["lm_head.weight_packed"] = packed
        keep["lm_head.weight_scale"] = scales
        keep["lm_head.weight_shape"] = shape
        saved = weight.numel() * weight.element_size() - (packed.numel() * 4 + scales.numel() * 2)
        print(f"lm_head {tuple(weight.shape)} -> int4, {human(saved)} saved")
        del weight

    total = sum(t.numel() * t.element_size() for t in keep.values())
    print(f"resulting checkpoint: {human(total)}")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    shards: list[tuple[dict[str, torch.Tensor], int]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    index = 1
    for name, tensor in sorted(keep.items()):
        nbytes = tensor.numel() * tensor.element_size()
        if current_bytes + nbytes > SHARD_BYTES and current:
            shards.append((current, index))
            current, current_bytes, index = {}, 0, index + 1
        current[name] = tensor.contiguous()
        current_bytes += nbytes
    if current:
        shards.append((current, index))

    weight_map = {}
    for tensors, number in shards:
        filename = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        save_file(tensors, os.path.join(dst, filename), metadata={"format": "pt"})
        for key in tensors:
            weight_map[key] = filename
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"  {filename}: {len(tensors)} tensors, {human(size)}")

    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    config = json.load(open(os.path.join(src, "config.json")))
    if args.causal:
        config["architectures"] = [args.arch]
        config.pop("vision_config", None)
    if args.quantize_lm_head:
        quant = config.get("quantization_config") or {}
        if quant.get("ignore"):
            quant["ignore"] = [x for x in quant["ignore"] if "lm_head" not in x]
        config["quantization_config"] = quant
    json.dump(config, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for name in os.listdir(src):
        if name in SKIP_COPY or name.startswith("model-") or not name.endswith(COPY_SUFFIXES):
            continue
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))

    print(f"\nwrote {dst}")
    print(f"  architectures: {config.get('architectures')}, "
          f"vision_config present: {'vision_config' in config}, "
          f"tied: {config.get('tie_word_embeddings')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="awq_surgery", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="where the bytes went, and what the quant config says")
    p_inspect.add_argument("src")
    p_inspect.set_defaults(func=cmd_inspect)

    p_strip = sub.add_parser("strip", help="produce a smaller checkpoint")
    p_strip.add_argument("--src", required=True)
    p_strip.add_argument("--dst", required=True)
    p_strip.add_argument("--drop-vision", action="store_true", default=True)
    p_strip.add_argument("--keep-vision", dest="drop_vision", action="store_false")
    p_strip.add_argument("--drop-mtp", action="store_true", help="multi-token-prediction head")
    p_strip.add_argument("--quantize-lm-head", action="store_true", help="int4, group 32, symmetric")
    p_strip.add_argument("--causal", action="store_true", help="retarget config to the text-only class")
    p_strip.add_argument("--arch", default="Qwen3_5ForCausalLM")
    p_strip.set_defaults(func=cmd_strip)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
