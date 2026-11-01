# awq-checkpoint-surgery

Tools for understanding and shrinking a 4-bit AWQ checkpoint, with the one thing
that matters when you rewrite quantised weights: **byte-exact verification of the
packing order**.

AWQ checkpoints published with compressed-tensors keep the vocabulary in BF16.
On a small GPU that is the difference between fitting and not fitting — which is
why the surgery below exists.

## Why the packing order has to be verified

A 4-bit pack-quantized tensor is not self-describing. The format fixes:

- how many values share a scale (group size 32),
- whether the scale is symmetric (here: yes, `-8..7`),
- **which nibble of the `int32` holds which value**.

Get the last one wrong and nothing fails loudly. The model loads, the shapes
match, the logits come out garbage. So the library does not *assume* the layout —
it reads a tensor the AWQ tooling itself wrote, unpacks it, re-packs it, and
demands identical bytes:

```bash
AWQ_CHECKPOINT=~/models/qwen3.5-9b-awq python -m pytest tests/ -v
```

```
tests/test_packing.py::test_repack_matches_real_checkpoint_bytes PASSED
```

That is the difference between "I implemented a quantisation scheme" and "I
implemented *this* quantisation scheme".

## The layout, as verified

```
weight [out, in]                    float
  -> groups of 32 along `in`, one bf16 scale per group (symmetric, /8)
  -> int4 values in -8..7
  -> 8 values into one int32, LOW nibble first:
        word = v0 | v1<<4 | v2<<8 | v3<<12 | ... | v7<<28
  -> packed [out, in//8] int32  +  scales [out, in//32] bf16
```

Negative values are two's-complement nibbles: `-1` is `0xF`, `-8` is `0x8`.

## What the surgery does

```bash
python -m awq_surgery.surgery inspect ~/models/qwen3.5-9b-awq
```

```
tensors 1189   real bytes 8.45 GiB

group                 tensors        size   share
body (quantised)          854     3.81 GiB   45.1%
lm_head                     1     1.90 GiB   22.4%
embedding                   1     1.90 GiB   22.4%
vision                    333     0.85 GiB   10.1%

quant format : compressed-tensors / pack-quantized
group_0: 4-bit group 32 symmetric=True targets=['Linear']
ignore list  : 160 entries, vocabulary ignored: True
tied embeds  : False
```

`targets: ["Linear"]` plus `lm_head` in the ignore list is the whole reason a
4-bit 9B weighs 8.45 GiB: two vocabulary matrices, 1.90 GiB each, never touched.

```bash
python -m awq_surgery.surgery strip \
    --src  ~/models/qwen3.5-9b-awq \
    --dst  ~/models/qwen3.5-9b-text \
    --drop-mtp --quantize-lm-head --causal
```

| Change | Saves | Result |
|---|---:|---:|
| vision tower (333 tensors) | 0.85 GiB | 7.60 GiB |
| MTP head (29 tensors) | 0.30 GiB | 7.30 GiB |
| `lm_head` → int4 | 1.34 GiB | **6.06 GiB** |
| `architectures` → text-only class | — | — |

The output is a normal compressed-tensors checkpoint: same config schema, same
shard layout, `lm_head` removed from the ignore list so the 4-bit group applies
to it.

## Caveat worth stating plainly

A smaller checkpoint is not the same as a checkpoint that loads. On an 8 GB card
the 6.06 GiB result above still failed — vLLM 0.31 creates weights on real GPU
memory and then repacks them into the Marlin layout, needing roughly twice the
weight footprint in transient VRAM. The surgery was correct; the loader was the
constraint. Details in
[small-gpu-llm-notes/docs/03-vllm-limits.md](https://github.com/Op-Singularity/small-gpu-llm-notes/blob/main/docs/03-vllm-limits.md).

## Install

```bash
pip install torch safetensors pytest
python -m pytest tests/ -v
```

## API

```python
from awq_surgery.packing import unpack_int4, pack_int4, dequantize, quantize_symmetric

values  = unpack_int4(packed)                 # [out, in//8] int32 -> [out, in] int8
packed  = pack_int4(values)                   # back, byte-identical
weight  = dequantize(packed, scales)          # float reconstruction
packed, scales, shape = quantize_symmetric(w) # quantise anything
```

## License

MIT.
