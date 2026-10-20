#!/usr/bin/env python3
"""Idempotently ensure verl's monkey_patch.py contains the [GEMMA4] flex_attention
Triton tile-pinning patch for head_dim > 256 (needed when running against an
upstream verl checkout such as /workspace/verl-pin).
"""
import os
import sys

PATCH_MARKER = "_verl_patch_flex_kernel_options"

PATCH_BLOCK = """

# ===== [GEMMA4] pin flex-attention Triton tiles for head_dim>256 =====
# Default inductor tiles overflow the 227KB SMEM/CTA limit at head_dim=512;
# EXHAUSTIVE autotune (GB200, L~15k) found the winners but costs ~3min/shape
# and can poison the CUDA context via a buggy candidate (misaligned address).
# So: inject the measured winner directly, no autotune env needed.
#   fwd winner: BLOCK_M=32, BLOCK_N=128, num_stages=1, num_warps=8 (41.4ms)
#   bwd: conservative 32x32 tiles (not benchmarked; must also fit SMEM)
def _verl_gemma4_flex_kernel_options(head_dim):
    if head_dim <= 256:
        return None  # default config compiles fine for 256
    return {
        "BLOCK_M": 32, "BLOCK_N": 128,
        "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32,
        "num_warps": 8, "num_stages": 1,
    }


def _verl_patch_flex_kernel_options():
    from transformers.integrations import flex_attention as _fa
    if getattr(_fa, "_verl_kernel_options_patched", False):
        return
    _orig = _fa.compile_friendly_flex_attention

    def _wrapped(query, key, value, training=False, **kwargs):
        opts = _verl_gemma4_flex_kernel_options(query.shape[-1])
        if opts is not None:
            merged = dict(kwargs.get("kernel_options") or {})
            merged.update(opts)
            kwargs["kernel_options"] = merged
        return _orig(query, key, value, training=training, **kwargs)

    _fa.compile_friendly_flex_attention = _wrapped
    _fa._verl_kernel_options_patched = True
    print("Monkey patch: flex_attention kernel_options pinned for head_dim>256")


_verl_patch_flex_kernel_options()
# ===== end [GEMMA4] flex kernel_options patch =====
"""


def main():
    targets = []
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        import verl.models.transformers.monkey_patch as mp
        targets = [os.path.abspath(mp.__file__)]

    for path in targets:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        if PATCH_MARKER in src:
            print(f"[patch_verl_gemma4_flex_attn] already patched: {path}")
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(PATCH_BLOCK)
        print(f"[patch_verl_gemma4_flex_attn] patched: {path}")


if __name__ == "__main__":
    main()
