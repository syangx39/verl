#!/usr/bin/env python3
"""Append the [GEMMA4] RoPE-buffer fixes to verl's monkey_patch.py (idempotent).

Root cause (see LOGPROB_PARITY.md):
  transformers' Gemma4TextRotaryEmbedding.__init__ registers {layer_type}_inv_freq /
  {layer_type}_original_inv_freq while iterating `set(config.layer_types)`, so the buffer
  registration order depends on each process's string-hash randomization (PYTHONHASHSEED).
  verl wraps the actor with FSDP1(sync_module_states=True), which broadcasts rank 0's buffers to
  the other ranks *by position*. On every rank whose order differs from rank 0 (about half of
  them), the RoPE frequency buffers end up scrambled (sliding <- full values, zeros, ...), and all
  trainer log-probs computed on that rank are wrong. The vLLM rollout (no FSDP) is unaffected.

Fixes appended to verl/models/transformers/monkey_patch.py, applied through apply_monkey_patch(),
which verl's FSDP engine calls after from_pretrained() and before the FSDP wrap:
  1. VERL_SORT_MODULE_BUFFERS (default 1): sort every module's buffer dict by name, so the buffer
     order (and therefore the FSDP broadcast pairing) is identical on every rank. Values unchanged.
  2. VERL_GEMMA4_FP32_ROPE (default 1): on the first forward of each Gemma4TextRotaryEmbedding,
     recompute its inv_freq buffers in fp32 on device from the config. verl's _build_module calls
     module.to(model_dtype) after apply_monkey_patch, which rounds the RoPE frequencies to bf16;
     FSDP's fp32 buffer_dtype cast cannot restore the lost precision. vLLM keeps them in fp32.

Prints: GEMMA4-ROPE PATCH: APPLIED | already patched | FAILED <reason>
Usage: python3 patch_verl_gemma4_rope_buffers.py /workspace/verl-pin/verl/models/transformers/monkey_patch.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl-pin/verl/models/transformers/monkey_patch.py"
src = open(path, encoding="utf-8").read()
MARK = "_verl_sort_module_buffers"
if MARK in src:
    print("GEMMA4-ROPE PATCH: already patched")
    sys.exit(0)
if "\ndef apply_monkey_patch(" not in src:
    print("GEMMA4-ROPE PATCH: FAILED apply_monkey_patch() not found")
    sys.exit(1)

BLOCK = '''

# ===== [GEMMA4] rank-independent buffer order + fp32 RoPE frequencies =====
# Gemma4TextRotaryEmbedding registers its RoPE buffers in set(config.layer_types) order, which
# depends on per-process hash randomization; FSDP1 sync_module_states broadcasts buffers from
# rank 0 by position, so on ~half of the ranks the RoPE buffers were scrambled. verl also casts
# the module to model_dtype after apply_monkey_patch, rounding inv_freq to bf16. See
# examples/grpo_trainer/my_scripts_GKE_gemma4/LOGPROB_PARITY.md.
import os  # noqa: E402

import torch  # noqa: E402


def _verl_sort_module_buffers(model):
    """Sort each module's buffers by name (values untouched) so every rank has the same order."""
    changed = 0
    for mod in model.modules():
        bufs = mod._buffers
        if len(bufs) > 1 and list(bufs) != sorted(bufs):
            items = sorted(bufs.items())
            bufs.clear()
            bufs.update(items)
            changed += 1
    return changed


def _verl_gemma4_fp32_rope_pre_hook(module, args):
    """Recompute Gemma4TextRotaryEmbedding inv_freq buffers in fp32 on device (once per process)."""
    if getattr(module, "_verl_fp32_rope_done", False):
        return
    module._verl_fp32_rope_done = True
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for layer_type, init_fn in module.rope_init_fns.items():
            ref = getattr(module, f"{layer_type}_inv_freq")
            kwargs = {"device": ref.device, "layer_type": layer_type}
            if layer_type == "full_attention" and module.rope_type.get(layer_type) == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, _ = init_fn(module.config, **kwargs)
            for name in (f"{layer_type}_inv_freq", f"{layer_type}_original_inv_freq"):
                setattr(module, name, inv_freq.to(device=ref.device, dtype=torch.float32).clone())


_verl_orig_apply_monkey_patch = apply_monkey_patch


def apply_monkey_patch(model, *args, **kwargs):  # noqa: F811 -- wraps the original
    if os.environ.get("VERL_SORT_MODULE_BUFFERS", "1") == "1":
        n = _verl_sort_module_buffers(model)
        print(f"Monkey patch: sorted buffers of {n} module(s) (rank-independent FSDP buffer sync order)")
    if os.environ.get("VERL_GEMMA4_FP32_ROPE", "1") == "1":
        n = 0
        for mod in model.modules():
            if type(mod).__name__ == "Gemma4TextRotaryEmbedding" and hasattr(mod, "rope_init_fns"):
                mod.register_forward_pre_hook(_verl_gemma4_fp32_rope_pre_hook)
                n += 1
        if n:
            print(f"Monkey patch: fp32 RoPE recompute hook on {n} Gemma4TextRotaryEmbedding module(s)")
    return _verl_orig_apply_monkey_patch(model, *args, **kwargs)
# ===== end [GEMMA4] rank-independent buffer order + fp32 RoPE frequencies =====
'''
new = src.rstrip("\n") + "\n" + BLOCK
try:
    ast.parse(new)
except SyntaxError as e:
    print(f"GEMMA4-ROPE PATCH: FAILED patched file does not parse: {e}")
    sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("GEMMA4-ROPE PATCH: APPLIED")
