#!/usr/bin/env python3
"""Install or upgrade the [GEMMA4-ROPE] fix + guard in a verl checkout (idempotent, upgrade-safe).

Root cause (see LOGPROB_PARITY.md):
  transformers (< 5.16) Gemma4TextRotaryEmbedding.__init__ registers {layer_type}_inv_freq /
  {layer_type}_original_inv_freq while iterating `set(config.layer_types)`, so the buffer registration order
  depends on each process's string-hash randomization (PYTHONHASHSEED). verl wraps the actor with
  FSDP1(sync_module_states=True), which broadcasts rank 0's buffers to the other ranks *by position*. On every rank
  whose order differs from rank 0 (about half of them) the RoPE frequency buffers end up scrambled and all trainer
  log-probs computed on that rank are wrong. verl also casts the module to model_dtype after apply_monkey_patch(),
  which rounds the fp32 RoPE frequencies to bf16. The vLLM rollout (no FSDP) is unaffected.

What this installs (see ROPE_GUARD.md):
  verl/models/transformers/monkey_patch.py      the [GEMMA4-ROPE] block: buffer sort + fp32 rebuild (the fixes),
                                                registration check, init checkpoint, first-forward check, evidence
  verl/workers/engine/fsdp/transformer_impl.py  2 lines right after the FSDP wrap in
                                                FSDPEngine._build_model_optimizer that call the init checkpoint

Upgrade semantics: every earlier [GEMMA4-ROPE] block (any version, including the v1 block delimited by
"[GEMMA4] rank-independent buffer order + fp32 RoPE frequencies") and every earlier hook line are removed first,
then the current version is installed. A pod patched by an older version therefore gets the new checks, and a
file that already holds exactly the current version is left untouched.

Usage:
  python3 patch_verl_gemma4_rope_buffers.py --verl-root /workspace/verl-pin
  python3 patch_verl_gemma4_rope_buffers.py /workspace/verl-pin/verl/models/transformers/monkey_patch.py  # legacy
  python3 patch_verl_gemma4_rope_buffers.py --print-version
Prints one line per file, "GEMMA4-ROPE PATCH: <file> APPLIED | UPGRADED from v<N> | REPAIRED | UP-TO-DATE", then
"GEMMA4-ROPE PATCH RESULT {json}" (version, block sha256, file sha256s). Exits 1 on any failure.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys

PATCH_VERSION = "2"
MONKEY_PATCH_REL = "verl/models/transformers/monkey_patch.py"
TRANSFORMER_IMPL_REL = "verl/workers/engine/fsdp/transformer_impl.py"

BEGIN = f"# ===== [GEMMA4-ROPE] begin (version {PATCH_VERSION}) ====="
END = "# ===== [GEMMA4-ROPE] end ====="
BEGIN_RE = re.compile(r"^# ===== \[GEMMA4-ROPE\] begin \(version (\d+)\) =====$")
V1_BEGIN = "# ===== [GEMMA4] rank-independent buffer order + fp32 RoPE frequencies ====="
V1_END = "# ===== end [GEMMA4] rank-independent buffer order + fp32 RoPE frequencies ====="

BODY = '''# Managed by examples/grpo_trainer/my_scripts_GKE_gemma4/patch_verl_gemma4_rope_buffers.py: edit the patcher,
# not this block. Docs: examples/grpo_trainer/my_scripts_GKE_gemma4/ROPE_GUARD.md and LOGPROB_PARITY.md.
#
# Fixes (Gemma-4, transformers < 5.16, verl FSDP1 engine):
#   - VERL_SORT_MODULE_BUFFERS=1 (default): sort every module's buffer dict by name before the FSDP wrap, so the
#     FSDP1 sync_module_states broadcast (by position) pairs the same buffers on every rank.
#   - VERL_GEMMA4_FP32_ROPE=1 (default): verl casts the module to model_dtype after apply_monkey_patch(), which
#     rounds inv_freq to bf16; the init checkpoint rebuilds the RoPE buffers in fp32.
# Guard:
#   - registration (apply_monkey_patch): a Gemma-4 model must expose >= 1 Gemma4TextRotaryEmbedding with
#     rope_init_fns and both buffers per layer type, otherwise raise (same outcome on every rank).
#   - init checkpoint (FSDPEngine._build_model_optimizer, right after the FSDP wrap, before the optimizer is
#     built): every rank classifies each RoPE buffer against an fp32 reference and the bf16 rounding of that
#     reference (absolute + relative tolerance; zero and non-finite entries handled), rebuilds it in fp32 and
#     re-verifies it. The per-rank results are all-gathered; rank 0 prints and writes one report covering every
#     rank; if any rank failed, ALL ranks raise the same error after the collective.
#   - first forward: every rank re-verifies its buffers against the rebuilt reference, restores them if they
#     changed, and writes a per-rank record (never raises inside the forward).
# Evidence: $VERL_GEMMA4_ROPE_REPORT_DIR (one JSON per rank and phase + one aggregate) and a single
# "GEMMA4-ROPE-GUARD-REPORT {json}" line from rank 0 (Ray de-duplicates per-rank log lines).
# Test-only fault injection: VERL_GEMMA4_ROPE_FAULT_RANK=<global rank> with VERL_GEMMA4_ROPE_FAULT_MODE=
#   reverse_order (default: reverse that rank's RoPE buffer order before the FSDP wrap, i.e. the original bug),
#   scale (x1.05 after the sync) or nan (one NaN after the sync).
import json as _g4_json  # noqa: E402
import os  # noqa: E402
import socket as _g4_socket  # noqa: E402

import torch  # noqa: E402

GEMMA4_ROPE_PATCH_VERSION = "2"
_G4_ROTARY_CLS = "Gemma4TextRotaryEmbedding"
_G4_ATOL = 1e-10  # absolute floor; the only way a zero reference entry matches (proportional RoPE pads zeros)
_G4_RTOL_FP32 = 4e-6  # ~32 fp32 ulp: same formula recomputed on another device / op order
_G4_RTOL_BF16 = 2.0**-7  # <= 1 bf16 ulp from the bf16 rounding of the fp32 reference
_G4_ACCEPT_PRE = ("fp32", "bf16")


def _g4_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _g4_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _g4_fault():
    rank = os.environ.get("VERL_GEMMA4_ROPE_FAULT_RANK", "").strip()
    if not rank:
        return None
    return {"rank": int(rank), "mode": os.environ.get("VERL_GEMMA4_ROPE_FAULT_MODE", "reverse_order").strip()}


def _g4_write(name, obj):
    """Write one evidence JSON into $VERL_GEMMA4_ROPE_REPORT_DIR (never raises)."""
    out_dir = os.environ.get("VERL_GEMMA4_ROPE_REPORT_DIR", "").strip()
    if not out_dir:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            _g4_json.dump(obj, f, indent=1, sort_keys=True, default=str)
        return path
    except Exception as e:  # evidence must never take the job down
        print(f"[GEMMA4-ROPE] WARNING: could not write {name} to {out_dir}: {e!r}", flush=True)
        return None


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


def _g4_rotary_modules(model):
    return [m for m in model.modules() if type(m).__name__ == _G4_ROTARY_CLS and hasattr(m, "rope_init_fns")]


def _g4_is_gemma4(model):
    cfg = getattr(model, "config", None)
    types = [getattr(cfg, "model_type", None), getattr(getattr(cfg, "text_config", None), "model_type", None)]
    if any(str(t).startswith("gemma4") for t in types if t):
        return True
    return any(type(m).__name__.startswith("Gemma4") for m in model.modules())


def _g4_buffer_names(layer_type):
    return (f"{layer_type}_inv_freq", f"{layer_type}_original_inv_freq")


def _g4_reference(module, layer_type, device):
    """fp32 inv_freq exactly as Gemma4TextRotaryEmbedding.__init__ computes it (same init fn, same kwargs)."""
    device = torch.device(device)
    kwargs = {"device": device, "layer_type": layer_type}
    if layer_type == "full_attention" and module.rope_type.get(layer_type) == "proportional":
        kwargs["head_dim_key"] = "global_head_dim"
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        inv_freq, scaling = module.rope_init_fns[layer_type](module.config, **kwargs)
    return inv_freq.to(device=device, dtype=torch.float32), scaling


def _g4_classify(cur, ref):
    """Classify one RoPE buffer against its fp32 reference.

    fp32: |cur - ref| <= ATOL + RTOL_FP32 * |ref| for every entry.
    bf16: cur is exactly bf16-representable and |cur - bf16(ref)| <= ATOL + RTOL_BF16 * |bf16(ref)| for every entry.
    Otherwise mismatch, nonfinite or shape_mismatch. Zero reference entries can only match through ATOL; relative
    errors are reported over the non-zero reference entries only.
    """
    rec = {"dtype": str(cur.dtype).replace("torch.", ""), "device": str(cur.device), "shape": list(cur.shape)}
    if tuple(cur.shape) != tuple(ref.shape):
        rec.update(state="shape_mismatch", ref_shape=list(ref.shape))
        return rec
    c = cur.detach().to(device="cpu", dtype=torch.float64)
    r = ref.detach().to(device="cpu", dtype=torch.float64)
    rb = ref.detach().to(device="cpu").to(torch.bfloat16).to(torch.float64)
    zero_ref = r == 0
    nonfinite = int((~torch.isfinite(c)).sum())
    rec.update(numel=int(c.numel()), n_zero_ref=int(zero_ref.sum()), n_nonfinite=nonfinite)
    if nonfinite:
        rec["state"] = "nonfinite"
        return rec
    nz = ~zero_ref
    d32 = (c - r).abs()
    d16 = (c - rb).abs()
    has_nz = bool(nz.any())
    rec["max_abs_err_fp32"] = float(d32.max()) if c.numel() else 0.0
    rec["max_rel_err_fp32"] = float((d32[nz] / r[nz].abs()).max()) if has_nz else 0.0
    rec["max_abs_err_bf16"] = float(d16.max()) if c.numel() else 0.0
    rec["max_rel_err_bf16"] = float((d16[nz] / rb[nz].abs().clamp_min(1e-300)).max()) if has_nz else 0.0
    rec["n_zero_mismatch"] = int(((c == 0) != zero_ref).sum())
    ok32 = bool((d32 <= _G4_ATOL + _G4_RTOL_FP32 * r.abs()).all())
    cf = cur.detach().to(device="cpu", dtype=torch.float32)
    bf16_exact = bool((cf.to(torch.bfloat16).to(torch.float32) == cf).all())
    ok16 = bf16_exact and bool((d16 <= _G4_ATOL + _G4_RTOL_BF16 * rb.abs()).all())
    rec["bf16_representable"] = bf16_exact
    rec["state"] = "fp32" if ok32 else ("bf16" if ok16 else "mismatch")
    return rec


def _g4_check_module(mod, rebuild):
    """Classify every RoPE buffer (pre), rebuild it in fp32 if requested, re-verify (post). Returns (record, errors)."""
    rec = {"layer_types": sorted(mod.rope_init_fns), "reference": {}, "attention_scaling": {}, "buffers": {}}
    errors, refs = [], {}
    for lt in sorted(mod.rope_init_fns):
        names = _g4_buffer_names(lt)
        missing = [n for n in names if n not in mod._buffers]
        if missing:
            errors.append(f"{lt}: missing buffers {missing}")
            continue
        dev = mod._buffers[names[0]].device
        ref, scaling = _g4_reference(mod, lt, dev)
        ref_cpu, _ = _g4_reference(mod, lt, "cpu")
        diff = (ref.detach().cpu().double() - ref_cpu.double()).abs()
        oracle_ok = bool((diff <= _G4_ATOL + _G4_RTOL_FP32 * ref_cpu.double().abs()).all())
        rec["reference"][lt] = {"device": str(dev), "numel": int(ref.numel()), "n_zero": int((ref == 0).sum()),
                                "cpu_oracle_max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
                                "cpu_oracle_ok": oracle_ok}
        if not oracle_ok:
            errors.append(f"{lt}: fp32 reference on {dev} disagrees with the CPU fp32 reference")
        have = getattr(mod, f"{lt}_attention_scaling", None)
        rec["attention_scaling"][lt] = {"module": have, "reference": scaling}
        if have != scaling:
            errors.append(f"{lt}: attention_scaling {have!r} != reference {scaling!r}")
        for n in names:
            pre = _g4_classify(mod._buffers[n], ref)
            entry = {"pre": pre}
            if pre["state"] not in _G4_ACCEPT_PRE:
                errors.append(f"{n}: {pre['state']} (max_rel_err vs fp32 {pre.get('max_rel_err_fp32')}, "
                              f"vs bf16 {pre.get('max_rel_err_bf16')})")
            if rebuild:
                setattr(mod, n, ref.clone())
                post_t = mod._buffers.get(n)
                post = _g4_classify(post_t, ref)
                exact = bool(post_t.dtype == torch.float32 and torch.equal(post_t, ref))
                entry["post"] = {"state": post["state"], "dtype": post["dtype"], "bit_exact": exact}
                if not (exact and post["state"] == "fp32"):
                    errors.append(f"{n}: post-rebuild check failed (state={post['state']}, dtype={post['dtype']}, "
                                  f"bit_exact={exact})")
            rec["buffers"][n] = entry
        refs[lt] = ref.detach().clone()
    mod._verl_g4_refs = refs
    return rec, errors


def _g4_first_forward_pre_hook(module, args):
    """Per-rank use-time check on the first forward. Never raises: restores the reference and records."""
    if getattr(module, "_verl_g4_first_forward_done", False):
        return
    module._verl_g4_first_forward_done = True
    rank = _g4_rank()
    rebuild = os.environ.get("VERL_GEMMA4_FP32_ROPE", "1") == "1"
    rec = {"rank": rank, "host": _g4_socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "pid": os.getpid(), "phase": "first_forward", "version": GEMMA4_ROPE_PATCH_VERSION,
           "role": getattr(module, "_verl_g4_role", "unknown"), "fp32_rope": rebuild, "buffers": {}, "restored": [],
           "errors": []}
    try:
        refs = getattr(module, "_verl_g4_refs", None)
        if refs is None:  # engine path without the init checkpoint: check + rebuild here, flag it
            rec["errors"].append("init checkpoint did not run for this module; checked at first forward instead")
            fallback, errs = _g4_check_module(module, rebuild)
            rec["fallback"] = fallback
            rec["errors"].extend(errs)
        else:
            for lt, ref in refs.items():
                for n in _g4_buffer_names(lt):
                    b = module._buffers.get(n)
                    want = ref.to(b.device)
                    if rebuild:
                        ok = bool(b.dtype == torch.float32 and torch.equal(b, want))
                        state = "fp32_exact" if ok else _g4_classify(b, want)["state"]
                    else:
                        state = _g4_classify(b, want)["state"]
                        ok = state in _G4_ACCEPT_PRE
                    rec["buffers"][n] = {"state": state, "dtype": str(b.dtype).replace("torch.", ""), "ok": ok}
                    if not ok:
                        rec["errors"].append(f"{n}: changed after the init checkpoint ({state})")
                        if rebuild:
                            module._buffers[n] = want.clone()
                            rec["restored"].append(n)
    except Exception as e:
        rec["errors"].append(f"exception: {type(e).__name__}: {e}")
    rec["status"] = "FAIL" if rec["errors"] else "PASS"
    _g4_write(f"first_forward_{rec['role']}_rank{rank:04d}.json", rec)
    print(f"[GEMMA4-ROPE] first-forward check rank={rank} role={rec['role']} status={rec['status']} "
          f"restored={rec['restored']}", flush=True)


def _g4_register_guard(model):
    """Registration-time checks (identical on every rank). Returns the number of guarded rotary modules."""
    if not _g4_is_gemma4(model):
        return 0
    mods = _g4_rotary_modules(model)
    if not mods:
        raise RuntimeError(f"[GEMMA4-ROPE] Gemma-4 model without a {_G4_ROTARY_CLS} exposing rope_init_fns: the RoPE "
                           "fix and guard cannot attach (did transformers change?). Refusing to train.")
    for m in mods:
        for lt in m.rope_init_fns:
            missing = [n for n in _g4_buffer_names(lt) if n not in m._buffers]
            if missing:
                raise RuntimeError(f"[GEMMA4-ROPE] {_G4_ROTARY_CLS} lacks buffers {missing}. Refusing to train.")
        if not getattr(m, "_verl_g4_hooked", False):
            m.register_forward_pre_hook(_g4_first_forward_pre_hook)
            m._verl_g4_hooked = True
    hooked = sum(1 for m in mods if getattr(m, "_verl_g4_hooked", False))
    if hooked != len(mods):
        raise RuntimeError(f"[GEMMA4-ROPE] registered {hooked} first-forward hook(s) for {len(mods)} rotary module(s)")
    model._verl_g4_expected_rotary = len(mods)
    fault = _g4_fault()
    if fault and fault["mode"] == "reverse_order" and fault["rank"] == _g4_rank():
        for m in mods:
            items = list(m._buffers.items())[::-1]
            m._buffers.clear()
            m._buffers.update(items)
        model._verl_g4_fault_applied = "reverse_order(before FSDP sync)"
        print(f"[GEMMA4-ROPE] TEST FAULT INJECTED on rank {_g4_rank()}: reversed the RoPE buffer order before the "
              "FSDP sync", flush=True)
    return len(mods)


def _g4_apply_fault_post_sync(mods, mode):
    m = mods[0]
    lt = "full_attention" if "full_attention" in m.rope_init_fns else sorted(m.rope_init_fns)[0]
    n = f"{lt}_inv_freq"
    with torch.no_grad():
        b = m._buffers[n].clone()
        if mode == "scale":
            b = b * 1.05
        elif mode == "nan":
            b.view(-1)[0] = float("nan")
        else:
            return None
        m._buffers[n] = b
    return f"{mode}({n}, after FSDP sync)"


def _g4_compact(local):
    bufs = {}
    for i, m in enumerate(local.get("modules", [])):
        for n, e in m["buffers"].items():
            pre, post = e["pre"], e.get("post", {})
            bufs[f"{i}:{n}"] = [pre["state"], pre["dtype"], pre.get("max_rel_err_fp32"), post.get("state"),
                                post.get("bit_exact")]
    keep = ("rank", "local_rank", "host", "cuda_visible_devices", "pid", "role", "status", "n_rotary", "fault_applied")
    out = {k: local.get(k) for k in keep}
    out["errors"] = local["errors"][:6]
    out["buffers"] = bufs
    return out


def _verl_gemma4_rope_init_checkpoint(module, forward_only=False):
    """All-rank RoPE checkpoint right after the FSDP wrap (called from FSDPEngine._build_model_optimizer)."""
    hf = next((m for m in module.modules() if hasattr(m, "_verl_g4_expected_rotary")), None)
    if hf is None:
        return None  # not a guarded (Gemma-4) model: identical decision on every rank, no collective
    role = "ref" if forward_only else "actor"
    rank, world = _g4_rank(), _g4_world()
    rebuild = os.environ.get("VERL_GEMMA4_FP32_ROPE", "1") == "1"
    fault = _g4_fault()
    local = {"rank": rank, "local_rank": int(os.environ.get("LOCAL_RANK", "-1")), "host": _g4_socket.gethostname(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), "role": role,
             "phase": "init_checkpoint", "version": GEMMA4_ROPE_PATCH_VERSION, "fp32_rope": rebuild, "fault": fault,
             "fault_applied": getattr(hf, "_verl_g4_fault_applied", None), "errors": [], "modules": []}
    try:
        mods = _g4_rotary_modules(module)
        local["n_rotary"] = len(mods)
        if len(mods) != hf._verl_g4_expected_rotary:
            local["errors"].append(f"found {len(mods)} rotary module(s) after the FSDP wrap, "
                                   f"registered {hf._verl_g4_expected_rotary}")
        if fault and fault["rank"] == rank and fault["mode"] in ("scale", "nan") and mods:
            local["fault_applied"] = _g4_apply_fault_post_sync(mods, fault["mode"])
            print(f"[GEMMA4-ROPE] TEST FAULT INJECTED on rank {rank}: {local['fault_applied']}", flush=True)
        for i, m in enumerate(mods):
            m._verl_g4_role = role
            rec, errs = _g4_check_module(m, rebuild)
            local["modules"].append(rec)
            local["errors"].extend(f"rotary[{i}] {e}" for e in errs)
    except Exception as e:  # a local exception becomes a FAIL status so every rank still reaches the collective
        local["errors"].append(f"exception: {type(e).__name__}: {e}")
    local["status"] = "FAIL" if local["errors"] else "PASS"
    _g4_write(f"init_{role}_rank{rank:04d}.json", local)
    compact = _g4_compact(local)
    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, compact)
    else:
        gathered = [compact]
    gathered = sorted(gathered, key=lambda g: g["rank"])
    failed = [g for g in gathered if g["status"] != "PASS"]
    states, max_rel = {}, 0.0
    for g in gathered:
        for v in g["buffers"].values():
            states[v[0]] = states.get(v[0], 0) + 1
            if v[2] is not None and v[0] in _G4_ACCEPT_PRE:
                max_rel = max(max_rel, v[2])
    summary = {"kind": "gemma4_rope_guard", "version": GEMMA4_ROPE_PATCH_VERSION, "phase": "init_checkpoint",
               "role": role, "world_size": world, "ranks": [g["rank"] for g in gathered],
               "n_pass": len(gathered) - len(failed), "n_fail": len(failed),
               "failed_ranks": [g["rank"] for g in failed], "pre_states": states,
               "max_rel_err_vs_fp32_accepted_pre": max_rel, "fp32_rope": rebuild, "fault": fault,
               "tolerances": {"atol": _G4_ATOL, "rtol_fp32": _G4_RTOL_FP32, "rtol_bf16": _G4_RTOL_BF16}}
    if rank == 0:
        line = dict(summary)
        line["per_rank"] = [[g["rank"], g["status"], g["host"], g["cuda_visible_devices"],
                             sorted({v[0] for v in g["buffers"].values()}),
                             all(v[4] for v in g["buffers"].values()) if rebuild else None] for g in gathered]
        line["per_rank_fields"] = ["rank", "status", "host", "cuda_visible_devices", "pre_states", "post_bit_exact"]
        # leading newline: NCCL's C-level stdout buffering can leave a partial line in the worker log
        print("\\nGEMMA4-ROPE-GUARD-REPORT " + _g4_json.dumps(line, sort_keys=True, separators=(",", ":")), flush=True)
        _g4_write(f"init_{role}_aggregate.json", dict(summary, per_rank=gathered))
    if failed:
        detail = "; ".join(f"rank {g['rank']} ({g['host']}): {' | '.join(g['errors'][:2])}" for g in failed)
        raise RuntimeError(f"[GEMMA4-ROPE] guard FAILED on {len(failed)}/{world} rank(s) (role={role}, "
                           f"phase=init_checkpoint, before the optimizer is built): {detail[:3000]}")
    return summary


_verl_orig_apply_monkey_patch = apply_monkey_patch


def apply_monkey_patch(model, *args, **kwargs):  # noqa: F811 -- wraps the original
    if os.environ.get("VERL_SORT_MODULE_BUFFERS", "1") == "1":
        n = _verl_sort_module_buffers(model)
        print(f"Monkey patch: sorted buffers of {n} module(s) (rank-independent FSDP buffer sync order)")
    n = _g4_register_guard(model)
    if n:
        print(f"Monkey patch: [GEMMA4-ROPE v{GEMMA4_ROPE_PATCH_VERSION}] guard registered on {n} {_G4_ROTARY_CLS} "
              "module(s)")
    return _verl_orig_apply_monkey_patch(model, *args, **kwargs)
'''

BLOCK = BEGIN + "\n" + BODY + END + "\n"

HOOK_MARK = "# [GEMMA4-ROPE]"
TI_ANCHOR = "        module = self._build_fsdp_module(module)\n"
TI_HOOK = (
    "        from verl.models.transformers.monkey_patch import _verl_gemma4_rope_init_checkpoint as _g4_ckpt"
    "  # [GEMMA4-ROPE]\n"
    "        _g4_ckpt(module, forward_only=self.engine_config.forward_only)"
    f"  # [GEMMA4-ROPE] v{PATCH_VERSION} init checkpoint, see ROPE_GUARD.md\n"
)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def block_sha256():
    return sha256(BLOCK)


def strip_blocks(src):
    """Remove every [GEMMA4-ROPE] block (any version) and the v1 block. Returns (text, removed versions)."""
    lines = src.split("\n")
    out, removed, i = [], [], 0
    while i < len(lines):
        m = BEGIN_RE.match(lines[i])
        if m or lines[i] == V1_BEGIN:
            end = END if m else V1_END
            j = i + 1
            while j < len(lines) and lines[j] != end:
                j += 1
            if j == len(lines):
                raise ValueError(f"unterminated block starting at line {i + 1}")
            removed.append(m.group(1) if m else "1")
            while out and out[-1] == "":
                out.pop()
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).rstrip("\n") + "\n", removed


def patch_monkey_patch_text(src):
    if "\ndef apply_monkey_patch(" not in src:
        raise ValueError("apply_monkey_patch() not found")
    base, removed = strip_blocks(src)
    new = base.rstrip("\n") + "\n" + "\n\n" + BLOCK
    if new == src:
        action = "UP-TO-DATE"
    elif not removed:
        action = "APPLIED"
    elif all(v == PATCH_VERSION for v in removed):
        action = "REPAIRED"
    else:
        action = "UPGRADED from v" + ",v".join(removed)
    return new, action


def patch_transformer_impl_text(src):
    stripped = "".join(line for line in src.splitlines(keepends=True) if HOOK_MARK not in line)
    count = stripped.count(TI_ANCHOR)
    if count != 1:
        raise ValueError(f"FSDP wrap anchor found {count} times (expected 1)")
    new = stripped.replace(TI_ANCHOR, TI_ANCHOR + TI_HOOK, 1)
    if new == src:
        action = "UP-TO-DATE"
    elif stripped == src:
        action = "APPLIED"
    else:
        action = "UPGRADED"
    return new, action


def write_atomic(path, text):
    tmp = path + ".g4rope.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("monkey_patch_path", nargs="?", help="legacy form: path to verl/models/transformers/monkey_patch.py")
    ap.add_argument("--verl-root", help="verl checkout root (contains verl/)")
    ap.add_argument("--print-version", action="store_true")
    args = ap.parse_args()
    if args.print_version:
        print(json.dumps({"version": PATCH_VERSION, "block_sha256": block_sha256()}))
        return 0
    if args.verl_root:
        root = args.verl_root
    elif args.monkey_patch_path:
        root = os.path.abspath(os.path.join(os.path.dirname(args.monkey_patch_path), "..", "..", ".."))
    else:
        root = "/workspace/verl-pin"
    result = {"version": PATCH_VERSION, "block_sha256": block_sha256(), "verl_root": root, "files": {}}
    try:
        for rel, fn in ((MONKEY_PATCH_REL, patch_monkey_patch_text), (TRANSFORMER_IMPL_REL, patch_transformer_impl_text)):
            path = os.path.join(root, rel)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            new, action = fn(src)
            ast.parse(new)
            if new != src:
                write_atomic(path, new)
            result["files"][rel] = {"action": action, "sha256": sha256(new)}
            print(f"GEMMA4-ROPE PATCH: {rel} {action} (version {PATCH_VERSION}, sha256 {sha256(new)[:16]})")
    except Exception as e:
        print(f"GEMMA4-ROPE PATCH: FAILED {type(e).__name__}: {e}")
        result["error"] = f"{type(e).__name__}: {e}"
        print("GEMMA4-ROPE PATCH RESULT " + json.dumps(result, sort_keys=True))
        return 1
    print("GEMMA4-ROPE PATCH RESULT " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
