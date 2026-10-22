#!/usr/bin/env python3
"""Deterministic single-process tests of the [GEMMA4-ROPE] guard as installed in the local verl checkout.

Run on a pod after gemma4_node_prep.py (imports verl.models.transformers.monkey_patch from the image's verl checkout):
  python3 test_gemma4_rope_guard.py --model /workspace/meta-RL/models/gemma-4-E2B-it [--out result.json]
Builds real Gemma4TextRotaryEmbedding modules from the model config (CPU) and checks the classifier states, the
tolerance design (not a single relative threshold), zero / non-finite handling, the rebuild, the registration check,
the init checkpoint (world size 1) with and without fault injection, the first-forward check, and a simulated FSDP1
coalesced broadcast between two different buffer orders (the original bug).
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig
from transformers.models.gemma4 import modeling_gemma4

import verl.models.transformers.monkey_patch as mp

RESULTS = []


def case(name):
    def deco(fn):
        def run(*a):
            try:
                detail = fn(*a)
                RESULTS.append({"case": name, "ok": True, "detail": detail})
            except AssertionError as e:
                RESULTS.append({"case": name, "ok": False, "detail": f"assertion: {e}"})
            except Exception as e:
                RESULTS.append({"case": name, "ok": False, "detail": f"{type(e).__name__}: {e}"})
        return run
    return deco


def fresh(cfg, order=None):
    m = modeling_gemma4.Gemma4TextRotaryEmbedding(cfg)
    if order is not None:
        items = sorted(m._buffers.items())
        if order == "reversed":
            items = items[::-1]
        m._buffers.clear()
        m._buffers.update(items)
    return m


def states(m):
    out = {}
    for lt in sorted(m.rope_init_fns):
        ref, _ = mp._g4_reference(m, lt, "cpu")
        for n in mp._g4_buffer_names(lt):
            out[n] = mp._g4_classify(m._buffers[n], ref)
    return out


def broadcast_coalesced(src, dst):
    """What FSDP1 sync_module_states does to buffers: flatten rank 0's buffers in ITS order, unflatten by position."""
    flat = torch.cat([b.detach().reshape(-1).float() for b in src._buffers.values()])
    off = 0
    for n, b in list(dst._buffers.items()):
        k = b.numel()
        dst._buffers[n] = flat[off:off + k].reshape(b.shape).to(b.dtype).clone()
        off += k
    assert off == flat.numel()


class Holder(torch.nn.Module):
    """Stands in for the HF model: a container whose class name marks it as Gemma-4."""

    def __init__(self, rot):
        super().__init__()
        self.rotary_emb = rot


Holder.__name__ = "Gemma4Holder"


@case("fresh fp32 buffers -> fp32")
def t_fresh(cfg):
    s = states(fresh(cfg))
    assert all(v["state"] == "fp32" for v in s.values()), {k: v["state"] for k, v in s.items()}
    return {k: [v["state"], v["n_zero_ref"]] for k, v in s.items()}


@case("module.to(bf16) -> bf16")
def t_bf16(cfg):
    s = states(fresh(cfg).to(torch.bfloat16))
    assert all(v["state"] == "bf16" for v in s.values()), {k: v["state"] for k, v in s.items()}
    return {k: round(v["max_rel_err_fp32"], 6) for k, v in s.items()}


@case("bf16 then fp32 (FSDP buffer_dtype cast) -> bf16")
def t_bf16_fp32(cfg):
    s = states(fresh(cfg).to(torch.bfloat16).to(torch.float32))
    assert all(v["state"] == "bf16" for v in s.values()), {k: v["state"] for k, v in s.items()}
    return {k: v["dtype"] for k, v in s.items()}


@case("0.2% error on fp32 values -> mismatch (no single 0.4% threshold)")
def t_small_err(cfg):
    m = fresh(cfg)
    n = "full_attention_inv_freq" if "full_attention" in m.rope_init_fns else sorted(m._buffers)[0]
    m._buffers[n] = m._buffers[n] * (1 + 2.0 ** -9)
    s = states(m)[n]
    assert s["state"] == "mismatch", s
    return {n: [s["state"], s["max_rel_err_fp32"], s["bf16_representable"]]}


@case("5% error -> mismatch; NaN -> nonfinite; inf -> nonfinite")
def t_scale_nan(cfg):
    out = {}
    for label, f in (("x1.05", lambda b: b * 1.05), ("nan", None), ("inf", None)):
        m = fresh(cfg)
        n = sorted(m._buffers)[0]
        b = m._buffers[n].clone()
        if f:
            b = f(b)
        else:
            b.view(-1)[1] = float(label)
        m._buffers[n] = b
        out[label] = states(m)[n]["state"]
    assert out == {"x1.05": "mismatch", "nan": "nonfinite", "inf": "nonfinite"}, out
    return out


@case("zero handling (proportional RoPE zero padding)")
def t_zeros(cfg):
    m = fresh(cfg)
    zero_bufs = {n: int((b == 0).sum()) for n, b in m._buffers.items() if int((b == 0).sum())}
    assert zero_bufs, "expected zero-padded entries in at least one RoPE buffer (proportional full_attention)"
    n = sorted(zero_bufs)[0]
    b = m._buffers[n].clone()
    zi = int((b == 0).nonzero()[0])
    b[zi] = 1e-9
    m._buffers[n] = b
    s1 = states(m)[n]
    m2 = fresh(cfg)
    b2 = m2._buffers[n].clone()
    nzi = int((b2 != 0).nonzero()[-1])
    b2[nzi] = 0.0
    m2._buffers[n] = b2
    s2 = states(m2)[n]
    s0 = states(fresh(cfg).to(torch.bfloat16))[n]
    assert s1["state"] == "mismatch" and s2["state"] == "mismatch" and s0["state"] == "bf16", (s1, s2, s0)
    return {"zero_entries": zero_bufs, "zero->1e-9": s1["state"], "nonzero->0": [s2["state"], s2["n_zero_mismatch"]],
            "bf16 cast keeps zeros": s0["state"]}


@case("FSDP1 coalesced broadcast: same order OK, reversed order scrambles")
def t_broadcast(cfg):
    src = fresh(cfg, order="sorted")
    same = fresh(cfg, order="sorted")
    rev = fresh(cfg, order="reversed")
    broadcast_coalesced(src, same)
    broadcast_coalesced(src, rev)
    s_same = {k: v["state"] for k, v in states(same).items()}
    s_rev = {k: v["state"] for k, v in states(rev).items()}
    assert all(v == "fp32" for v in s_same.values()), s_same
    assert any(v != "fp32" for v in s_rev.values()), s_rev
    return {"same_order": s_same, "reversed_order": s_rev}


@case("rebuild after bf16 cast -> fp32 bit-exact, CPU oracle agrees")
def t_rebuild(cfg):
    m = fresh(cfg).to(torch.bfloat16)
    rec, errs = mp._g4_check_module(m, rebuild=True)
    assert not errs, errs
    posts = {n: e["post"] for n, e in rec["buffers"].items()}
    assert all(p["bit_exact"] and p["state"] == "fp32" and p["dtype"] == "float32" for p in posts.values()), posts
    assert all(r["cpu_oracle_ok"] for r in rec["reference"].values()), rec["reference"]
    return {"pre": {n: e["pre"]["state"] for n, e in rec["buffers"].items()}, "post": posts}


@case("registration: Gemma-4 model without rotary module raises")
def t_reg_none(cfg):
    class Gemma4NoRotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

    try:
        mp._g4_register_guard(Gemma4NoRotary())
    except RuntimeError as e:
        return str(e)[:120]
    raise AssertionError("no exception")


@case("registration: rotary module -> 1 hook, expected count recorded; non-Gemma model untouched")
def t_reg_ok(cfg):
    h = Holder(fresh(cfg))
    n = mp._g4_register_guard(h)
    assert n == 1 and h._verl_g4_expected_rotary == 1 and len(h.rotary_emb._forward_pre_hooks) == 1
    assert mp._g4_register_guard(torch.nn.Sequential(torch.nn.Linear(2, 2))) == 0
    return {"guarded": n}


@case("init checkpoint (world 1): bf16 buffers -> PASS, rebuilt fp32; first forward PASS")
def t_ckpt_pass(cfg):
    h = Holder(fresh(cfg))
    mp._g4_register_guard(h)
    h.to(torch.bfloat16)
    rep = mp._verl_gemma4_rope_init_checkpoint(h, forward_only=False)
    assert rep["n_fail"] == 0 and rep["ranks"] == [0], rep
    assert all(b.dtype == torch.float32 for b in h.rotary_emb._buffers.values())
    lt = sorted(h.rotary_emb.rope_init_fns)[0]
    h.rotary_emb(torch.zeros(1, 4, 8), torch.arange(4)[None], layer_type=lt)
    assert h.rotary_emb._verl_g4_first_forward_done
    return {"pre_states": rep["pre_states"], "max_rel_err_vs_fp32_accepted_pre": rep["max_rel_err_vs_fp32_accepted_pre"]}


@case("first forward restores buffers changed after the checkpoint")
def t_ff_restore(cfg):
    h = Holder(fresh(cfg))
    mp._g4_register_guard(h)
    mp._verl_gemma4_rope_init_checkpoint(h)
    rot = h.rotary_emb
    n = sorted(rot._buffers)[0]
    rot._buffers[n] = rot._buffers[n].to(torch.bfloat16).to(torch.float32)
    lt = sorted(rot.rope_init_fns)[0]
    rot(torch.zeros(1, 4, 8), torch.arange(4)[None], layer_type=lt)
    ref = rot._verl_g4_refs[n.replace("_original_inv_freq", "").replace("_inv_freq", "")]
    assert torch.equal(rot._buffers[n], ref)
    return {"restored": n}


@case("init checkpoint fault injection (scale, nan) -> RuntimeError naming the rank")
def t_ckpt_fault(cfg):
    out = {}
    for mode in ("scale", "nan"):
        os.environ["VERL_GEMMA4_ROPE_FAULT_RANK"] = "0"
        os.environ["VERL_GEMMA4_ROPE_FAULT_MODE"] = mode
        try:
            h = Holder(fresh(cfg))
            mp._g4_register_guard(h)
            try:
                mp._verl_gemma4_rope_init_checkpoint(h)
                out[mode] = "no exception"
            except RuntimeError as e:
                out[mode] = str(e)[:160]
        finally:
            os.environ.pop("VERL_GEMMA4_ROPE_FAULT_RANK")
            os.environ.pop("VERL_GEMMA4_ROPE_FAULT_MODE")
    assert all("guard FAILED on 1/1 rank(s)" in v and "rank 0" in v for v in out.values()), out
    return out


@case("reverse_order fault reverses the buffer order at registration")
def t_rev_fault(cfg):
    os.environ["VERL_GEMMA4_ROPE_FAULT_RANK"] = "0"
    try:
        h = Holder(fresh(cfg, order="sorted"))
        before = list(h.rotary_emb._buffers)
        mp._g4_register_guard(h)
        after = list(h.rotary_emb._buffers)
    finally:
        os.environ.pop("VERL_GEMMA4_ROPE_FAULT_RANK")
    assert after == before[::-1] and h._verl_g4_fault_applied, (before, after)
    return {"before": before, "after": after}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/workspace/meta-RL/models/gemma-4-E2B-it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.pop("VERL_GEMMA4_ROPE_REPORT_DIR", None)
    cfg = AutoConfig.from_pretrained(args.model)
    cfg = getattr(cfg, "text_config", cfg)
    for t in (t_fresh, t_bf16, t_bf16_fp32, t_small_err, t_scale_nan, t_zeros, t_broadcast, t_rebuild, t_reg_none,
              t_reg_ok, t_ckpt_pass, t_ff_restore, t_ckpt_fault, t_rev_fault):
        t(cfg)
    ok = all(r["ok"] for r in RESULTS)
    for r in RESULTS:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['case']}: {json.dumps(r['detail'], default=str)[:300]}")
    summary = {"verdict": "PASS" if ok else "FAIL", "patch_version": mp.GEMMA4_ROPE_PATCH_VERSION,
               "monkey_patch_file": mp.__file__, "torch": torch.__version__,
               "transformers": __import__("transformers").__version__,
               "tolerances": {"atol": mp._G4_ATOL, "rtol_fp32": mp._G4_RTOL_FP32, "rtol_bf16": mp._G4_RTOL_BF16},
               "cases": RESULTS}
    print(f"GEMMA4-ROPE-GUARD-TEST: {summary['verdict']} ({sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} cases)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
