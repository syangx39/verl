#!/usr/bin/env python3
"""Recover verl's ACTUAL step-1 gradient from the FSDP optimizer checkpoint and compare it, per parameter, with the
reference-implementation gradient saved by replay_single_step.py --save_grad.

Why it works: the first optimizer.step() ran at lr = 0 (weights unchanged) but Adam still updated its moments from
zero:  exp_avg = (1 - beta1) * g1,  exp_avg_sq = (1 - beta2) * g1^2.  With beta1 = 0.9 and no gradient clipping
(pre-clip norm 0.45 < 1.0), g1 = exp_avg / 0.1 exactly.

Checkpoint layout (verl FSDP1, use_orig_params): global_step_1/actor/optim_world_size_W_rank_r.pt holds
{"state": {param_index: {"step", "exp_avg", "exp_avg_sq"}}, "param_groups": [...]}; each exp_avg is the flat 1-D shard
of that parameter held by rank r. Concatenating ranks 0..W-1 in order gives the flattened parameter (FSDP pads the
tail to a multiple of W); param_index follows named_parameters() order of the HF model (tied lm_head not listed).

Self-checks: per index, sum of shard sizes == numel (+ tail padding < W*?); exp_avg_sq == (1-beta2)/(1-beta1)^2 *
exp_avg^2 elementwise (validates beta1/beta2, step == 1, no clipping, and the index mapping); param_groups lr/betas.

Usage:
  python3 verl_grad_from_optim.py --actor_dir <ckpt>/global_step_1/actor --model <HF init dir> \
      --replay_grad <replay_grad dir>/grad_step1.safetensors [--beta1 0.9 --beta2 0.999] [--out grad_compare.json]
"""
import argparse
import glob
import json
import os
import re

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--actor_dir", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--replay_grad", required=True)
  ap.add_argument("--beta1", type=float, default=0.9); ap.add_argument("--beta2", type=float, default=0.999)
  ap.add_argument("--out", default="grad_compare.json")
  args = ap.parse_args()

  files = sorted(glob.glob(os.path.join(args.actor_dir, "optim_world_size_*_rank_*.pt")), key=lambda f: int(re.search(r"rank_(\d+)", f).group(1)))
  if not files:
    raise SystemExit(f"no optim_world_size_*_rank_*.pt in {args.actor_dir}")
  W = int(re.search(r"world_size_(\d+)", files[0]).group(1))
  if len(files) != W:
    raise SystemExit(f"found {len(files)} optimizer shards, world size says {W}")
  shards = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
  pg = shards[0]["param_groups"][0]
  st0 = shards[0]["state"][0]
  step0 = int(st0["step"].item()) if torch.is_tensor(st0["step"]) else int(st0["step"])
  print(f"world size {W} | param_groups: lr {pg.get('lr')} betas {pg.get('betas')} eps {pg.get('eps')} wd {pg.get('weight_decay')} | "
        f"state[0]: step {step0}, exp_avg dtype {st0['exp_avg'].dtype}, exp_avg_sq dtype {st0['exp_avg_sq'].dtype}, rank-0 shard numel {st0['exp_avg'].numel()}")
  if st0["exp_avg"].dtype != torch.float32 or st0["exp_avg_sq"].dtype != torch.float32:
    raise SystemExit(f"Adam moments are {st0['exp_avg'].dtype}, not fp32: the recovered gradient would be quantized -- stop and check the optimizer precision config")
  if step0 != 1:
    raise SystemExit(f"optimizer step count is {step0}, expected 1 (first update)")
  if tuple(pg.get("betas", ())) != (args.beta1, args.beta2):
    raise SystemExit(f"checkpoint betas {pg.get('betas')} != expected {(args.beta1, args.beta2)}")

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
  names = [n for n, _ in model.named_parameters()]
  shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
  n_state = len(shards[0]["state"])
  if n_state != len(names):
    raise SystemExit(f"optimizer state has {n_state} params, model has {len(names)} named parameters -> index mapping unknown")
  ref = load_file(args.replay_grad)
  c1, c2 = 1.0 / (1.0 - args.beta1), (1.0 - args.beta2) / (1.0 - args.beta1) ** 2      # g = c1*m ; v = c2 * m^2

  rows = []; num = den = dot = 0.0; ref_sq = 0.0; consistency_worst = 0.0
  for idx, name in enumerate(names):
    parts = []
    steps = set()
    for r in range(W):
      st = shards[r]["state"].get(idx)
      if st is None:
        continue
      steps.add(int(st["step"]) if not torch.is_tensor(st["step"]) else int(st["step"].item()))
      parts.append((st["exp_avg"].float().flatten(), st["exp_avg_sq"].float().flatten()))
    if steps != {1}:
      raise SystemExit(f"{name}: optimizer step count {steps}, expected {{1}}")
    m = torch.cat([p[0] for p in parts]); v = torch.cat([p[1] for p in parts])
    numel = 1
    for d in shapes[name]:
      numel *= d
    if m.numel() < numel or m.numel() - numel >= W * 256:
      raise SystemExit(f"{name} (index {idx}): shards give {m.numel()} elements, parameter has {numel} -> index mapping or padding assumption wrong")
    if m.numel() > numel and float(m[numel:].abs().max()) != 0.0:
      raise SystemExit(f"{name}: non-zero values in the presumed padding tail")
    m, v = m[:numel], v[:numel]
    cons = float((v - c2 * m * m).abs().max() / (v.abs().max() + 1e-30))
    consistency_worst = max(consistency_worst, cons)
    g_run = (c1 * m).reshape(shapes[name])
    if name not in ref:
      raise SystemExit(f"{name} missing from the replay gradient file")
    g_ref = ref[name].float()
    if tuple(g_ref.shape) != tuple(g_run.shape):
      raise SystemExit(f"{name}: replay grad shape {tuple(g_ref.shape)} != {tuple(g_run.shape)}")
    d = g_ref - g_run
    nr, nn = float(g_run.norm()), float(g_ref.norm())
    cos = float((g_ref * g_run).sum() / max(1e-30, nr * nn))
    rows.append({"name": name, "numel": numel, "run_norm": nr, "replay_norm": nn, "norm_ratio": nn / max(nr, 1e-30), "cos": cos,
                 "rel_err": float(d.norm() / max(nr, 1e-30)), "adam_consistency": cons})
    num += float((d * d).sum()); den += nr * nr; dot += float((g_ref * g_run).sum()); ref_sq += nn * nn

  tot = {"global_cos": dot / max(1e-30, (den ** 0.5) * (ref_sq ** 0.5)), "global_rel_err": (num / max(den, 1e-30)) ** 0.5,
         "run_grad_norm": den ** 0.5, "replay_grad_norm": ref_sq ** 0.5, "adam_consistency_worst": consistency_worst}
  print(f"\nverl step-1 gradient (from exp_avg/{1 - args.beta1:g}) vs replay gradient: global cosine {tot['global_cos']:.6f} | rel err {tot['global_rel_err']:.3e} | "
        f"norms run {tot['run_grad_norm']:.6f} replay {tot['replay_grad_norm']:.6f} | Adam identity worst rel dev {consistency_worst:.1e} (expect ~1e-6)")
  # per-type aggregation
  def kind(n):
    if "embed_tokens" in n: return "embed"
    if n.endswith("norm.weight") or "layernorm" in n: return "norm"
    if "self_attn" in n: return "attn"
    if "mlp" in n: return "mlp"
    return "other"
  agg = {}
  for r in rows:
    a = agg.setdefault(kind(r["name"]), {"n": 0, "cos_min": 1.0, "cos_mean": 0.0, "rel_err_max": 0.0})
    a["n"] += 1; a["cos_min"] = min(a["cos_min"], r["cos"]); a["cos_mean"] += r["cos"]; a["rel_err_max"] = max(a["rel_err_max"], r["rel_err"])
  for k, a in agg.items():
    a["cos_mean"] /= a["n"]
    print(f"  {k:<6} n={a['n']:>3} cos min {a['cos_min']:.5f} mean {a['cos_mean']:.5f} | rel_err max {a['rel_err_max']:.3e}")
  worst = sorted(rows, key=lambda r: r["cos"])[:8]
  print("  lowest-cosine tensors:")
  for r in worst:
    print(f"    {r['name']:<52} cos {r['cos']:.5f} rel_err {r['rel_err']:.3e} norm_ratio {r['norm_ratio']:.4f} numel {r['numel']}")
  json.dump({"total": tot, "by_kind": agg, "per_param": rows}, open(args.out, "w"), indent=1)
  print("saved", args.out)


if __name__ == "__main__":
  main()