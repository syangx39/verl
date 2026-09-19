#!/usr/bin/env python3
"""Recover verl's ACTUAL step-1 gradient from the FSDP optimizer checkpoint and compare it, per parameter, with the
reference-implementation gradient saved by replay_single_step.py --save_grad.

Why it works: the first optimizer.step() ran at lr = 0 (weights unchanged) but Adam still updated its moments from
zero:  exp_avg = (1 - beta1) * g1,  exp_avg_sq = (1 - beta2) * g1^2.  With beta1 = 0.9, g1 = exp_avg / 0.1 is the
gradient that entered Adam. Whether clipping touched it is NOT decidable from the moments (a scaled gradient satisfies
the same identities); it follows from the logged pre-clip norm 0.45 < max_grad_norm 1.0.

Checkpoint layout (verl FSDP1, use_orig_params=False): optim_world_size_W_rank_r.pt holds
{"state": {flat_param_index: {"step", "exp_avg", "exp_avg_sq"}}, "param_groups": [...]}. Each optimizer entry is one
FSDP unit's FLAT parameter (all of that unit's parameters flattened and concatenated in registration order, padded at
the tail to a multiple of W), and each rank holds its contiguous 1/W shard. Units: index 0 = the root unit (parameters
not inside any wrapped layer: embed_tokens, final norm; lm_head only if untied), index i = decoder layer i-1.
Reconstruction: concatenate ranks 0..W-1 -> strip tail padding -> split by the unit's parameter numels -> reshape.
The unit list is derived from the HF model and the wrap class in fsdp_config.json; every unit's element count must match
exactly (up to < W padding), otherwise the script aborts rather than compare misaligned tensors.

Self-checks: exp_avg dtype fp32 and step == 1; per unit, shard sizes vs numel; exp_avg_sq == (1-beta2)/(1-beta1)^2 *
exp_avg^2 (this only certifies the two moments are consistent with beta1/beta2 at step 1 -- it cannot detect a
consistent mis-ordering or clipping; ordering is certified by the numel accounting, absence of clipping by the logs).

Usage:
  python3 verl_grad_from_optim.py --actor_dir <ckpt>/global_step_1/actor --model <HF init dir> \
      --replay_grad <replay_grad dir>/grad_step1.safetensors [--wrap_class Qwen3DecoderLayer] [--out grad_compare.json]
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
  ap.add_argument("--wrap_class", default=None, help="FSDP wrap class (default: read from fsdp_config.json, else Qwen3DecoderLayer)")
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
  # ---- FSDP units: wrapped layer class from fsdp_config.json, root = everything else
  wrap = args.wrap_class
  cfg_path = os.path.join(args.actor_dir, "fsdp_config.json")
  if wrap is None and os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    txt = json.dumps(cfg)
    m = re.search(r"(Qwen\d*DecoderLayer|[A-Za-z0-9]+DecoderLayer)", txt)
    wrap = m.group(1) if m else None
    print(f"fsdp_config.json: {cfg}")
  wrap = wrap or "Qwen3DecoderLayer"
  layer_mods = [(n, mod) for n, mod in model.named_modules() if type(mod).__name__ == wrap]
  if not layer_mods:
    raise SystemExit(f"no module of class {wrap} in the model")
  units = []
  owned = set()
  for ln, mod in layer_mods:
    params = [(f"{ln}.{pn}", tuple(pp.shape)) for pn, pp in mod.named_parameters()]
    owned.update(n for n, _ in params)
    units.append((ln, params))
  root = [(n, tuple(pp.shape)) for n, pp in model.named_parameters() if n not in owned]
  units.insert(0, ("root", root))
  n_state = len(shards[0]["state"])
  if n_state != len(units):
    raise SystemExit(f"optimizer state has {n_state} flat params, model has {len(units)} FSDP units ({len(layer_mods)} x {wrap} + root) -> wrap policy mismatch")
  print(f"FSDP units: root ({len(root)} params: {[n for n, _ in root]}) + {len(layer_mods)} x {wrap} | optimizer entries {n_state}")
  ref = load_file(args.replay_grad)
  c1, c2 = 1.0 / (1.0 - args.beta1), (1.0 - args.beta2) / (1.0 - args.beta1) ** 2      # g = c1*m ; v = c2 * m^2

  rows = []; num = den = dot = 0.0; ref_sq = 0.0; consistency_worst = 0.0
  for idx, (uname, params) in enumerate(units):
    parts = []
    for r in range(W):
      st = shards[r]["state"].get(idx)
      if st is None:
        raise SystemExit(f"rank {r} has no optimizer state for flat param {idx} ({uname})")
      stp = int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])
      if stp != 1:
        raise SystemExit(f"{uname}: rank {r} optimizer step {stp}, expected 1")
      parts.append((st["exp_avg"].float().flatten(), st["exp_avg_sq"].float().flatten()))
    m = torch.cat([p[0] for p in parts]); v = torch.cat([p[1] for p in parts])
    numel = sum(int(torch.tensor(shp).prod()) if shp else 1 for _, shp in params)
    pad = m.numel() - numel
    if pad < 0 or pad >= W:
      raise SystemExit(f"{uname} (flat param {idx}): shards give {m.numel()} elements, unit has {numel} ({len(params)} params) -> unit composition/order wrong")
    if pad and float(m[numel:].abs().max()) != 0.0:
      raise SystemExit(f"{uname}: non-zero values in the presumed {pad}-element padding tail")
    m, v = m[:numel], v[:numel]
    cons = float((v - c2 * m * m).abs().max() / (v.abs().max() + 1e-30))
    consistency_worst = max(consistency_worst, cons)
    off = 0
    for name, shp in params:
      n_el = int(torch.tensor(shp).prod()) if shp else 1
      g_run = (c1 * m[off:off + n_el]).reshape(shp); off += n_el
      if name not in ref:
        raise SystemExit(f"{name} missing from the replay gradient file")
      g_ref = ref[name].float()
      if tuple(g_ref.shape) != tuple(g_run.shape):
        raise SystemExit(f"{name}: replay grad shape {tuple(g_ref.shape)} != {tuple(g_run.shape)}")
      d = g_ref - g_run
      nr, nn = float(g_run.norm()), float(g_ref.norm())
      cos = float((g_ref * g_run).sum() / max(1e-30, nr * nn))
      rows.append({"name": name, "unit": uname, "numel": n_el, "run_norm": nr, "replay_norm": nn, "norm_ratio": nn / max(nr, 1e-30), "cos": cos,
                   "rel_err": float(d.norm() / max(nr, 1e-30)), "adam_consistency": cons})
      num += float((d * d).sum()); den += nr * nr; dot += float((g_ref * g_run).sum()); ref_sq += nn * nn
    print(f"  unit {idx:>2} {uname:<22} {len(params):>2} params, {numel:>10} elements, padding {pad} -> ok")

  tot = {"global_cos": dot / max(1e-30, (den ** 0.5) * (ref_sq ** 0.5)), "global_rel_err": (num / max(den, 1e-30)) ** 0.5,
         "run_grad_norm": den ** 0.5, "replay_grad_norm": ref_sq ** 0.5, "adam_consistency_worst": consistency_worst}
  print(f"\nverl step-1 gradient (from exp_avg/{1 - args.beta1:g}) vs replay gradient: global cosine {tot['global_cos']:.6f} | rel err {tot['global_rel_err']:.3e} | "
        f"norms run {tot['run_grad_norm']:.6f} replay {tot['replay_grad_norm']:.6f} | Adam moment identity worst rel dev {consistency_worst:.1e} (expect ~1e-6; certifies beta/step consistency only)")
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