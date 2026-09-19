#!/usr/bin/env python3
"""Optimizer-only check: given the trainer's ACTUAL Adam moments after step 2, does PyTorch's AdamW formula reproduce
the trainer's saved theta_2 from theta_0?  This isolates "applying the update" from "forming the gradient".
Scope: one update applied from theta_0 (valid because step 1 ran at lr 0 so theta_1 == theta_0); it does not
generalize to step 3+ without theta_{t-1}. No pass threshold is asserted; the error is reported in fp32 ulps.

Inputs (all from one run):
  theta_0   --model/model.safetensors                          (initial weights; theta_1 == theta_0 because step 1 ran at lr 0)
  m_2, v_2  --actor_dir/optim_world_size_W_rank_r.pt           (Adam state AFTER the second update, step == 2)
  theta_2   --actor_dir/huggingface/model.safetensors          (weights AFTER the second update)

PyTorch AdamW (weight_decay = 0 here, so identical to Adam), evaluated exactly in the order torch does it, in fp32:
  bias1 = 1 - beta1^step,  bias2 = 1 - beta2^step
  step_size = lr / bias1
  denom = sqrt(v) / sqrt(bias2) + eps
  theta_2 = theta_1 * (1 - lr*wd) - step_size * m / denom
with step = 2 and lr = the lr ACTUALLY used by update 2 (2e-6 = logged actor/lr of step 1; the checkpoint's
param_groups lr is already the NEXT one, 4e-6).  Comparison statistics are accumulated in chunked float64.

FSDP layout (use_orig_params=False): optimizer entry i = flat param of FSDP unit i (root = params outside the wrapped
layer class; i>0 = layer i-1), each rank holding a contiguous 1/W shard; concatenated in rank order, tail padding
stripped, split by the unit's parameters in registration order.  Element counts must match exactly.

Usage:
  python3 adam_apply_check.py --model $MODEL_PATH --actor_dir $CKPT_DIR/$EF/global_step_2/actor --lr 2e-6 --step 2 \
      [--beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.0] [--out adam_apply_check.json]
"""
import argparse
import glob
import json
import math
import os
import re

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def check_stats(st, where, tol_cos=1e-9, tol_res=1e-9):
  if not (math.isfinite(st["norm_a"]) and math.isfinite(st["norm_b"])):
    raise SystemExit(f"{where}: non-finite norm {st}")
  if st["norm_a"] == 0.0 or st["norm_b"] == 0.0:
    return "degenerate"
  if any(not math.isfinite(v) for v in (st["cos"], st["rel_err_vs_a"])):
    raise SystemExit(f"{where}: non-finite statistic {st}")
  if abs(st["cos"]) > 1.0 + tol_cos:
    raise SystemExit(f"{where}: |cosine| = {st['cos']!r} > 1 -- accumulation error, refusing to report")
  r = st["norm_b"] / st["norm_a"]
  res = abs(st["rel_err_vs_a"] ** 2 - (1.0 + r * r - 2.0 * r * st["cos"]))
  if res > tol_res * max(1.0, st["rel_err_vs_a"] ** 2):
    raise SystemExit(f"{where}: cosine/rel_err inconsistent (residual {res:.3e})")
  return "ok"


def pair_stats(a, b, chunk=1 << 23):
  fa, fb = a.reshape(-1), b.reshape(-1)
  sa = sb = sab = sd = 0.0; mx = 0.0
  for i in range(0, fa.numel(), chunk):
    x = fa[i:i + chunk].double(); y = fb[i:i + chunk].double()
    sa += float((x * x).sum()); sb += float((y * y).sum()); sab += float((x * y).sum()); sd += float(((x - y) ** 2).sum())
    mx = max(mx, float((x - y).abs().max()))
  na, nb = sa ** 0.5, sb ** 0.5
  return {"sq_a": sa, "sq_b": sb, "dot": sab, "sq_diff": sd, "norm_a": na, "norm_b": nb, "max_abs_diff": mx,
          "cos": sab / (na * nb) if na > 0 and nb > 0 else float("nan"), "rel_err_vs_a": (sd ** 0.5) / na if na > 0 else float("nan")}


def load_ckpt_tensors(d):
  idx = os.path.join(d, "model.safetensors.index.json")
  files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
  out = {}
  for f in files:
    out.update(load_file(os.path.join(d, f)))
  return out


def fsdp_units(model, actor_dir, wrap_class=None):
  cfg_path = os.path.join(actor_dir, "fsdp_config.json")
  if wrap_class is None and os.path.exists(cfg_path):
    m = re.search(r"([A-Za-z0-9]+DecoderLayer)", json.dumps(json.load(open(cfg_path))))
    wrap_class = m.group(1) if m else None
  wrap_class = wrap_class or "Qwen3DecoderLayer"
  layers = [(n, mod) for n, mod in model.named_modules() if type(mod).__name__ == wrap_class]
  owned = set(); units = []
  for ln, mod in layers:
    ps = [(f"{ln}.{pn}", tuple(pp.shape)) for pn, pp in mod.named_parameters()]
    owned.update(n for n, _ in ps); units.append((ln, ps))
  root = [(n, tuple(pp.shape)) for n, pp in model.named_parameters() if n not in owned]
  return [("root", root)] + units


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True); ap.add_argument("--actor_dir", required=True)
  ap.add_argument("--lr", type=float, required=True, help="lr actually used by this update (step 2: 2e-6)")
  ap.add_argument("--step", type=int, default=2)
  ap.add_argument("--beta1", type=float, default=0.9); ap.add_argument("--beta2", type=float, default=0.999)
  ap.add_argument("--eps", type=float, default=1e-8); ap.add_argument("--weight_decay", type=float, default=0.0)
  ap.add_argument("--wrap_class", default=None); ap.add_argument("--out", default="adam_apply_check.json")
  args = ap.parse_args()

  files = sorted(glob.glob(os.path.join(args.actor_dir, "optim_world_size_*_rank_*.pt")), key=lambda f: int(re.search(r"rank_(\d+)", f).group(1)))
  W = int(re.search(r"world_size_(\d+)", files[0]).group(1))
  shards = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
  pg = shards[0]["param_groups"][0]
  st0 = shards[0]["state"][0]
  stp = int(st0["step"].item()) if torch.is_tensor(st0["step"]) else int(st0["step"])
  print(f"world size {W} | checkpoint param_groups lr {pg.get('lr')} (this is the NEXT lr) betas {pg.get('betas')} eps {pg.get('eps')} wd {pg.get('weight_decay')} | "
        f"state step {stp} | moments dtype {st0['exp_avg'].dtype}/{st0['exp_avg_sq'].dtype}")
  if stp != args.step:
    raise SystemExit(f"optimizer state step {stp} != --step {args.step}")
  if st0["exp_avg"].dtype != torch.float32 or st0["exp_avg_sq"].dtype != torch.float32:
    raise SystemExit("moments are not fp32")
  if tuple(pg.get("betas", ())) != (args.beta1, args.beta2) or float(pg.get("eps")) != args.eps or float(pg.get("weight_decay", 0.0)) != args.weight_decay:
    raise SystemExit(f"checkpoint hyperparameters {pg} differ from the arguments")

  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
  theta0 = load_ckpt_tensors(args.model)
  theta2 = load_ckpt_tensors(os.path.join(args.actor_dir, "huggingface"))
  tied = bool(getattr(model.config, "tie_word_embeddings", False))
  units = fsdp_units(model, args.actor_dir, args.wrap_class)
  if len(shards[0]["state"]) != len(units):
    raise SystemExit(f"{len(shards[0]['state'])} optimizer entries vs {len(units)} FSDP units")

  bias1 = 1.0 - args.beta1 ** args.step; bias2 = 1.0 - args.beta2 ** args.step
  step_size = args.lr / bias1; bias2_sqrt = math.sqrt(bias2)
  print(f"AdamW step {args.step}: lr {args.lr:g}, bias1 {bias1:.6f}, bias2 {bias2:.6f}, step_size {step_size:.6e}, wd {args.weight_decay}")

  num = den = dot = pred_sq = 0.0; worst = []; max_abs_theta = 0.0; n_params = 0
  for idx, (uname, params) in enumerate(units):
    m = torch.cat([shards[r]["state"][idx]["exp_avg"].float().flatten() for r in range(W)])
    v = torch.cat([shards[r]["state"][idx]["exp_avg_sq"].float().flatten() for r in range(W)])
    numel = sum(int(torch.tensor(s).prod()) if s else 1 for _, s in params)
    if m.numel() < numel or m.numel() - numel >= W:
      raise SystemExit(f"{uname}: {m.numel()} elements vs {numel}")
    m, v = m[:numel], v[:numel]
    off = 0
    for name, shp in params:
      n_el = int(torch.tensor(shp).prod()) if shp else 1
      key = name if name in theta0 else ("model.embed_tokens.weight" if tied and name == "lm_head.weight" else name)
      if key not in theta0 or key not in theta2:
        raise SystemExit(f"{name} missing from theta_0/theta_2 checkpoints")
      t0 = theta0[key].float().reshape(-1); t2 = theta2[key].float().reshape(-1)
      if t0.numel() != n_el:
        raise SystemExit(f"{name}: checkpoint numel {t0.numel()} != {n_el}")
      mm, vv = m[off:off + n_el], v[off:off + n_el]; off += n_el
      # PyTorch AdamW, fp32, same op order as torch/optim/adamw.py (single-tensor path)
      t1 = t0 * (1.0 - args.lr * args.weight_decay)              # theta_1 == theta_0; decoupled weight decay (0 here)
      denom = (vv.sqrt() / bias2_sqrt).add_(args.eps)
      t2_pred = t1.addcdiv(mm, denom, value=-step_size)
      st = pair_stats(t2 - t0, t2_pred - t0)                       # delta actual (a) vs delta predicted (b)
      status = check_stats(st, name)
      num += st["sq_diff"]; den += st["sq_a"]; dot += st["dot"]; pred_sq += st["sq_b"]
      max_abs_theta = max(max_abs_theta, float((t2.double() - t2_pred.double()).abs().max()))
      worst.append((name, st["rel_err_vs_a"], st["cos"], st["max_abs_diff"], status)); n_params += 1
  gcos = dot / ((den ** 0.5) * (pred_sq ** 0.5)); grel = (num / den) ** 0.5; r = (pred_sq / den) ** 0.5
  res = abs(grel ** 2 - (1 + r * r - 2 * r * gcos))
  check_stats({"norm_a": den ** 0.5, "norm_b": pred_sq ** 0.5, "cos": gcos, "rel_err_vs_a": grel}, "GLOBAL")
  print(f"\ndelta_theta actual (theta_2 - theta_0) vs predicted from the trainer's own moments: ||actual|| {den ** 0.5:.6e} ||pred|| {pred_sq ** 0.5:.6e} | "
        f"cosine {gcos:.9f} | rel err {grel:.3e} | max |theta_2 - theta_2_pred| {max_abs_theta:.3e} | residual {res:.1e}")
  worst.sort(key=lambda x: -x[1])
  print("  largest per-parameter rel err:")
  for name, rel, cos, mx, status in worst[:6]:
    print(f"    {name:<52} rel_err {rel:.3e} cos {cos:.9f} max|d| {mx:.2e} {status}")
  # No calibrated pass threshold: CPU and CUDA optimizer kernels round differently. Report the error against the fp32
  # resolution of the weights (ulp at the largest |theta|) and let the reader judge; a mismatch in lr / bias correction /
  # eps / mapping would show up as a rel err of order 1 or a wrong ||pred|| / ||actual|| ratio, not as a few ulps.
  theta_scale = max(float(t.float().abs().max()) for t in theta0.values())
  ulp = theta_scale * 2.0 ** -23
  verdict = (f"delta rel err {grel:.2e}, ||pred||/||actual|| = {r:.6f}, max |theta_2 - theta_2_pred| = {max_abs_theta:.2e} "
             f"= {max_abs_theta / ulp:.1f} fp32 ulps at |theta|max {theta_scale:.3g}")
  print("SUMMARY:", verdict)
  print("  read as: a few ulps and ||pred||/||actual|| ~ 1 -> consistent with CPU-vs-CUDA rounding given the trainer's moments; "
        "rel err ~O(0.1-1) or a norm ratio far from 1 -> lr / bias correction / eps / mapping / save precision problem")
  json.dump({"global_cos": gcos, "global_rel_err": grel, "max_abs_theta_diff": max_abs_theta, "residual": res, "verdict": verdict,
             "per_param": [{"name": n, "rel_err": rl, "cos": c, "max_abs_diff": m_, "status": s} for n, rl, c, m_, s in worst]}, open(args.out, "w"), indent=1)
  print("saved", args.out)


if __name__ == "__main__":
  main()
