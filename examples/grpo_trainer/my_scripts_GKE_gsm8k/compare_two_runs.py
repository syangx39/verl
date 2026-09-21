#!/usr/bin/env python3
"""Direct comparison of two native-trainer runs that trained on IDENTICAL tokens (batch injection):
  - step-1 gradient of each run recovered from its Adam moments (exp_avg / (1 - beta1); first update at lr 0, no clipping)
  - post-update weights after the last common checkpoint step (delta_theta from theta_0)
Both runs must share theta_0 (same --model), the same FSDP layout, and must have trained on the same injected batch.

Usage:
  python3 compare_two_runs.py --model $MODEL_PATH --run_a <ckpt>/<RUN_A> --run_b <ckpt>/<RUN_B> --label_a single-forward --label_b two-pass \
      [--grad_step 1] [--delta_step 2] [--beta1 0.9] [--out compare_two_runs.json]
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
    raise SystemExit(f"{where}: |cosine| = {st['cos']!r} > 1")
  r = st["norm_b"] / st["norm_a"]
  if abs(st["rel_err_vs_a"] ** 2 - (1.0 + r * r - 2.0 * r * st["cos"])) > tol_res * max(1.0, st["rel_err_vs_a"] ** 2):
    raise SystemExit(f"{where}: cosine/rel_err inconsistent")
  return "ok"


def pair_stats(a, b, chunk=1 << 23):
  fa, fb = a.reshape(-1), b.reshape(-1)
  sa = sb = sab = sd = 0.0
  for i in range(0, fa.numel(), chunk):
    x = fa[i:i + chunk].double(); y = fb[i:i + chunk].double()
    sa += float((x * x).sum()); sb += float((y * y).sum()); sab += float((x * y).sum()); sd += float(((x - y) ** 2).sum())
  na, nb = sa ** 0.5, sb ** 0.5
  return {"sq_a": sa, "sq_b": sb, "dot": sab, "sq_diff": sd, "norm_a": na, "norm_b": nb,
          "cos": sab / (na * nb) if na > 0 and nb > 0 else float("nan"), "rel_err_vs_a": (sd ** 0.5) / na if na > 0 else float("nan")}


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


def load_grads(actor_dir, model, beta1):
  files = sorted(glob.glob(os.path.join(actor_dir, "optim_world_size_*_rank_*.pt")), key=lambda f: int(re.search(r"rank_(\d+)", f).group(1)))
  W = int(re.search(r"world_size_(\d+)", files[0]).group(1))
  shards = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
  st0 = shards[0]["state"][0]
  step = int(st0["step"].item()) if torch.is_tensor(st0["step"]) else int(st0["step"])
  if step != 1 or st0["exp_avg"].dtype != torch.float32:
    raise SystemExit(f"{actor_dir}: optimizer step {step} / dtype {st0['exp_avg'].dtype}; need step 1 fp32 moments")
  units = fsdp_units(model, actor_dir)
  if len(shards[0]["state"]) != len(units):
    raise SystemExit(f"{actor_dir}: {len(shards[0]['state'])} optimizer entries vs {len(units)} units")
  grads = {}
  for idx, (uname, params) in enumerate(units):
    m = torch.cat([shards[r]["state"][idx]["exp_avg"].float().flatten() for r in range(W)])
    numel = sum(int(torch.tensor(s).prod()) if s else 1 for _, s in params)
    if m.numel() < numel or m.numel() - numel >= W:
      raise SystemExit(f"{uname}: {m.numel()} vs {numel}")
    m = m[:numel]; off = 0
    for name, shp in params:
      n_el = int(torch.tensor(shp).prod()) if shp else 1
      grads[name] = (m[off:off + n_el] / (1.0 - beta1)).reshape(shp); off += n_el
  return grads


def load_weights(hf_dir):
  idx = os.path.join(hf_dir, "model.safetensors.index.json")
  files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
  out = {}
  for f in files:
    out.update(load_file(os.path.join(hf_dir, f)))
  return out


def compare(dict_a, dict_b, names, tied_alias, title):
  num = den = dot = sq_b = 0.0; rows = []
  for n in names:
    ka = n if n in dict_a else tied_alias.get(n, n); kb = n if n in dict_b else tied_alias.get(n, n)
    if ka not in dict_a or kb not in dict_b:
      raise SystemExit(f"{title}: {n} missing")
    st = pair_stats(dict_a[ka].float(), dict_b[kb].float()); status = check_stats(st, n)
    num += st["sq_diff"]; den += st["sq_a"]; dot += st["dot"]; sq_b += st["sq_b"]
    rows.append((n, st["cos"], st["rel_err_vs_a"], status))
  if den == 0.0 or sq_b == 0.0:
    print(f"{title}: zero on one side -> N/A"); return {"cos": None, "rel_err": None}
  gcos = dot / (den ** 0.5 * sq_b ** 0.5); grel = (num / den) ** 0.5
  check_stats({"norm_a": den ** 0.5, "norm_b": sq_b ** 0.5, "cos": gcos, "rel_err_vs_a": grel}, title)
  print(f"{title}: ||a|| {den ** 0.5:.6e} ||b|| {sq_b ** 0.5:.6e} | cosine {gcos:.6f} | rel err {grel:.3e}")
  worst = sorted([r for r in rows if r[3] == "ok"], key=lambda r: r[1])[:5]
  for n, c, e, _ in worst:
    print(f"    lowest: {n:<52} cos {c:.5f} rel_err {e:.3e}")
  return {"cos": gcos, "rel_err": grel, "norm_a": den ** 0.5, "norm_b": sq_b ** 0.5, "per_param": [{"name": n, "cos": c, "rel_err": e, "status": s} for n, c, e, s in rows]}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True); ap.add_argument("--run_a", required=True); ap.add_argument("--run_b", required=True)
  ap.add_argument("--label_a", default="A"); ap.add_argument("--label_b", default="B")
  ap.add_argument("--grad_step", type=int, default=1); ap.add_argument("--delta_step", type=int, default=2)
  ap.add_argument("--beta1", type=float, default=0.9); ap.add_argument("--out", default="compare_two_runs.json")
  args = ap.parse_args()
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
  names = [n for n, _ in model.named_parameters()]
  tied = {"lm_head.weight": "model.embed_tokens.weight"} if getattr(model.config, "tie_word_embeddings", False) else {}
  theta0 = load_weights(args.model)
  report = {"model": args.model, "run_a": args.run_a, "run_b": args.run_b}

  ga = load_grads(os.path.join(args.run_a, f"global_step_{args.grad_step}", "actor"), model, args.beta1)
  gb = load_grads(os.path.join(args.run_b, f"global_step_{args.grad_step}", "actor"), model, args.beta1)
  print(f"\n== step-{args.grad_step} gradient: {args.label_a} (a) vs {args.label_b} (b), recovered from Adam moments ==")
  report["gradient"] = compare(ga, gb, names, tied, "gradient")

  wa = load_weights(os.path.join(args.run_a, f"global_step_{args.delta_step}", "actor", "huggingface"))
  wb = load_weights(os.path.join(args.run_b, f"global_step_{args.delta_step}", "actor", "huggingface"))
  da = {n: (wa[n if n in wa else tied.get(n, n)].float() - theta0[n if n in theta0 else tied.get(n, n)].float()) for n in names}
  db = {n: (wb[n if n in wb else tied.get(n, n)].float() - theta0[n if n in theta0 else tied.get(n, n)].float()) for n in names}
  print(f"\n== delta_theta after step {args.delta_step} (theta - theta_0): {args.label_a} (a) vs {args.label_b} (b) ==")
  report["delta_theta"] = compare(da, db, names, {}, "delta_theta")
  json.dump(report, open(args.out, "w"), indent=1)
  print("saved", args.out)


if __name__ == "__main__":
  main()
