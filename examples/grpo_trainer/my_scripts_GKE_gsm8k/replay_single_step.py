#!/usr/bin/env python3
"""Multi-step replay against the written spec (REPRODUCTION.md v1.0 §1-§4), independent of verl and of Meta's trainer.

Replays K consecutive optimizer steps from the initial weights, one batch dump per step, on one GPU in fp32 master /
bf16 autocast, and compares every stage with what the trainer reported for the same batches:
    advantages   A = (R - mean_group) / (std_group[ddof=1] + 1e-6), broadcast over completion tokens (all valid tokens compared)
    log-probs    logp_actor(token) from the training graph, teacher-forced  (compared FIRST: HF autocast vs FSDP path differ)
    IS weight    w = min(exp(clamp(logp_actor - logp_rollout, -20, 20)), beta)      (detached)
    ratio        = 1 by construction (old = new.detach(), ppo_epochs = 1) -> clip terms inert
    loss         = sum(-A * w * mask) / N_completion_tokens_global            (loss_agg "token")
    gradient     pre-clip global norm, then clip to max_grad_norm
    update       AdamW step with the lr ACTUALLY used by that update; Adam state carried across the replayed steps
Why K steps: verl (like HF's warmup schedulers) runs the FIRST optimizer.step() at lr = lambda(0) = 0 and logs the lr
AFTER the scheduler step, so the logged `actor/lr` at step 1 (2e-6) is the lr of step 2. Replaying steps 1 and 2 from
theta_0 with --lrs 0 2e-6 reproduces Adam's exp_avg / exp_avg_sq / step count without loading optimizer state, and the
post-step-2 weights can be compared with the trainer's global_step_2 checkpoint. Step 1 alone checks loss and gradient.

Input: one npz per step, keys as produced by patch_verl_logprob_fixture.py (also the export spec sent to Meta):
    prompts [B,Lp] int (left-padded), responses [B,Lr] int (right-padded), attention_mask [B,Lp+Lr], response_mask [B,Lr],
    rollout_log_probs [B,Lr], old_log_probs [B,Lr] (optional), token_level_scores [B,Lr], advantages [B,Lr] (optional),
    nt__uid [B] str; sidecar <dump>.json (from the patch) with sha256.model.safetensors, batch_size, global_step.
Integrity gates: the sidecar's model hash must equal --model's model.safetensors, batch = --groups x --group_size rows
with exactly --group_size rows per uid, dumps in ascending global_step. Missing parameter names in --post_weights abort.

Usage:
  python3 replay_single_step.py --dumps raw/fixture_step1.npz raw/fixture_step2.npz --lrs 0 2e-6 \
      --model /path/Qwen3-0.6B-Base-stop --reported_loss <pg_loss@1> <pg_loss@2> --reported_grad_norm <gn@1> <gn@2> \
      --post_weights /path/global_step_2/actor/huggingface [--groups 128 --group_size 16] [--micro 8] [--out replay_report.json]
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def check_stats(st, where, tol_cos=1e-9, tol_res=1e-9):
  """Abort on non-finite values, |cos| > 1 (beyond float64 rounding) or an inconsistent residual. Zero-norm inputs are
  reported as 'degenerate' (cos undefined) rather than compared."""
  if not (math.isfinite(st["norm_a"]) and math.isfinite(st["norm_b"])):   # finiteness first: a zero norm next to NaN/Inf must not pass as 'degenerate'
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
  """Chunked float64 accumulation of sum(a^2), sum(b^2), sum(a*b), sum((a-b)^2) over two tensors of the same shape.
  Returns dict(norm_a, norm_b, cos, rel_err_vs_a, dot, sq_a, sq_b, sq_diff). cos and rel_err come from the SAME
  accumulators (rel_err^2 == 1 + r^2 - 2 r cos with r = norm_b/norm_a), and nothing is clamped."""
  fa, fb = a.reshape(-1), b.reshape(-1)
  if fa.numel() != fb.numel():
    raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
  sa = sb = sab = sd = 0.0
  for i in range(0, fa.numel(), chunk):
    x = fa[i:i + chunk].double(); y = fb[i:i + chunk].double()
    sa += float((x * x).sum()); sb += float((y * y).sum()); sab += float((x * y).sum()); sd += float(((x - y) ** 2).sum())
  na, nb = sa ** 0.5, sb ** 0.5
  return {"sq_a": sa, "sq_b": sb, "dot": sab, "sq_diff": sd, "norm_a": na, "norm_b": nb,
          "cos": sab / (na * nb) if na > 0 and nb > 0 else float("nan"), "rel_err_vs_a": (sd ** 0.5) / na if na > 0 else float("nan")}


def group_advantages(seq_reward, uids, eps=1e-6):
  A = np.zeros_like(seq_reward, dtype=np.float64)
  groups = {}
  for i, u in enumerate(uids):
    groups.setdefault(u, []).append(i)
  for idx in groups.values():
    r = seq_reward[idx]
    if len(idx) > 1:
      A[idx] = (r - r.mean()) / (r.std(ddof=1) + eps)
    else:
      A[idx] = 0.0
  return A, len(groups)


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def load_checkpoint_tensors(ckpt_dir):
  """All tensors of an HF safetensors checkpoint (single file or sharded), by name, without instantiating a model."""
  from safetensors.torch import load_file
  idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
  files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
  out = {}
  for f in files:
    fp = os.path.join(ckpt_dir, f)
    if not os.path.exists(fp):
      raise SystemExit(f"checkpoint shard missing: {fp}")
    out.update(load_file(fp))
  if not out:
    raise SystemExit(f"no tensors found in {ckpt_dir}")
  return out


def load_dump(path, groups, group_size, model_hash):
  z = np.load(path, allow_pickle=False)
  meta_path = path[:-4] + ".json"
  meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
  need = ["prompts", "responses", "attention_mask", "response_mask", "rollout_log_probs", "token_level_scores", "nt__uid"]
  missing = [k for k in need if k not in z.files]
  if missing:
    raise SystemExit(f"{path}: missing keys {missing}")
  d = {k: z[k] for k in z.files}
  B = d["prompts"].shape[0]
  uids = d["nt__uid"].astype(str)
  counts = {}
  for u in uids:
    counts[u] = counts.get(u, 0) + 1
  if B != groups * group_size or len(counts) != groups or any(c != group_size for c in counts.values()):
    raise SystemExit(f"{path}: batch is {B} rows / {len(counts)} groups, expected {groups} x {group_size} with {group_size} rows per uid")
  h = (meta.get("sha256") or {}).get("model.safetensors")
  if h is None:
    raise SystemExit(f"{meta_path}: no model hash in the sidecar (dump not produced by the fixture patch?)")
  if h != model_hash:
    raise SystemExit(f"{path}: dump was collected on model {h[:12]}..., but --model is {model_hash[:12]}... -- not the same run/checkpoint")
  step = meta.get("global_step")
  return d, meta, int(step) if step is not None else None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dumps", nargs="+", required=True, help="one npz per consecutive step, ascending")
  ap.add_argument("--lrs", nargs="+", type=float, required=True, help="lr ACTUALLY used by each replayed update (verl: 0 for step 1, then the logged actor/lr of the previous step)")
  ap.add_argument("--model", required=True, help="initial (pre-update) HF checkpoint dir; must match the dumps' model hash")
  ap.add_argument("--groups", type=int, default=128); ap.add_argument("--group_size", type=int, default=16)
  ap.add_argument("--betas", default="0.9,0.999"); ap.add_argument("--eps", type=float, default=1e-8); ap.add_argument("--weight_decay", type=float, default=0.0)
  ap.add_argument("--beta", type=float, default=3.0, help="IS truncation (token_truncate)"); ap.add_argument("--no_is", action="store_true")
  ap.add_argument("--is_from_dump", action="store_true",
                  help="fixed-IS, ratio=1 CONTROL: take the IS weight from the dumped old_log_probs (the trainer's own values) instead of the "
                       "reference forward's log-probs. This fixes the IS weights to the trainer's; the PPO ratio is still forced to 1 here "
                       "(verl uses exp(training_logp - old_log_probs)) and advantages are still recomputed (they match the dump to ~1e-7), "
                       "so remaining differences are forward/backward plus the ratio term, not forward/backward alone")
  ap.add_argument("--max_grad_norm", type=float, default=1.0)
  ap.add_argument("--micro", type=int, default=8)
  ap.add_argument("--attn", default=None, help="HF attn_implementation for the reference forward (eager|sdpa|flash_attention_2); default = HF's choice. Use two different values to measure the kernel-level numerics floor")
  ap.add_argument("--reported_loss", nargs="*", type=float, default=None, help="trainer's pg_loss per replayed step")
  ap.add_argument("--reported_grad_norm", nargs="*", type=float, default=None, help="trainer's pre-clip grad norm per replayed step")
  ap.add_argument("--post_weights", default=None, help="HF dir of the trainer's weights after the LAST replayed step")
  ap.add_argument("--save_grad", default=None, help="save the PRE-CLIP gradient of the FIRST replayed step, per parameter, as safetensors (for comparison with the trainer's exp_avg/(1-beta1))")
  ap.add_argument("--save_post", default=None, help="save the replayed weights after the last step as an HF dir (model.safetensors), e.g. for a replay-vs-replay numerics floor")
  ap.add_argument("--out", default="replay_report.json")
  args = ap.parse_args()
  if len(args.lrs) != len(args.dumps):
    raise SystemExit("--lrs must have one value per dump")
  dev = "cuda"
  model_hash = sha256(os.path.join(args.model, "model.safetensors"))
  dumps = [load_dump(p, args.groups, args.group_size, model_hash) for p in args.dumps]
  steps = [st for _, _, st in dumps]
  if steps != list(range(1, len(dumps) + 1)):
    raise SystemExit(f"replay must start at step 1 and be consecutive (model and Adam state are initialized from theta0); got {steps}")
  runs = {m.get("experiment_name") for _, m, _ in dumps}
  if len(runs) != 1 or None in runs:
    raise SystemExit(f"all dumps must come from the same run (sidecar experiment_name); got {runs}")
  run_name = runs.pop()
  if args.post_weights and run_name not in os.path.abspath(args.post_weights) and "replay" not in os.path.abspath(args.post_weights):
    raise SystemExit(f"--post_weights {args.post_weights} does not belong to run {run_name} (path must contain the experiment name, or 'replay' for a saved replay)")
  print(f"model {args.model} sha256 {model_hash[:16]} | run {run_name} | replaying steps {steps} with lrs {args.lrs}")

  kw = {"attn_implementation": args.attn} if args.attn else {}
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, **kw).to(dev)
  print(f"reference attention implementation: {getattr(model.config, '_attn_implementation', 'default')}")
  model.gradient_checkpointing_enable(); model.train(); model.config.use_cache = False
  theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
  b1, b2 = (float(x) for x in args.betas.split(","))
  opt = torch.optim.AdamW(model.parameters(), lr=args.lrs[0], betas=(b1, b2), eps=args.eps, weight_decay=args.weight_decay, fused=False)
  report = {"model_sha256": model_hash, "steps": []}

  for k, ((d, meta, step), lr) in enumerate(zip(dumps, args.lrs)):
    P, R, AM, RM = d["prompts"], d["responses"], d["attention_mask"], d["response_mask"]
    LS = d["rollout_log_probs"].astype(np.float32)
    B, Lp = P.shape; Lr = R.shape[1]
    resp_len = RM.sum(1).astype(int); prompt_len = AM[:, :Lp].sum(1).astype(int)
    uids = d["nt__uid"].astype(str)
    seq_reward = d["token_level_scores"].sum(1).astype(np.float64)
    N = float(resp_len.sum())
    rep = {"step": step, "lr": lr, "B": int(B), "n_tokens": int(N)}
    print(f"\n===== step {step} (lr {lr:g}, {int(N)} completion tokens; IS weights from {'DUMPED old_log_probs -- fixed-IS, ratio=1 control' if args.is_from_dump else 'reference log-probs'}) =====")

    # [1] advantages on every valid token
    A_seq, n_groups = group_advantages(seq_reward, uids)
    if "advantages" in d:
      A_dump = d["advantages"].astype(np.float64)
      diff = np.abs(A_seq[:, None] - A_dump) * RM
      rep["advantage_vs_dump"] = {"tokens": int(RM.sum()), "max_abs": float(diff.max()), "mean_abs": float(diff.sum() / N)}
      print(f"[1] advantages: {n_groups} groups | vs dump over {int(RM.sum())} valid tokens: max|d| {rep['advantage_vs_dump']['max_abs']:.2e} mean|d| {rep['advantage_vs_dump']['mean_abs']:.2e}")
    else:
      print(f"[1] advantages: {n_groups} groups (no dumped advantages)")

    # [2] forward/backward, global token divisor
    opt.zero_grad(set_to_none=True)
    for g in opt.param_groups:
      g["lr"] = lr
    logp_all = np.zeros_like(LS, dtype=np.float64); loss_total = 0.0; is_stats = []
    order = np.argsort(-resp_len); t0 = time.time()
    for s0 in range(0, B, args.micro):
      idx = order[s0:s0 + args.micro]
      seqs = [np.concatenate([P[i, Lp - prompt_len[i]:], R[i, :resp_len[i]]]) for i in idx]
      T = max(len(q) for q in seqs)
      ids = torch.zeros((len(idx), T), dtype=torch.long); am = torch.zeros((len(idx), T), dtype=torch.long)
      for j, q in enumerate(seqs):
        ids[j, :len(q)] = torch.tensor(q); am[j, :len(q)] = 1
      ids, am = ids.to(dev), am.to(dev)
      with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
      lp = torch.log_softmax(logits.float()[:, :-1], dim=-1).gather(2, ids[:, 1:, None])[:, :, 0]
      loss_mb = 0.0
      for j, i in enumerate(idx):
        pl, rl = int(prompt_len[i]), int(resp_len[i])
        lpk = lp[j, pl - 1:pl - 1 + rl]
        logp_all[i, :rl] = lpk.detach().double().cpu().numpy()
        w = torch.ones_like(lpk)
        if not args.no_is:
          if args.is_from_dump:
            if "old_log_probs" not in d:
              raise SystemExit("--is_from_dump needs old_log_probs in the dump")
            base = torch.tensor(d["old_log_probs"][i, :rl], device=dev, dtype=torch.float32)
          else:
            base = lpk.detach()
          w = torch.clamp(torch.exp(torch.clamp(base - torch.tensor(LS[i, :rl], device=dev), -20.0, 20.0)), max=args.beta)
          is_stats.append(w.cpu().numpy())
        ratio = torch.exp(lpk - lpk.detach())
        loss_mb = loss_mb + (-(float(A_seq[i]) * ratio * w)).sum() / N
      loss_mb.backward()
      loss_total += float(loss_mb.detach())
    gn = math.sqrt(sum(float(torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64) ** 2) for p in model.parameters() if p.grad is not None))
    rep["loss"] = loss_total; rep["grad_norm_preclip"] = gn
    # log-prob comparison first (numerics), then loss / grad
    if "old_log_probs" in d:
      dd = np.abs(logp_all - d["old_log_probs"].astype(np.float64)) * RM
      rep["logp_vs_dump"] = {"mean_abs": float(dd.sum() / N), "p99": float(np.percentile(dd[RM > 0], 99)), "max_abs": float(dd.max())}
      print(f"[2a] actor logp (HF autocast) vs dumped old_log_probs (FSDP): mean|d| {rep['logp_vs_dump']['mean_abs']:.2e} p99 {rep['logp_vs_dump']['p99']:.2e} max {rep['logp_vs_dump']['max_abs']:.2e}  <- read this before the gradient numbers")
    print(f"[2b] loss {loss_total:.6f} | pre-clip grad norm {gn:.6f} | {time.time() - t0:.0f}s")
    if is_stats:
      wv = np.concatenate(is_stats)
      rep["is"] = {"mean": float(wv.mean()), "std": float(wv.std()), "ess": float(wv.sum() ** 2 / (wv.size * (wv ** 2).sum())), "trunc_frac": float((wv >= args.beta).mean())}
      print(f"     IS weight mean {rep['is']['mean']:.5f} std {rep['is']['std']:.4f} ESS {rep['is']['ess']:.4f} trunc {rep['is']['trunc_frac']:.2e}")
    if args.reported_loss and k < len(args.reported_loss):
      rl_ = args.reported_loss[k]; rep["reported_loss"] = rl_
      print(f"     reported loss {rl_:.6f} -> rel diff {abs(loss_total - rl_) / max(1e-9, abs(rl_)):.2e}")
    if args.reported_grad_norm and k < len(args.reported_grad_norm):
      rg = args.reported_grad_norm[k]; rep["reported_grad_norm"] = rg
      print(f"     reported grad norm {rg:.6f} -> rel diff {abs(gn - rg) / max(1e-9, rg):.2e}")

    if k == 0 and args.save_grad:
      from safetensors.torch import save_file
      os.makedirs(args.save_grad, exist_ok=True)
      save_file({n: p.grad.detach().float().cpu().contiguous() for n, p in model.named_parameters() if p.grad is not None},
                os.path.join(args.save_grad, "grad_step1.safetensors"))
      print(f"     saved pre-clip step-1 gradient -> {args.save_grad}/grad_step1.safetensors")
    # [3] clip + AdamW step with the lr actually used
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    opt.step()
    with torch.no_grad():
      dnorm = math.sqrt(sum(float(((p.detach() - theta0[n]).float() ** 2).sum()) for n, p in model.named_parameters()))
    rep["delta_theta_norm_from_theta0"] = dnorm
    print(f"[3] AdamW step at lr {lr:g}: ||theta - theta0|| = {dnorm:.4e}" + ("  (lr 0: weights unchanged, Adam state initialized)" if lr == 0 else ""))
    report["steps"].append(rep)

  # [4] compare with the trainer's weights after the last replayed step -- read the checkpoint tensors directly
  #     (from_pretrained would silently re-initialize missing parameters); tied lm_head resolves to embed_tokens
  if args.post_weights:
    post = load_checkpoint_tensors(args.post_weights)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    num = den = dot = rep_sq = 0.0; per = {}
    with torch.no_grad():
      for n, p in model.named_parameters():
        key = n
        if key not in post and tied and n == "lm_head.weight" and "model.embed_tokens.weight" in post:
          key = "model.embed_tokens.weight"
        if key not in post:
          raise SystemExit(f"parameter {n} missing from the checkpoint tensors in {args.post_weights}")
        if tuple(post[key].shape) != tuple(p.shape):
          raise SystemExit(f"parameter {n}: checkpoint shape {tuple(post[key].shape)} != model shape {tuple(p.shape)}")
        d_run = post[key].to(dev).float() - theta0[n]; d_rep = p.detach() - theta0[n]
        st = pair_stats(d_run, d_rep)                          # chunked float64; a = run, b = replay
        status = check_stats(st, n)
        num += st["sq_diff"]; den += st["sq_a"]; dot += st["dot"]; rep_sq += st["sq_b"]
        if any(t in n for t in ("embed_tokens", "lm_head", "layers.0.self_attn.q_proj", "layers.13.mlp.down_proj", "model.norm.weight")):
          per[n] = {"rel_err": st["rel_err_vs_a"], "cos": st["cos"], "status": status}
    rep_norm = rep_sq ** 0.5
    if den == 0.0 or rep_norm == 0.0:
      report["delta_theta_vs_run"] = {"run_norm": den ** 0.5, "replay_norm": rep_norm, "rel_err": None, "cosine": None, "identity_residual": None, "per_tensor": per}
      print(f"\n[4] delta_theta: run ||.|| {den ** 0.5:.4e} | replay ||.|| {rep_norm:.4e} | cosine N/A | rel err N/A | residual N/A"
            + ("  (run delta is zero: post weights equal theta0 -- an lr-0 step; compare a later step)" if den == 0.0 else ""))
    else:
      gcos = dot / ((den ** 0.5) * rep_norm); grel = (num / den) ** 0.5; r = rep_norm / (den ** 0.5)
      res = abs(grel ** 2 - (1 + r * r - 2 * r * gcos))
      check_stats({"norm_a": den ** 0.5, "norm_b": rep_norm, "cos": gcos, "rel_err_vs_a": grel}, "GLOBAL delta_theta")
      report["delta_theta_vs_run"] = {"run_norm": den ** 0.5, "replay_norm": rep_norm, "rel_err": grel, "cosine": gcos, "identity_residual": res, "per_tensor": per}
      print(f"\n[4] delta_theta (theta_after_last_step - theta0) [chunked float64]: run ||.|| {den ** 0.5:.4e} | replay ||.|| {rep_norm:.4e} | rel err {grel:.3e} | cosine {gcos:.6f} | residual {res:.1e}")
    for n, v in per.items():
      if v["status"] == "ok":
        print(f"    {n:<48} rel_err {v['rel_err']:.3e} cos {v['cos']:.6f}")
      else:
        print(f"    {n:<48} rel_err N/A cos N/A (zero delta on one side)")
  if args.save_post:
    os.makedirs(args.save_post, exist_ok=True)
    model.save_pretrained(args.save_post, safe_serialization=True)
    print(f"saved replayed weights -> {args.save_post}")
  json.dump(report, open(args.out, "w"), indent=1)
  print("saved", args.out)


if __name__ == "__main__":
  main()