#!/usr/bin/env python3
"""Single-step replay against the written spec (REPRODUCTION.md v1.0 §1-§4), independent of verl and of Meta's trainer.

Given one pre-update batch dump (tokens, masks, rollout log-probs, rewards, uids) and the initial weights, this script
recomputes on one GPU, in fp32 master / bf16 autocast:
    advantages   A = (R - mean_group) / (std_group[ddof=1] + 1e-6), broadcast over completion tokens
    log-probs    logp_actor(token) from the training graph, teacher-forced
    IS weight    w = min(exp(clamp(logp_actor - logp_rollout, -20, 20)), beta)      (detached)
    ratio        = 1 by construction (old = new.detach(), ppo_epochs = 1) -> clip terms inert
    loss         = sum(-A * w * mask) / N_completion_tokens_global            (loss_agg "token")
    gradient     pre-clip global norm, then clip to max_grad_norm
    update       one AdamW step (lr at this step, betas, eps, weight decay) -> delta_theta
and compares each stage with what the trainer reported for the same batch (advantages, old/actor log-probs, loss,
grad norm, post-update weights). Every comparison is printed with the number of elements and max/mean deviation.

Input format (npz; keys as produced by patch_verl_logprob_fixture.py, also the export spec sent to Meta):
    prompts [B,Lp] int (left-padded), responses [B,Lr] int (right-padded), attention_mask [B,Lp+Lr], response_mask [B,Lr],
    rollout_log_probs [B,Lr] (sampler), old_log_probs [B,Lr] (trainer, pre-update, optional), token_level_scores [B,Lr]
    (sequence reward at the last valid token; the sum over the row is the sequence reward), advantages [B,Lr] (optional),
    nt__uid [B] str (group id). Optional scalars via --reported_loss / --reported_grad_norm; optional --post_weights dir.

Usage:
  python3 replay_single_step.py --dump fixtures/raw/fixture_step1.npz --model /path/Qwen3-0.6B-Base \
      --lr 2e-6 --beta 3.0 --max_grad_norm 1.0 --reported_loss 0.6519 --reported_grad_norm 0.4404 \
      [--post_weights /path/global_step_1/actor/huggingface] [--micro 8] [--out replay_report.json]
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM


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


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--model", required=True, help="initial (pre-update) HF checkpoint dir")
  ap.add_argument("--lr", type=float, required=True, help="learning rate AT THIS STEP (e.g. 2e-6 at step 1 of a 10-step warmup to 2e-5)")
  ap.add_argument("--betas", default="0.9,0.999"); ap.add_argument("--eps", type=float, default=1e-8); ap.add_argument("--weight_decay", type=float, default=0.0)
  ap.add_argument("--beta", type=float, default=3.0, help="IS truncation (token_truncate)"); ap.add_argument("--no_is", action="store_true")
  ap.add_argument("--max_grad_norm", type=float, default=1.0)
  ap.add_argument("--micro", type=int, default=8)
  ap.add_argument("--reported_loss", type=float, default=None); ap.add_argument("--reported_grad_norm", type=float, default=None)
  ap.add_argument("--post_weights", default=None, help="HF dir of the weights after this single update (to compare delta_theta)")
  ap.add_argument("--out", default="replay_report.json")
  args = ap.parse_args()
  dev = "cuda"
  z = np.load(args.dump, allow_pickle=False)
  P, R, AM, RM = z["prompts"], z["responses"], z["attention_mask"], z["response_mask"]
  LS = z["rollout_log_probs"]
  B, Lp = P.shape; Lr = R.shape[1]
  resp_len = RM.sum(1).astype(int); prompt_len = AM[:, :Lp].sum(1).astype(int)
  uids = z["nt__uid"].astype(str)
  seq_reward = z["token_level_scores"].sum(1).astype(np.float64)
  rep = {"dump": args.dump, "B": int(B), "n_tokens": int(resp_len.sum())}

  # ---- 1. advantages from the spec
  A_seq, n_groups = group_advantages(seq_reward, uids)
  rep["n_groups"] = n_groups
  if "advantages" in z.files:
    A_dump = np.array([z["advantages"][i, 0] for i in range(B)], dtype=np.float64)
    rep["advantage_vs_dump"] = {"max_abs": float(np.abs(A_seq - A_dump).max()), "mean_abs": float(np.abs(A_seq - A_dump).mean())}
    print(f"[1] advantages: {n_groups} groups | spec vs dump max|d| {rep['advantage_vs_dump']['max_abs']:.2e}")
  else:
    print(f"[1] advantages: {n_groups} groups (no dumped advantages to compare)")

  # ---- 2. forward/backward with the global token divisor
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
  model.gradient_checkpointing_enable(); model.train(); model.config.use_cache = False
  theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
  N = float(resp_len.sum())
  logp_all = np.zeros_like(LS, dtype=np.float64); loss_total = 0.0; is_stats = []
  order = np.argsort(-resp_len)                                   # long first: stable memory
  t0 = time.time()
  for s in range(0, B, args.micro):
    idx = order[s:s + args.micro]
    seqs = [np.concatenate([P[i, Lp - prompt_len[i]:], R[i, :resp_len[i]]]) for i in idx]
    T = max(len(q) for q in seqs)
    ids = torch.full((len(idx), T), 0, dtype=torch.long); am = torch.zeros((len(idx), T), dtype=torch.long)
    for k, q in enumerate(seqs):
      ids[k, :len(q)] = torch.tensor(q); am[k, :len(q)] = 1
    ids, am = ids.to(dev), am.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
    lp = torch.log_softmax(logits.float()[:, :-1], dim=-1).gather(2, ids[:, 1:, None])[:, :, 0]   # position t predicts t+1
    loss_mb = 0.0
    for k, i in enumerate(idx):
      pl, rl = int(prompt_len[i]), int(resp_len[i])
      lpk = lp[k, pl - 1:pl - 1 + rl]                                                          # response tokens
      logp_all[i, :rl] = lpk.detach().double().cpu().numpy()
      w = torch.ones_like(lpk)
      if not args.no_is:
        lr_ = torch.tensor(LS[i, :rl], device=dev, dtype=torch.float32)
        w = torch.clamp(torch.exp(torch.clamp(lpk.detach() - lr_, -20.0, 20.0)), max=args.beta)   # detached
        is_stats.append(w.cpu().numpy())
      ratio = torch.exp(lpk - lpk.detach())                                                    # == 1, carries the gradient
      loss_mb = loss_mb + (-(float(A_seq[i]) * ratio * w)).sum() / N
    loss_mb.backward()
    loss_total += float(loss_mb.detach())
  gn = float(torch.norm(torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None])))
  rep["loss"] = loss_total; rep["grad_norm_preclip"] = gn
  print(f"[2] loss {loss_total:.6f} | pre-clip grad norm {gn:.6f} | {time.time() - t0:.0f}s")
  if is_stats:
    wv = np.concatenate(is_stats); rep["is"] = {"mean": float(wv.mean()), "std": float(wv.std()), "ess": float(wv.sum() ** 2 / (wv.size * (wv ** 2).sum())), "trunc_frac": float((wv >= args.beta).mean())}
    print(f"    IS weight mean {rep['is']['mean']:.5f} std {rep['is']['std']:.4f} ESS {rep['is']['ess']:.4f} trunc {rep['is']['trunc_frac']:.2e}")
  if args.reported_loss is not None:
    print(f"    reported loss {args.reported_loss:.6f} -> rel diff {abs(loss_total - args.reported_loss) / max(1e-9, abs(args.reported_loss)):.2e}")
  if args.reported_grad_norm is not None:
    print(f"    reported grad norm {args.reported_grad_norm:.6f} -> rel diff {abs(gn - args.reported_grad_norm) / max(1e-9, args.reported_grad_norm):.2e}")
  if "old_log_probs" in z.files:
    d = np.abs(logp_all - z["old_log_probs"].astype(np.float64)) * RM
    rep["logp_vs_dump"] = {"mean_abs": float(d.sum() / N), "max_abs": float(d.max())}
    print(f"    actor logp vs dumped old_log_probs: mean|d| {rep['logp_vs_dump']['mean_abs']:.2e} max {rep['logp_vs_dump']['max_abs']:.2e} (bf16 autocast, different partition)")

  # ---- 3. clip + one AdamW step
  torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
  b1, b2 = (float(x) for x in args.betas.split(","))
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(b1, b2), eps=args.eps, weight_decay=args.weight_decay, fused=False)
  opt.step()
  with torch.no_grad():
    dth = {n: (p.detach() - theta0[n]) for n, p in model.named_parameters()}
    dnorm = math.sqrt(sum(float((v.float() ** 2).sum()) for v in dth.values()))
  rep["delta_theta_norm"] = dnorm
  print(f"[3] one AdamW step at lr {args.lr:g}: ||delta_theta|| {dnorm:.4e} (Adam step 1 ~ lr*sign(g): expected ~ lr*sqrt(n_params) = {args.lr * math.sqrt(sum(p.numel() for p in model.parameters())):.3e})")

  # ---- 4. compare with the trainer's post-update weights
  if args.post_weights:
    post = AutoModelForCausalLM.from_pretrained(args.post_weights, torch_dtype=torch.float32)
    sd = post.state_dict(); num = den = dot = 0.0; per = {}
    with torch.no_grad():
      for n, v in dth.items():
        if n not in sd:
          continue
        d_run = (sd[n].to(dev) - theta0[n]).float(); d_rep = v.float()
        num += float(((d_rep - d_run) ** 2).sum()); den += float((d_run ** 2).sum()); dot += float((d_rep * d_run).sum())
        if any(k in n for k in ("embed_tokens", "lm_head", "layers.0.self_attn.q_proj", "layers.13.mlp.down_proj", "norm.weight")):
          per[n] = {"rel_err": float((d_rep - d_run).norm() / max(1e-12, d_run.norm())), "cos": float((d_rep * d_run).sum() / max(1e-12, d_rep.norm() * d_run.norm()))}
    rep["delta_theta_vs_run"] = {"rel_err": math.sqrt(num / max(den, 1e-30)), "cosine": dot / max(1e-30, math.sqrt(den) * dnorm), "per_tensor": per}
    print(f"[4] delta_theta replay vs run: rel err {rep['delta_theta_vs_run']['rel_err']:.3e} cosine {rep['delta_theta_vs_run']['cosine']:.6f}")
    for n, v in per.items():
      print(f"    {n:<48} rel_err {v['rel_err']:.3e} cos {v['cos']:.6f}")
  json.dump(rep, open(args.out, "w"), indent=1)
  print("saved", args.out)


if __name__ == "__main__":
  main()