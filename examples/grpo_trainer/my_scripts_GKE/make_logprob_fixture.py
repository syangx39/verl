#!/usr/bin/env python3
"""Build the Step-4 numerics fixture from a REAL verl dump (fixture_step<N>.npz, written by
the _LOGPROB_FIXTURE fork patch during a 1-step run on the initial checkpoint).

Per selected sequence the fixture carries the actual training-time tensors, unpadded:
  prompt_ids, response_ids, response_mask, position_ids (response region), and
  logp_sampler   vLLM sampler log-prob of each sampled token   (raw, T=1 recipe)
  logp_trainer   verl trainer log-prob, pre-update              (the training-time value)
  logp_trainer_repeat  the same trainer pass run a second time (repeatability)
and, for the replay gate, uid / qid / reward fields / advantage.
Optionally (--hf_model) an AUXILIARY HF fp32 teacher-forced log-prob on the same ids is added as
`logp_hf_fp32_aux` -- a precision reference only, NOT a trainer output.

Statistics reported (units stated): trainer-vs-sampler probability MAE (verl's
training/rollout_probs_diff_mean definition) and log-domain |dlogp| / signed dlogp in nats;
trainer repeat error in nats; the HF-fp32-vs-trainer difference (precision, nats) if requested.

Selection: --n sequences stratified by response length so that long and cap-hit (truncated)
responses are included (--n_long of them from the longest quartile, --n_trunc that hit the cap).
Usage:
  python3 make_logprob_fixture.py --dump <dir>/fixture_step1.npz --out <H>/fixtures/logprob_fixture \
      [--n 96 --n_long 24 --n_trunc 8] [--hf_model $MODEL_PATH] [--seed 0]
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random

import numpy as np


def sha256(path):
  return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=96)
  ap.add_argument("--n_long", type=int, default=24)
  ap.add_argument("--n_trunc", type=int, default=8)
  ap.add_argument("--hf_model", default=None, help="add an auxiliary HF fp32 log-prob reference (needs a GPU)")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  z = np.load(args.dump, allow_pickle=False)
  meta_path = args.dump[:-4] + ".json"
  meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
  need = ["prompts", "responses", "attention_mask", "position_ids", "response_mask",
          "rollout_log_probs", "old_log_probs", "old_log_probs_repeat"]
  missing = [k for k in need if k not in z.files]
  if missing:
    raise SystemExit(f"dump lacks {missing}; was calculate_log_probs=True and the _LOGPROB_FIXTURE patch active?")
  P, R, AM, PID, RM = (z[k] for k in ("prompts", "responses", "attention_mask", "position_ids", "response_mask"))
  LS, LT, LR = z["rollout_log_probs"], z["old_log_probs"], z["old_log_probs_repeat"]
  B, Lp = P.shape
  Lr = R.shape[1]
  resp_len = RM.sum(1).astype(int)
  prompt_len = AM[:, :Lp].sum(1).astype(int)
  cap = Lr
  nt = {k[4:]: z[k] for k in z.files if k.startswith("nt__")}

  # ---- selection: stratified by response length
  rng = random.Random(args.seed)
  idx_all = list(range(B))
  trunc = [i for i in idx_all if resp_len[i] >= cap]
  q75 = np.percentile(resp_len, 75)
  longs = [i for i in idx_all if resp_len[i] >= q75 and i not in trunc]
  rest = [i for i in idx_all if i not in trunc and i not in longs]
  pick = rng.sample(trunc, min(args.n_trunc, len(trunc))) + rng.sample(longs, min(args.n_long, len(longs)))
  pick += rng.sample(rest, max(0, args.n - len(pick)))
  pick = sorted(pick)

  hf = None
  if args.hf_model:
    import torch
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.float32).to(dev).eval()

  seqs, stats = [], {"prob_mae": [], "abs_dlogp": [], "signed_dlogp": [], "repeat_abs": [], "hf_abs": []}
  for i in pick:
    pl, rl = int(prompt_len[i]), int(resp_len[i])
    # prompts are LEFT-padded in verl: real prompt tokens are the last pl positions; responses are right-padded
    p_ids = P[i, Lp - pl:].tolist()
    r_ids = R[i, :rl].tolist()
    r_mask = RM[i, :rl].astype(int).tolist()
    pos = PID[i, Lp:Lp + rl].tolist() if PID.ndim == 2 else PID[i, ..., Lp:Lp + rl].tolist()
    ls, lt, lr = LS[i, :rl].astype(float), LT[i, :rl].astype(float), LR[i, :rl].astype(float)
    for name, arr in (("sampler", ls), ("trainer", lt), ("trainer_repeat", lr)):
      if not np.all(np.isfinite(arr)):
        raise SystemExit(f"non-finite logp in {name} for batch row {i}")
    dl = lt - ls
    stats["prob_mae"].extend(np.abs(np.exp(lt) - np.exp(ls)).tolist())
    stats["abs_dlogp"].extend(np.abs(dl).tolist())
    stats["signed_dlogp"].extend(dl.tolist())
    stats["repeat_abs"].extend(np.abs(lt - lr).tolist())
    row = {"batch_row": int(i), "prompt_ids": p_ids, "response_ids": r_ids, "response_mask": r_mask,
           "position_ids_response": pos, "n_prompt_tokens": pl, "n_response_tokens": rl, "truncated": bool(rl >= cap),
           "logp_sampler": ls.tolist(), "logp_trainer": lt.tolist(), "logp_trainer_repeat": lr.tolist()}
    for k, v in nt.items():
      val = v[i]
      row[k] = val.item() if hasattr(val, "item") else str(val)
    if "advantages" in z.files:
      row["advantage"] = float(z["advantages"][i, 0])
    if "token_level_scores" in z.files:
      row["sequence_score"] = float(z["token_level_scores"][i].sum())
    if hf is not None:
      import torch
      ids = torch.tensor([p_ids + r_ids], device=next(hf.parameters()).device)
      with torch.no_grad():
        lp = torch.log_softmax(hf(ids).logits[0, :-1].float(), dim=-1)
      tl = lp.gather(1, ids[0, 1:, None])[:, 0][pl - 1:].cpu().numpy()
      row["logp_hf_fp32_aux"] = tl.tolist()
      stats["hf_abs"].extend(np.abs(tl - lt).tolist())
    seqs.append(row)

  def summ(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p50": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(np.abs(x).max()), "n_tokens": int(x.size)} if x.size else None

  summary = {
      "source_dump": os.path.basename(args.dump), "source_dump_sha256": sha256(args.dump),
      "dump_meta": meta,
      "n_sequences": len(seqs), "n_truncated": int(sum(s["truncated"] for s in seqs)),
      "response_len_min_max": [int(min(s["n_response_tokens"] for s in seqs)), int(max(s["n_response_tokens"] for s in seqs))],
      "trainer_vs_sampler_probability_MAE (mean |p_trainer - p_sampler|, verl rollout_probs_diff definition)": summ(stats["prob_mae"]),
      "trainer_vs_sampler_abs_dlogp_nats": summ(stats["abs_dlogp"]),
      "trainer_vs_sampler_signed_dlogp_nats (trainer - sampler)": summ(stats["signed_dlogp"]),
      "trainer_repeat_error_nats (|trainer - trainer_repeat|, same weights/inputs)": summ(stats["repeat_abs"]),
      "hf_fp32_aux_vs_trainer_nats (precision reference only)": summ(stats["hf_abs"]) if stats["hf_abs"] else "not computed",
      "versions": {"python": platform.python_version(),
                   **{p: (md.version(p) if _has(p) else None) for p in ("torch", "vllm", "verl", "transformers", "numpy")}},
      "units": "log-probs in nats; probability MAE dimensionless; GPU 300-step run-level references: prob MAE 0.0057, log-domain 0.0007",
      "how_to_compare": "TPU trainer log-probs on the SAME prompt_ids+response_ids (teacher forced, response_mask, position_ids_response, "
                        "T=1) vs logp_trainer; TPU sampler-vs-trainer on TPU's own rollouts vs the trainer_vs_sampler stats here.",
  }
  if args.hf_model:
    summary["hf_model_safetensors_sha256"] = sha256(os.path.join(args.hf_model, "model.safetensors"))
  with open(args.out + ".json", "w") as f:
    json.dump({"summary": summary, "sequences": seqs}, f)
  print(json.dumps({k: v for k, v in summary.items() if k not in ("dump_meta",)}, indent=1))
  print(f"saved {args.out}.json")


def _has(pkg):
  try:
    md.version(pkg)
    return True
  except md.PackageNotFoundError:
    return False


if __name__ == "__main__":
  main()
