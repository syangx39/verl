#!/usr/bin/env python3
"""Log-prob fixture for the trainer-vs-sampler numerics gate.

Takes N real sequences (prompt + the initial model's own greedy output) from a
step-0 validation dump, tokenizes them exactly as the trainer does (chat
template rendered, response tokens appended, EOS included), and records the
per-token log-probability of every response token under the initial checkpoint,
computed with plain HuggingFace forward passes in two precisions:
  fp32  -- deterministic reference anchor (weights + compute in fp32)
  bf16  -- the GPU training-compute precision (fp32 weights cast to bf16 for the
           forward pass, as under FSDP mixed precision)
Both stacks compare their trainer log-probs on these exact token IDs / masks to
the fp32 anchor; |bf16 - fp32| is the GPU self-comparison baseline.

Output: <out>.json with, per sequence: source, question, prompt_token_ids,
response_token_ids, logp_fp32 (list), logp_bf16 (list); plus summary stats.
Usage (on a GPU host):
  python3 make_logprob_fixture.py --model $MODEL_PATH --val_dump <EXP>/val_dump/0.jsonl \
      --data_dir $DATA_DIR --n 64 --out $H/fixtures/logprob_fixture
"""
import argparse
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_WS = re.compile(r"\s+")
norm = lambda s: _WS.sub(" ", s).strip()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--val_dump", required=True, help="a step-0 validation dump (<EXP>/val_dump/0.jsonl)")
  ap.add_argument("--data_dir", required=True)
  ap.add_argument("--n", type=int, default=64)
  ap.add_argument("--max_len", type=int, default=4096, help="skip sequences longer than this (fixture stays small)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  tok = AutoTokenizer.from_pretrained(args.model)
  # map normalized question -> (messages, source) from the eval parquets
  lookup = {}
  for src, fn in (("omi2_val1k", "val_1k_qsplit.parquet"), ("gsm8k", "gsm8k_test.parquet")):
    df = pd.read_parquet(f"{args.data_dir}/{fn}")
    for _, r in df.iterrows():
      lookup[norm(r["extra_info"]["question"])] = ([dict(m) for m in r["prompt"]], src, r["extra_info"]["question"])

  rows = [json.loads(l) for l in open(args.val_dump, encoding="utf-8") if l.strip()]
  random.seed(args.seed)
  random.shuffle(rows)

  # recover the question from the dumped input the same way paired_eval_bootstrap does
  any_msgs = next(iter(lookup.values()))[0][0]["content"]
  q0 = next(iter(lookup.values()))[2]
  k = any_msgs.find(q0)
  prefix_tail, suffix_head = any_msgs[:k][-40:], any_msgs[k + len(q0):][:14]

  seqs = []
  for r in rows:
    inp = r["input"]
    i = inp.rfind(prefix_tail)
    j = inp.find(suffix_head, i if i >= 0 else 0)
    if i < 0 or j < 0:
      continue
    q = norm(inp[i + len(prefix_tail):j])
    if q not in lookup:
      continue
    msgs, src, question = lookup[q]
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tok(r["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(r_ids) > args.max_len:
      continue
    seqs.append({"source": src, "question": question, "prompt_token_ids": p_ids, "response_token_ids": r_ids})
    if len(seqs) >= args.n:
      break
  print(f"selected {len(seqs)} sequences ({sum(s['source']=='gsm8k' for s in seqs)} gsm8k, "
        f"{sum(s['source']=='omi2_val1k' for s in seqs)} omi2); max total len {max(len(s['prompt_token_ids'])+len(s['response_token_ids']) for s in seqs)}")

  dev = "cuda" if torch.cuda.is_available() else "cpu"
  model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()

  @torch.no_grad()
  def logps(seq, dtype):
    ids = torch.tensor([seq["prompt_token_ids"] + seq["response_token_ids"]], device=dev)
    with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)):
      logits = model(ids).logits.float()               # [1, T, V]
    lp = torch.log_softmax(logits[0, :-1], dim=-1)     # position t predicts token t+1
    tgt = ids[0, 1:]
    tok_lp = lp.gather(1, tgt[:, None])[:, 0]
    n_p = len(seq["prompt_token_ids"])
    return tok_lp[n_p - 1:].cpu().tolist()             # log-probs of the response tokens only

  out = []
  diffs = []
  for s in seqs:
    lp32 = logps(s, torch.float32)
    lp16 = logps(s, torch.bfloat16)
    assert len(lp32) == len(s["response_token_ids"])
    d = np.abs(np.array(lp32) - np.array(lp16))
    diffs.extend(d.tolist())
    out.append({**s, "logp_fp32": lp32, "logp_bf16": lp16, "n_response_tokens": len(lp32),
                "mean_logp_fp32": float(np.mean(lp32)), "mean_abs_diff_bf16_vs_fp32": float(d.mean())})
  diffs = np.array(diffs)
  summary = {"n_sequences": len(out), "n_response_tokens": int(diffs.size),
             "gpu_self_baseline_abs_logp_diff_bf16_vs_fp32": {"mean": float(diffs.mean()), "p50": float(np.median(diffs)),
                                                             "p95": float(np.percentile(diffs, 95)), "p99": float(np.percentile(diffs, 99)), "max": float(diffs.max())},
             "model": args.model, "model_safetensors_sha256": hashlib.sha256(open(f"{args.model}/model.safetensors", "rb").read()).hexdigest(),
             "note": "log-probs of response tokens under the initial checkpoint, plain HF forward, no temperature scaling (T=1), "
                     "EOS appended to each response; compare trainer log-probs on identical token ids/mask to logp_fp32."}
  with open(args.out + ".json", "w") as f:
    json.dump({"summary": summary, "sequences": out}, f)
  print(json.dumps(summary, indent=1))
  print(f"saved {args.out}.json")


if __name__ == "__main__":
  main()
