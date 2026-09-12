#!/usr/bin/env python3
"""Paired comparison of two validation dumps (same questions, two checkpoints).

Phase 0 decision statistic: for each eval data source, acc(step B) - acc(step A)
on the SAME questions, with a paired bootstrap 95% CI over questions and an
exact McNemar test on the discordant pairs. This is the right test for a fixed
greedy eval set (unpaired binomial std overstates the noise).

Usage:
  python3 paired_eval_bootstrap.py --dump <val_dump_dir> --a 0 --b 300 \
      --data_dir $DATA_DIR [--boot 20000] [--min_gain 0.04]
Rows are paired by question text (extracted from the dumped prompt using the
template of the parquet rows); data source is recovered by looking the question
up in gsm8k_test.parquet / val_1k_qsplit.parquet.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")


def norm(s):
  return _WS.sub(" ", s).strip()


def load_rows(path):
  return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def question_from_input(text, prefix_tail, suffix_head):
  """The dumped `input` is the prompt with special tokens stripped:
  '...<instruction>\\n\\n<question><end_of_turn>...'. Cut between the last
  occurrence of the instruction tail and the first occurrence of the suffix head."""
  i = text.rfind(prefix_tail)
  j = text.find(suffix_head, i if i >= 0 else 0)
  if i < 0 or j < 0:
    return None
  return norm(text[i + len(prefix_tail):j])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--a", type=int, default=0)
  ap.add_argument("--b", type=int, required=True)
  ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR"))
  ap.add_argument("--boot", type=int, default=20000)
  ap.add_argument("--min_gain", type=float, default=0.04)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  # data-source lookup + template pieces from the parquets
  src_of = {}
  tmpl = None
  for name, fn in (("gsm8k", "gsm8k_test.parquet"), ("omi2_val1k", "val_1k_qsplit.parquet")):
    p = os.path.join(args.data_dir, fn)
    if not os.path.exists(p):
      continue
    df = pd.read_parquet(p)
    for _, r in df.iterrows():
      q = r["extra_info"]["question"]
      src_of[norm(q)] = name
      if tmpl is None:
        c = r["prompt"][0]["content"]
        k = c.find(q)
        tmpl = (c[:k], c[k + len(q):])
  if tmpl is None:
    raise SystemExit("could not derive the prompt template from the parquets")
  prefix_tail = tmpl[0][-40:]          # the instruction's last 40 chars (survive special-token stripping)
  suffix_head = tmpl[1][:14]           # '<end_of_turn>' + ...

  rows_a = load_rows(os.path.join(args.dump, f"{args.a}.jsonl"))
  rows_b = load_rows(os.path.join(args.dump, f"{args.b}.jsonl"))

  def index(rows):
    out = {}
    miss = 0
    for r in rows:
      q = question_from_input(r["input"], prefix_tail, suffix_head)
      if q is None:
        miss += 1
        continue
      out[q] = float(r.get("acc", 1.0 if float(r.get("score", 0)) >= 1.0 else 0.0))
    return out, miss

  A, miss_a = index(rows_a)
  B, miss_b = index(rows_b)
  common = sorted(set(A) & set(B))
  print(f"rows: step {args.a}={len(rows_a)}  step {args.b}={len(rows_b)}  paired questions={len(common)}  "
        f"(unparsed: {miss_a}/{miss_b})")

  rng = np.random.default_rng(args.seed)
  print(f"\n{'source':<12}{'n':>6}{'acc_a':>8}{'acc_b':>8}{'delta':>8}{'95% CI (paired bootstrap)':>28}{'0->1':>6}{'1->0':>6}{'McNemar p':>11}  verdict")
  for src in ("gsm8k", "omi2_val1k", "all"):
    qs = [q for q in common if src == "all" or src_of.get(q) == src]
    if not qs:
      continue
    a = np.array([A[q] for q in qs]); b = np.array([B[q] for q in qs])
    d = b - a
    n = len(qs)
    delta = d.mean()
    idx = rng.integers(0, n, size=(args.boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    up = int(((a == 0) & (b == 1)).sum()); down = int(((a == 1) & (b == 0)).sum())
    # exact McNemar: discordant pairs ~ Binomial(up+down, 0.5)
    from math import comb
    m = up + down
    k = min(up, down)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m)) if m else 1.0
    if lo > 0 and delta >= args.min_gain:
      verdict = "PASS"
    elif lo > 0:
      verdict = f"improves, but < {args.min_gain:.2f}"
    elif hi < 0:
      verdict = "REGRESSION"
    else:
      verdict = "no significant change"
    print(f"{src:<12}{n:>6}{a.mean():>8.3f}{b.mean():>8.3f}{delta:>+8.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>28}{up:>6}{down:>6}{p:>11.4f}  {verdict}")


if __name__ == "__main__":
  main()