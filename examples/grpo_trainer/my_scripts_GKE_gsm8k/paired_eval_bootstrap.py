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


def question_from_input(text, prefix_tail, suffix_head, lookup=None):
  """The dumped `input` is the rendered prompt with special tokens stripped. Recover the question:
  (1) if prefix/suffix markers exist, cut between them; (2) otherwise (question is the whole user
  turn, e.g. Meta's system+user format) match by containment against the known questions."""
  if prefix_tail or suffix_head:
    i = text.rfind(prefix_tail) if prefix_tail else 0
    j = text.find(suffix_head, i if i >= 0 else 0) if suffix_head else len(text)
    if i >= 0 and j >= 0:
      q = norm(text[i + len(prefix_tail):j])
      if lookup is None or q in lookup:
        return q
  if lookup is not None:
    t = norm(text)
    hits = [q for q in lookup if q in t]
    if hits:
      return max(hits, key=len)            # longest containment match
  return None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True)
  ap.add_argument("--a", type=int, default=0)
  ap.add_argument("--b", type=int, required=True)
  ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR"))
  ap.add_argument("--sources", nargs="*", default=None,
                  help="eval sources as name=parquet_path; default: gsm8k=<data_dir>/gsm8k_test.parquet omi2_val1k=<data_dir>/val_1k_qsplit.parquet")
  ap.add_argument("--boot", type=int, default=20000)
  ap.add_argument("--min_gain", type=float, default=0.04)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--key", choices=["question", "qid"], default="question",
                  help="how dump rows are paired with eval questions. 'question': recover the question text from the dumped "
                       "prompt (Track A dumps). 'qid': use the reward's qid (= extra_info.index), required when several eval "
                       "sets share questions (e.g. Meta test512 is a subset of the full test); index ranges must not overlap")
  args = ap.parse_args()

  # data-source lookup + template pieces from the parquets
  src_of = {}
  tmpl = None
  if args.sources:
    sources = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in args.sources]
  else:
    sources = [("gsm8k", os.path.join(args.data_dir, "gsm8k_test.parquet")), ("omi2_val1k", os.path.join(args.data_dir, "val_1k_qsplit.parquet"))]
  source_names = [n for n, _ in sources]
  qid_of = {}                 # qid -> (source, normalized question)
  conflicts = set()
  qid_conflicts = set()
  for name, p in sources:
    if not os.path.exists(p):
      continue
    df = pd.read_parquet(p)
    for _, r in df.iterrows():
      q = r["extra_info"]["question"]
      nq = norm(q)
      if nq in src_of and src_of[nq] != name:
        conflicts.add(nq)
      src_of[nq] = name
      qi = int(r["extra_info"]["index"])
      if qi in qid_of and qid_of[qi][0] != name:
        qid_conflicts.add(qi)
      qid_of[qi] = (name, nq)
      if tmpl is None:
        c = r["prompt"][-1]["content"]           # the user turn (last message) carries the question
        k = c.find(q)
        tmpl = (c[:k], c[k + len(q):])
  if tmpl is None:
    raise SystemExit("could not derive the prompt template from the parquets")
  if qid_conflicts and args.key == "qid":
    raise SystemExit(f"{len(qid_conflicts)} qids are shared between eval sets (e.g. {sorted(qid_conflicts)[:3]}): "
                     f"--key qid needs disjoint extra_info.index ranges (rebuild the eval parquets)")
  if conflicts and args.key == "question":
    raise SystemExit(f"{len(conflicts)} questions appear in more than one eval set (e.g. test512 within the full test); "
                     f"pairing by question text would mix them -- rerun with --key qid")
  prefix_tail = tmpl[0][-40:]          # last chars before the question in the rendered user turn ("" if the question is the whole turn)
  suffix_head = tmpl[1][:14]           # chars right after the question ("" if none)

  rows_a = load_rows(os.path.join(args.dump, f"{args.a}.jsonl"))
  rows_b = load_rows(os.path.join(args.dump, f"{args.b}.jsonl"))

  def index(rows):
    """Map pairing key -> acc. Key = (source, question) via qid, or the question text (legacy)."""
    out = {}
    miss = 0
    dup = 0
    for r in rows:
      if args.key == "qid":
        qi = r.get("qid")
        if qi is None or int(qi) not in qid_of:
          miss += 1
          continue
        key = qid_of[int(qi)]                       # (source, question)
      else:
        q = question_from_input(r["input"], prefix_tail, suffix_head, src_of)
        if q is None:
          miss += 1
          continue
        key = (src_of.get(q), q)
      if key in out:
        dup += 1
      out[key] = float(r.get("acc", 1.0 if float(r.get("score", 0)) >= 1.0 else 0.0))
    if dup:
      raise SystemExit(f"{dup} duplicate pairing keys inside one dump -- overlapping eval sets? use --key qid with disjoint index ranges")
    return out, miss

  A, miss_a = index(rows_a)
  B, miss_b = index(rows_b)
  common = sorted(set(A) & set(B))
  print(f"rows: step {args.a}={len(rows_a)}  step {args.b}={len(rows_b)}  paired questions={len(common)}  "
        f"(unparsed: {miss_a}/{miss_b})")

  rng = np.random.default_rng(args.seed)
  print(f"\n{'source':<12}{'n':>6}{'acc_a':>8}{'acc_b':>8}{'delta':>8}{'95% CI (paired bootstrap)':>28}{'0->1':>6}{'1->0':>6}{'McNemar p':>11}  verdict")
  for src in source_names + ["all"]:
    qs = [k for k in common if src == "all" or k[0] == src]
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
