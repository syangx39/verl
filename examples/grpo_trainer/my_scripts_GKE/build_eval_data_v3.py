#!/usr/bin/env python3
"""Rebuild eval/train parquets for Track A Phase 0 (v4).

v4 changes (review): question identity is WHITESPACE-NORMALIZED everywhere
(holdout membership, leak checks, holdout list) -- the raw-string version let a
question with different spacing slip through. extra_info.index is re-assigned
to a unique global row id after concatenation (the old train/val files were
numbered independently, so indices collided). A sha256 MANIFEST is written.

Fixes two problems found in the pilot:
  1. val_1k was a random ROW split of train_1M, but OpenMathInstruct-2 has ~1.6
     rows (solutions) per question, so 77% of val questions were also in train.
     -> split at the QUESTION level: hold out N unique questions entirely.
  2. GSM8K test was never built. -> build it with the SAME prompt template as
     the training rows (template is taken from an existing row, not retyped).

Inputs  (all in $DATA_DIR):
  train.parquet, val.parquet      the existing row-split of train_1M (verl format)
  gsm8k_test.jsonl                raw GSM8K test (1319 rows, fields question/answer)
                                  -> get it on a machine with internet:
                                     python -c "from datasets import load_dataset; \\
                                       load_dataset('openai/gsm8k','main',split='test').to_json('gsm8k_test.jsonl')"
                                     then gsutil cp into the bucket's data dir.
Outputs (in $DATA_DIR):
  train_qsplit.parquet            train minus all rows of the held-out questions
  val_1k_qsplit.parquet           1000 held-out questions, one row each, data_source=omi2_val1k
  holdout_questions.json          the 2000 held-out questions (for the rulebook / TPU side)
  gsm8k_test.parquet              1319 rows, data_source=gsm8k
"""

import argparse
import hashlib
import json
import os
import re

import pandas as pd


_WS = re.compile(r"\s+")


def norm_q(text):
  """Question identity: collapse all whitespace runs, strip."""
  return _WS.sub(" ", text).strip()


def question_of(row):
  return row["extra_info"]["question"]


def qkey_of(row):
  return norm_q(row["extra_info"]["question"])


def build_prompt_template(df):
  """Extract prefix/suffix around the question from an existing row.

  Row content looks like  <prefix>{question}<suffix>; we recover both and
  assert they are identical across a sample of rows.
  """
  seen = set()
  for _, r in df.head(200).iterrows():
    q = question_of(r)
    content = r["prompt"][0]["content"]
    i = content.find(q)
    assert i >= 0, "question text not found inside prompt content"
    seen.add((content[:i], content[i + len(q):]))
  assert len(seen) == 1, f"prompt template is not uniform across rows: {len(seen)} variants"
  prefix, suffix = next(iter(seen))
  return prefix, suffix


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR"))
  ap.add_argument("--n_holdout", type=int, default=2000)
  ap.add_argument("--n_val", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  d = args.data_dir

  tr = pd.read_parquet(f"{d}/train.parquet")
  va = pd.read_parquet(f"{d}/val.parquet")
  full = pd.concat([tr, va], ignore_index=True)
  # unique global row id (old files were numbered independently -> collisions)
  full["extra_info"] = [dict(e, index=i, orig_index=e.get("index")) for i, e in enumerate(full["extra_info"])]
  full["_q"] = [qkey_of(r) for _, r in full.iterrows()]          # normalized identity
  uniq = full["_q"].drop_duplicates()
  print(f"rows={len(full)}  unique questions (normalized)={len(uniq)}")

  holdout = uniq.sample(n=args.n_holdout, random_state=args.seed)
  hold = set(holdout)
  train_q = full[~full["_q"].isin(hold)].drop(columns=["_q"]).reset_index(drop=True)
  val_rows = full[full["_q"].isin(hold)].drop_duplicates("_q")
  val_q = val_rows.sample(n=args.n_val, random_state=args.seed).drop(columns=["_q"]).reset_index(drop=True)
  val_q["data_source"] = "omi2_val1k"

  # re-verify no leakage (normalized) and index uniqueness
  tq = set(qkey_of(r) for _, r in train_q.iterrows())
  leak = sum(qkey_of(r) in tq for _, r in val_q.iterrows())
  assert leak == 0, f"leak: {leak}"
  idx = [e["index"] for e in train_q["extra_info"]]
  assert len(idx) == len(set(idx)), "extra_info.index not unique in train"

  train_q.to_parquet(f"{d}/train_qsplit.parquet", index=False)
  val_q.to_parquet(f"{d}/val_1k_qsplit.parquet", index=False)
  json.dump(sorted(hold), open(f"{d}/holdout_questions.json", "w"), ensure_ascii=False)
  print(f"train_qsplit={len(train_q)} rows   val_1k_qsplit={len(val_q)} rows   holdout={len(hold)} questions   leak={leak}")

  # ---- GSM8K test with the identical template --------------------------------
  prefix, suffix = build_prompt_template(tr)
  print("template prefix:", repr(prefix[:90]), "...  suffix:", repr(suffix))
  raw = f"{d}/gsm8k_test.jsonl"
  if not os.path.exists(raw):
    print(f"[skip] {raw} not found -- fetch GSM8K test first (see docstring)")
    return
  rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
  assert len(rows) == 1319, len(rows)
  out = []
  for i, r in enumerate(rows):
    ans = r["answer"].split("####")[-1].strip().replace(",", "")
    assert re.fullmatch(r"-?\d+(\.\d+)?", ans), f"unexpected gold: {ans!r}"
    out.append({
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": prefix + r["question"] + suffix}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps([ans, ans])},
        "extra_info": {"question": r["question"], "index": i,
                       "eval_id": hashlib.md5(r["question"].encode()).hexdigest()[:12]},
    })
  g = pd.DataFrame(out)
  gleak = sum(norm_q(r["question"]) in tq for r in rows)
  print(f"gsm8k test questions also in train (normalized): {gleak}")
  assert gleak == 0, f"GSM8K test leaks into train: {gleak}"
  g.to_parquet(f"{d}/gsm8k_test.parquet", index=False)
  print(f"gsm8k_test.parquet: {len(g)} rows; example gold={out[0]['reward_model']['ground_truth']}")

  # ---- manifest ------------------------------------------------------------
  names = ["train_qsplit.parquet", "val_1k_qsplit.parquet", "gsm8k_test.parquet", "holdout_questions.json"]
  with open(f"{d}/MANIFEST.sha256", "w") as f:
    for nme in names:
      h = hashlib.sha256(open(f"{d}/{nme}", "rb").read()).hexdigest()
      f.write(f"{h}  {nme}\n")
      print(f"  {h[:16]}...  {nme}")


if __name__ == "__main__":
  main()