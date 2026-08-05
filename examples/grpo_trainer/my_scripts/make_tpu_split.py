#!/usr/bin/env python3
"""Materialize the OpenMathInstruct-2 train/val split for the TPU team.

Produces RAW-schema parquet (original columns untouched: problem,
expected_answer, ...) — loadable by MaxText's HF pipeline directly — but
pre-split with EXACTLY the same rule as our verl preprocessing
(preprocess_openmathinstruct2.py): shards concatenated in filename order,
datasets.train_test_split(test_size=0.05, seed=42).

Handing both sides pre-split files removes split-implementation semantics
from the alignment surface entirely: MaxText points hf_train_files at
train_raw.parquet, verl preprocessing consumes the same files, sample sets
are identical by construction.

Usage:
  python3 make_tpu_split.py \
      --input '~/meta-RL/data/openmathinstruct2_raw/data/train_1M-*.parquet' \
      --output-dir ~/meta-RL/data/openmathinstruct2_tpu_split

Then verify the fingerprint against our verl train.parquet (printed at the
end) and upload:
  gsutil -m cp ~/meta-RL/data/openmathinstruct2_tpu_split/*.parquet gs://<shared-bucket>/openmathinstruct2_split/
"""

import argparse
import glob
import os

import datasets


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True)
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--val-frac", type=float, default=0.05)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--verl-train", default="~/meta-RL/data/openmathinstruct2/train.parquet",
                  help="our verl-side train.parquet, for the cross-check fingerprint")
  args = ap.parse_args()

  shards = sorted(glob.glob(os.path.expanduser(args.input)))
  assert shards, f"no shards match {args.input}"
  print(f"loading {len(shards)} shards (filename order):")
  for s in shards:
    print("  ", os.path.basename(s))
  ds = datasets.load_dataset("parquet", data_files=shards, split="train")
  print(f"total rows: {len(ds)}; columns: {ds.column_names}")

  # EXACTLY the split rule used by preprocess_openmathinstruct2.py
  split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
  train, val = split["train"], split["test"]
  print(f"split: train {len(train)}, val {len(val)} (seed={args.seed})")

  out = os.path.expanduser(args.output_dir)
  os.makedirs(out, exist_ok=True)
  train.to_parquet(os.path.join(out, "train_raw.parquet"))
  val.to_parquet(os.path.join(out, "val_raw.parquet"))
  print(f"wrote {out}/train_raw.parquet, val_raw.parquet (raw schema)")

  # ---- fingerprint --------------------------------------------------------
  print("=" * 60)
  print("FINGERPRINT (send to the TPU team alongside the files):")
  print(f"  rows: train={len(train)} val={len(val)}")
  for i in range(3):
    print(f"  train[{i}].problem[:60]: {train[i]['problem'][:60]!r}")
  for i in range(2):
    print(f"  val[{i}].problem[:60]:   {val[i]['problem'][:60]!r}")

  # ---- cross-check vs our verl train.parquet ------------------------------
  verl_path = os.path.expanduser(args.verl_train)
  if os.path.exists(verl_path):
    v = datasets.Dataset.from_parquet(verl_path)
    ok = len(v) == len(train)
    # verl prompt = [system, user]; user content embeds the problem text.
    contains = all(train[i]["problem"][:40] in v[i]["prompt"][-1]["content"]
                   for i in range(5))
    print("CROSS-CHECK vs verl train.parquet:",
          f"row-count {'OK' if ok else f'MISMATCH ({len(v)} vs {len(train)})'};",
          f"first-5 sample identity {'OK' if contains else 'MISMATCH'}")
  else:
    print(f"(verl train.parquet not found at {verl_path}; cross-check skipped)")


if __name__ == "__main__":
  main()
