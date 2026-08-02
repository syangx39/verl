#!/usr/bin/env python3
"""NuminaMath-CoT -> verl parquet, mirroring Meta's preprocess_numina.py.

Reproduces their prompt construction EXACTLY (same SYSTEM_PROMPT string, same
[system, user] structure, user turn = raw `problem` field), then wraps into
verl RLHFDataset columns. `solution` is carried into reward_model.ground_truth
for a future correctness reward; the current benchmark reward (per their CSV,
mean 0.951) appears format-only — plug in their reward fn when provided.

Usage:
  python3 convert_numina_to_verl.py \
      --output-dir ~/meta-RL/data/numina_verl \
      --max-rows 120000 --val-rows 512
"""

import argparse
import os

import datasets

# Verbatim from Meta's preprocess_numina.py — DO NOT EDIT.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
)

DATA_SOURCE = "numina_math_cot"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", default="AI-MO/NuminaMath-CoT")
  ap.add_argument("--split", default="train")
  ap.add_argument("--prompt-field", default="problem")
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--max-rows", type=int, default=0,
                  help="cap rows for speed (0 = all ~860K); benchmark needs "
                       "steps x 128 prompts only")
  ap.add_argument("--val-rows", type=int, default=512)
  args = ap.parse_args()

  ds = datasets.load_dataset(args.dataset, split=args.split)
  print(f"loaded {len(ds)} rows; columns: {ds.column_names}")
  if args.max_rows and len(ds) > args.max_rows:
    ds = ds.select(range(args.max_rows))
    print(f"capped to {len(ds)} rows")

  def to_verl(row, idx):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row[args.prompt_field]},
        ],
        "data_source": DATA_SOURCE,
        "reward_model": {"style": "rule",
                         "ground_truth": row.get("solution") or ""},
        "extra_info": {"index": idx},
    }

  ds = ds.map(to_verl, with_indices=True, remove_columns=ds.column_names,
              num_proc=os.cpu_count())

  assert len(ds) > args.val_rows
  train = ds.select(range(len(ds) - args.val_rows))
  val = ds.select(range(len(ds) - args.val_rows, len(ds)))

  out = os.path.expanduser(args.output_dir)
  os.makedirs(out, exist_ok=True)
  train.to_parquet(os.path.join(out, "train.parquet"))
  val.to_parquet(os.path.join(out, "val.parquet"))
  print(f"train {len(train)}, val {len(val)} -> {out}")

  ex = train[0]
  print("=" * 60)
  print("system:", ex["prompt"][0]["content"][:80], "...")
  print("user  :", ex["prompt"][1]["content"][:200])
  print("ground_truth:", str(ex["reward_model"]["ground_truth"])[:120])


if __name__ == "__main__":
  main()
