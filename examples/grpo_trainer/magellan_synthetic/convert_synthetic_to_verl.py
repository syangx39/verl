#!/usr/bin/env python3
"""Convert synthetic ranking JSONL (generate_synthetic_data.py output) into
verl RLHFDataset parquet.

Input row schema (from the synthetic generator):
  {"prompt": [{"role":"system","content":...},{"role":"user","content":...}],
   "extra_info": {"label_arr": [0,1,...]} or {"fm_labels": [0.42,...]}}

Output parquet columns (verl):
  prompt        : messages list, passed through UNCHANGED (verl applies the
                  tokenizer chat template at load time)
  data_source   : "magellan_synthetic_16k"
  reward_model  : {"style":"rule", "ground_truth": ""}  (dummy reward ignores it)
  extra_info    : original extra_info + {"index": i}    (dummy reward reads
                  label_arr/fm_labels from here for the item count)

Split: --val-rows rows are held out for validation (taken from the tail so the
train prefix is stable if the dataset is regenerated with more rows).

Usage:
  python3 convert_synthetic_to_verl.py \
      --input ~/meta-RL/data/magellan_synthetic/ads_synthetic_16k_typical.jsonl \
      --output-dir ~/meta-RL/data/magellan_synthetic_16k \
      --val-rows 512
"""

import argparse
import json
import os

import datasets

DATA_SOURCE = "magellan_synthetic_16k"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True)
  ap.add_argument("--tokenizer", default="~/meta-RL/models/Qwen3-0.6B",
                  help="model dir; used to render+count prompt tokens for the 16384 filter")
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--val-rows", type=int, default=512)
  args = ap.parse_args()

  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.tokenizer))

  # Sanity: a known sentence must tokenize to a plausible count. Guards
  # against silent degenerate behavior (observed: without PyTorch installed,
  # apply_chat_template(tokenize=True) returned length-2 results for every
  # prompt, wrecking both the filter and the recorded input_tokens).
  # Render-to-string then count. NOTE: apply_chat_template(tokenize=True)
  # returns a BatchEncoding dict in newer transformers (len()==2 keys!) —
  # never use len() on it. String render + tokenize() is version-stable.
  def count_prompt_tokens(messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer.tokenize(rendered))

  probe = count_prompt_tokens([{"role": "user", "content": "hello " * 100}])
  assert probe > 80, (
      f"tokenizer sanity check failed (probe={probe} tokens for a "
      "100-word message); transformers API/environment problem")

  rows = []
  dropped = 0
  with open(os.path.expanduser(args.input)) as f:
    for i, line in enumerate(f):
      r = json.loads(line)
      extra = dict(r.get("extra_info") or {})
      # Canonical dataset rule: prompts must fit max_prompt_length=16384,
      # measured by REAL tokenization of the rendered prompt (this jsonl has
      # no input_tokens field; and real token count is the binding quantity
      # anyway — verl renders with this same tokenizer/chat template). The
      # SAME filter must be applied on the TPU/MaxText side.
      n_tokens = count_prompt_tokens(r["prompt"])
      if n_tokens > 16384:
        dropped += 1
        continue
      extra["index"] = i
      extra["input_tokens"] = n_tokens  # record for downstream stats
      rows.append({
          "prompt": r["prompt"],
          "data_source": DATA_SOURCE,
          "reward_model": {"style": "rule", "ground_truth": ""},
          "extra_info": extra,
      })
  print(f"dropped {dropped} rows with rendered prompt > 16384 tokens")

  assert len(rows) > args.val_rows, "not enough rows for the requested val split"
  train, val = rows[: -args.val_rows], rows[-args.val_rows:]
  print(f"rows: {len(rows)}  -> train {len(train)}, val {len(val)}")

  os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
  for name, part in (("train", train), ("val", val)):
    ds = datasets.Dataset.from_list(part)
    path = os.path.join(os.path.expanduser(args.output_dir), f"{name}.parquet")
    ds.to_parquet(path)
    print(f"wrote {path}")

  # sanity: one rendered-ish peek (raw messages; verl renders with the model's
  # chat template at load time)
  print("=" * 60)
  print("SAMPLE ROW (train[0]) message roles/lengths:")
  for m in train[0]["prompt"]:
    print(f"  {m['role']}: {len(m['content'])} chars")
  print("  extra_info keys:", list(train[0]["extra_info"].keys()))


if __name__ == "__main__":
  main()
