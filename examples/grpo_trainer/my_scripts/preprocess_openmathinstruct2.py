#!/usr/bin/env python3
"""Preprocess nvidia/OpenMathInstruct-2 (train_1M shards) into verl parquet,
reproducing the MaxText RL data pipeline token-for-token.

Faithful to maxtext/trainers/post_train/rl/{train_rl.py, utils_rl.py} with the
TPU run's config (rl.yml defaults + JobSet overrides):

  MaxText step                                  | Reproduced here
  ----------------------------------------------|----------------------------
  prepare_train_and_eval_dataset:               |
    load train_1M-*.parquet                     | --input glob
    train_test_split(test_size=0.05, seed=S)    | HF datasets same call, same S
  process_data:                                 |
    question = x["problem"]                     | same
    answer   = x["expected_answer"]             | same
    processed_answer = [answer, answer]         | same (dedup happens in reward,
                                                |  json list kept identical)
    content = TEMPLATE.format(                  | same strings, hardcoded below
        system_prompt=SYSTEM_PROMPT.format(...),| from gsm8k_rl.json + rl.yml
        question=question)                      | token defaults
    prompts = tokenizer.apply_chat_template(    | verl applies the SAME
        [{"role":"user","content":content}],    | tokenizer chat template at
        tokenize=False,                         | load time (we store messages,
        add_generation_prompt=True)             | not the rendered string) —
                                                | identical rendering path.
  _filter_long_prompts:                         |
    len(tokenize(prompts)) <= 8192              | same filter, same tokenizer,
                                                | applied to the RENDERED prompt
Output schema (verl RLHFDataset):
  prompt        : [{"role":"user","content": <content>}]
  data_source   : "openmathinstruct2_maxtext"   (routing label only; reward is
                                                 forced via custom_reward_function)
  reward_model  : {"style":"rule", "ground_truth": json.dumps([ans, ans])}
                  # ground_truth carries the MaxText `answer` field verbatim
  extra_info    : {"question": <problem>, "index": i}

Known deltas (documented, not reproducible):
  * MaxText shuffles with grain using data_shuffle_seed; verl shuffles with its
    own sampler. Same sample SET (guaranteed by identical split+filter),
    different ORDER. Curve overlay tolerance must allow for this.

Usage:
  python3 preprocess_openmathinstruct2.py \
      --input  ~/meta-RL/data/openmathinstruct2_raw/data/train_1M-*.parquet \
      --tokenizer ~/meta-RL/models/Qwen3-0.6B \
      --output-dir ~/meta-RL/data/openmathinstruct2 \
      --seed 0
"""

import argparse
import glob
import json
import os

import datasets
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Frozen strings — copied verbatim from the MaxText side. DO NOT EDIT.
# Sources:
#   src/maxtext/examples/chat_templates/gsm8k_rl.json  (TEMPLATE, SYSTEM_PROMPT)
#   src/maxtext/configs/post_train/rl.yml L214-217      (special tokens)
# ---------------------------------------------------------------------------
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

SYSTEM_PROMPT_RAW = (
    "You are given a problem. Think about the problem and provide your reasoning. "
    "Place it between {reasoning_start_token} and {reasoning_end_token}. Then, "
    "provide the final answer (i.e., just one numerical value) between "
    "{solution_start_token} and {solution_end_token}."
)

# NOTE: Gemma-style turn markers INSIDE the user content. This mirrors
# MaxText's gsm8k_rl.json exactly (historical carry-over from a Gemma recipe);
# the Qwen chat template is applied ON TOP of this by apply_chat_template.
# Weird but frozen — the TPU model sees exactly this. Do not "fix".
TEMPLATE = "<start_of_turn>user\n{system_prompt}\n\n{question}<end_of_turn>\n<start_of_turn>model"

MAX_PROMPT_TOKENS = 8192  # rl.yml max_prefill_predict_length via JobSet
TEST_SIZE = 0.05          # prepare_train_and_eval_dataset default
DATA_SOURCE = "openmathinstruct2_maxtext"


def build_content(question: str) -> str:
  system_prompt = SYSTEM_PROMPT_RAW.format(
      reasoning_start_token=REASONING_START,
      reasoning_end_token=REASONING_END,
      solution_start_token=SOLUTION_START,
      solution_end_token=SOLUTION_END,
  )
  return TEMPLATE.format(system_prompt=system_prompt, question=question)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True, help="glob for train_1M-*.parquet")
  ap.add_argument("--tokenizer", required=True, help="path or HF id of Qwen3-0.6B")
  ap.add_argument("--output-dir", required=True)
  ap.add_argument("--seed", type=int, default=42,
                  help="MaxText data_shuffle_seed (rl.yml: data_shuffle_seed=42, confirmed)")
  args = ap.parse_args()

  files = sorted(glob.glob(os.path.expanduser(args.input)))
  assert files, f"no parquet matched {args.input}"
  print(f"[1/5] loading {len(files)} parquet shard(s)")
  ds = datasets.load_dataset("parquet", data_files={"train": files}, split="train")
  print(f"      rows: {len(ds)}  columns: {ds.column_names}")
  assert "problem" in ds.column_names and "expected_answer" in ds.column_names, (
      "schema mismatch: expected OpenMathInstruct-2 columns 'problem'/'expected_answer', "
      f"got {ds.column_names}")

  # --- split: same call, same test_size, same seed as MaxText -------------
  print(f"[2/5] train_test_split(test_size={TEST_SIZE}, seed={args.seed})")
  split = ds.train_test_split(test_size=TEST_SIZE, seed=args.seed)

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

  def to_verl_row(x, idx):
    question = str(x["problem"])
    answer = str(x["expected_answer"])
    content = build_content(question)
    return {
        "prompt": [{"role": "user", "content": content}],
        "data_source": DATA_SOURCE,
        "reward_model": {
            "style": "rule",
            # MaxText: process_answer(default) -> [answer, answer], json-encoded
            "ground_truth": json.dumps([answer, answer]),
        },
        "extra_info": {"question": question, "index": idx},
    }

  def keep(x):
    # MaxText _filter_long_prompts: tokenize the RENDERED prompt (after
    # apply_chat_template) and keep <= 8192. Rendering path identical to
    # MaxText process_data.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_content(str(x["problem"]))}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.tokenize(rendered)) <= MAX_PROMPT_TOKENS

  out = {}
  for name, part in (("train", split["train"]), ("val", split["test"])):
    print(f"[3/5] ({name}) filter prompts > {MAX_PROMPT_TOKENS} tokens (slow: full tokenize)")
    part = part.filter(keep, num_proc=os.cpu_count())
    print(f"      kept {len(part)} rows")
    print(f"[4/5] ({name}) map to verl schema")
    part = part.map(to_verl_row, with_indices=True,
                    remove_columns=part.column_names, num_proc=os.cpu_count())
    out[name] = part

  os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
  for name, part in out.items():
    path = os.path.join(os.path.expanduser(args.output_dir), f"{name}.parquet")
    print(f"[5/5] writing {path}")
    part.to_parquet(path)

  # --- sanity print: one fully rendered prompt for eyeball diff vs TPU ----
  sample = out["train"][0]
  rendered = tokenizer.apply_chat_template(
      sample["prompt"], tokenize=False, add_generation_prompt=True)
  print("=" * 70)
  print("SAMPLE RENDERED PROMPT (diff this against TPU-side dump):")
  print(rendered)
  print("=" * 70)


if __name__ == "__main__":
  main()
