#!/usr/bin/env python3
"""Build the benchmark prompt file: the first N rows of an OpenMathInstruct-2 parquet, in file order (no shuffle).
Step k of the benchmark uses rows 256k..256k+255. Prints the mean prompt length of step 1 (the TPU reference: 148.8 tokens)."""
import argparse, pandas as pd
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="OMI2 parquet in verl format (prompt = chat messages, reward_model.ground_truth)")
ap.add_argument("--out", required=True); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=5120); ap.add_argument("--step_prompts", type=int, default=256)
a = ap.parse_args()
df = pd.read_parquet(a.src)
assert len(df) >= a.n, f"source has only {len(df)} rows"
df = df.iloc[:a.n].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained(a.model)
def n_tokens(msgs):   # transformers 5: tokenize=True returns a BatchEncoding, so render text first and tokenize that
    text = tok.apply_chat_template([dict(x) for x in msgs], tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])
lens = [n_tokens(m) for m in df["prompt"].iloc[:a.step_prompts]]
print(f"rows {len(df)} | columns {list(df.columns)} | data_source {df['data_source'].iloc[0]} | step-1 mean prompt tokens {sum(lens)/len(lens):.1f} (max {max(lens)})")
df.to_parquet(a.out, index=False); print("wrote", a.out)
