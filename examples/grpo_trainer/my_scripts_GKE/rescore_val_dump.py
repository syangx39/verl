#!/usr/bin/env python3
"""Re-score verl validation dumps (<step>.jsonl) with the CURRENT reward file.

Why: Phase 0 seed 1 ran with math_verify's signal-based timeout, which raises in
verl's reward threads, so every symbolic-equivalence match scored 0. The dumps
keep input/output/gts, so the corrected eval curve can be recomputed offline.

Usage:
  python3 rescore_val_dump.py --dump /workspace/meta-RL/logs/<EXP>/val_dump \
      --reward /workspace/.../maxtext_math_reward.py
Prints, per step: original acc (from the dump) vs re-scored acc, and how many
answers flipped 0 -> 1 (equivalence rescued) or 1 -> 0 (should be none).
"""

import argparse
import glob
import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def load_reward(path):
  spec = importlib.util.spec_from_file_location("reward_mod", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dump", required=True, help="val_dump or rollout_dump directory")
  ap.add_argument("--reward", required=True, help="path to maxtext_math_reward.py (fixed version)")
  ap.add_argument("--max_rows", type=int, default=0, help="cap rows per file (0 = all)")
  args = ap.parse_args()

  os.environ.pop("REWARD_DUMP_DIR", None)   # no sample dump while re-scoring
  r = load_reward(args.reward)

  files = []
  for fn in glob.glob(os.path.join(args.dump, "*.jsonl")):
    try:
      files.append((int(os.path.splitext(os.path.basename(fn))[0]), fn))
    except ValueError:
      pass
  files.sort()
  if not files:
    sys.exit(f"no <step>.jsonl under {args.dump}")

  print(f"{'step':>6} {'n':>6} {'acc_orig':>9} {'acc_fixed':>10} {'delta':>7} {'0->1':>6} {'1->0':>6} {'fmt':>6}")
  for step, fn in files:
    n = acc_o = acc_f = up = down = fmt = 0
    with open(fn, encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        row = json.loads(line)
        if args.max_rows and n >= args.max_rows:
          break
        gts = row.get("gts")
        if not isinstance(gts, str):
          gts = json.dumps(gts)
        new = r.compute_score("rescore", row["output"], gts)
        a_new = float(new["acc"])
        a_old = float(row.get("acc", 1.0 if float(row.get("score", 0)) >= 1.0 else 0.0))
        n += 1
        acc_o += a_old
        acc_f += a_new
        fmt += float(new["fmt"])
        if a_new > a_old:
          up += 1
        elif a_new < a_old:
          down += 1
    if n:
      print(f"{step:>6} {n:>6} {acc_o / n:>9.3f} {acc_f / n:>10.3f} {acc_f / n - acc_o / n:>+7.3f} {up:>6} {down:>6} {fmt / n:>6.3f}")


if __name__ == "__main__":
  main()