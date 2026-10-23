#!/usr/bin/env python3
"""GRPO signal density per step window, from training rollout dumps (PR #10 item 4).

For every prompt group (8 completions of one prompt, grouped by `uid` if present, else `qid`) the group is classed as
  - correctness-mixed : the 8 `acc` values differ (the only groups whose advantage carries an answer signal);
  - all-correct / all-wrong : identical `acc` and identical `score` (zero advantage);
  - other-only : identical `acc` but different `score` (format credit and/or length penalty only).
Reports, per window, the mean training accuracy (sampled, rollout temperature) and the share of groups in each class.
Works on verl `trainer.rollout_data_dir` dumps (`<step>.jsonl` with acc / score per row), e.g. the Gemma-4 and the
Qwen3-0.6B runs.

Usage:
    python3 gemma4_signal_density.py --run gemma_fp32_s1=<dump_dir> [--run qwen_stab_kl0_seed1=<dump_dir> ...] \\
        [--windows 1-20,21-40,41-60,61-80,81-100] [--out-json x.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

CLASSES = ("mixed", "all_correct", "all_wrong", "other_only")


def step_stats(path: str) -> dict:
    groups = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[r.get("uid") or r.get("qid")].append(r)
    c = dict.fromkeys(CLASSES, 0)
    accs = []
    for g in groups.values():
        a = [float(r["acc"]) for r in g]
        s = [float(r["score"]) for r in g]
        accs += a
        if len(set(a)) > 1:
            c["mixed"] += 1
        elif len(set(s)) > 1:
            c["other_only"] += 1
        elif a[0] >= 0.5:
            c["all_correct"] += 1
        else:
            c["all_wrong"] += 1
    c["groups"] = len(groups)
    c["acc"] = 100 * sum(accs) / len(accs)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=dir with <step>.jsonl training dumps")
    ap.add_argument("--windows", default="1-20,21-40,41-60,61-80,81-100")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    wins = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")]
    out = {}
    for item in args.run:
        label, d = item.split("=", 1)
        steps = sorted(int(f[:-6]) for f in os.listdir(d) if f.endswith(".jsonl") and f[:-6].isdigit() and int(f[:-6]) > 0)
        out[label] = {}
        for lo, hi in wins:
            sel = [s for s in steps if lo <= s <= hi]
            if len(sel) < hi - lo + 1:  # only complete windows
                continue
            st = [step_stats(os.path.join(d, f"{s}.jsonl")) for s in sel]
            tot = sum(x["groups"] for x in st)
            w = {"steps": [lo, hi], "groups": tot, "train_acc": statistics.mean(x["acc"] for x in st)}
            w.update({k: 100 * sum(x[k] for x in st) / tot for k in CLASSES})
            out[label][f"{lo}-{hi}"] = w
            print(f"{label:22s} {lo:3d}-{hi:3d}: train acc {w['train_acc']:.1f}% | correctness-mixed {w['mixed']:.1f}% | "
                  f"all-correct {w['all_correct']:.1f}% | all-wrong {w['all_wrong']:.1f}% | "
                  f"format/penalty-only {w['other_only']:.1f}% | groups {tot}", flush=True)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
