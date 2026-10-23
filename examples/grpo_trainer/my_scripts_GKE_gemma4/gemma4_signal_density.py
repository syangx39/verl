#!/usr/bin/env python3
"""GRPO signal density per step window, from training rollout dumps (PR #10 item 4; PR #12 A6 shapes).

For every prompt group (the n completions of one prompt, grouped by `uid` if present, else `qid`) the group is
classed as
  - correctness-mixed : the `acc` values differ (the only groups whose advantage carries an answer signal);
  - all-correct / all-wrong : identical `acc` and identical `score` (zero advantage);
  - other-only : identical `acc` but different `score` (format credit and/or length penalty only).
Reports, per window, the mean training accuracy (sampled, rollout temperature) and the share of groups in each class.

A larger group is more likely to be correctness-mixed for the same policy and prompts. So every window also reports
`mixed_n8eq`: the expected correctness-mixed share had only 8 of each group's completions been drawn at random without
replacement, 1 - [C(c, 8) + C(n - c, 8)] / C(n, 8) for a group with c correct of n (equal to `mixed` when n = 8). Use it
to compare an n=16 run (A6) with n=8 runs; the observed n=16 share alone partly rises mechanically with n.

--shape label=PROMPTSxN validates every step file of that run before counting: exactly PROMPTS groups of exactly N rows,
one input and one ground truth per group (e.g. 256x8 for A3a, 256x16 for A6). Runs without --shape are read as they are
(e.g. Qwen dumps with other batch shapes).

Works on verl `trainer.rollout_data_dir` dumps (`<step>.jsonl` with acc / score per row), e.g. the Gemma-4 and the
Qwen3-0.6B runs.

Usage:
    python3 gemma4_signal_density.py --run gemma_fp32_s1=<dump_dir> [--run qwen_stab_kl0_seed1=<dump_dir> ...] \\
        [--shape gemma_fp32_s1=256x8] [--windows 1-20,21-40,41-60,61-80,81-100] [--out-json x.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict

CLASSES = ("mixed", "all_correct", "all_wrong", "other_only")
SUB = 8  # reference completions per prompt for mixed_n8eq


def mixed_prob_at(c: int, n: int, k: int = SUB) -> float:
    """P(a random k-subset of a group with c correct of n is correctness-mixed); the observed indicator if n <= k."""
    if n <= k:
        return float(0 < c < n)
    return 1.0 - (math.comb(c, k) + math.comb(n - c, k)) / math.comb(n, k)


def step_stats(path: str, shape=None) -> dict:
    groups = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[r.get("uid") or r.get("qid")].append(r)
    if shape:
        prompts, n = shape
        if len(groups) != prompts:
            raise SystemExit(f"{path}: {len(groups)} prompt groups, expected {prompts}")
        for q, g in groups.items():
            if len(g) != n:
                raise SystemExit(f"{path}: group {q} has {len(g)} rows, expected {n}")
            if len({str(r.get("input")) for r in g}) != 1 or len({str(r.get("gts")) for r in g}) != 1:
                raise SystemExit(f"{path}: group {q} mixes inputs or ground truths")
    c = dict.fromkeys(CLASSES, 0)
    accs = []
    n8 = 0.0
    sizes = set()
    for g in groups.values():
        a = [float(r["acc"]) for r in g]
        s = [float(r["score"]) for r in g]
        accs += a
        sizes.add(len(g))
        n8 += mixed_prob_at(sum(x >= 0.5 for x in a), len(a))
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
    c["mixed_n8eq"] = n8
    c["group_sizes"] = sorted(sizes)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=dir with <step>.jsonl training dumps")
    ap.add_argument("--shape", action="append", default=[], help="label=PROMPTSxN: validate that run's batch shape")
    ap.add_argument("--windows", default="1-20,21-40,41-60,61-80,81-100")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()
    wins = [tuple(int(x) for x in w.split("-")) for w in args.windows.split(",")]
    shapes = {}
    for item in args.shape:
        label, spec = item.split("=", 1)
        p, n = spec.lower().split("x")
        shapes[label] = (int(p), int(n))
    out = {}
    for item in args.run:
        label, d = item.split("=", 1)
        steps = sorted(
            int(f[:-6]) for f in os.listdir(d) if f.endswith(".jsonl") and f[:-6].isdigit() and int(f[:-6]) > 0
        )
        out[label] = {}
        for lo, hi in wins:
            sel = [s for s in steps if lo <= s <= hi]
            if len(sel) < hi - lo + 1:  # only complete windows
                continue
            st = [step_stats(os.path.join(d, f"{s}.jsonl"), shapes.get(label)) for s in sel]
            tot = sum(x["groups"] for x in st)
            w = {
                "steps": [lo, hi],
                "groups": tot,
                "train_acc": statistics.mean(x["acc"] for x in st),
                "group_sizes": sorted({n for x in st for n in x["group_sizes"]}),
                "shape_validated": label in shapes,
            }
            w.update({k: 100 * sum(x[k] for x in st) / tot for k in (*CLASSES, "mixed_n8eq")})
            out[label][f"{lo}-{hi}"] = w
            n8 = f" (n=8-equivalent {w['mixed_n8eq']:.1f}%)" if w["group_sizes"] != [SUB] else ""
            print(
                f"{label:22s} {lo:3d}-{hi:3d}: train acc {w['train_acc']:.1f}% | "
                f"correctness-mixed {w['mixed']:.1f}%{n8} | "
                f"all-correct {w['all_correct']:.1f}% | all-wrong {w['all_wrong']:.1f}% | "
                f"format/penalty-only {w['other_only']:.1f}% | groups {tot}",
                flush=True,
            )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
