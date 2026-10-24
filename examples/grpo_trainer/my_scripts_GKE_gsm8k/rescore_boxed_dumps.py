#!/usr/bin/env python3
"""Re-score verl GSM8K generation dumps under the boxed scorer v1 (Meta boxed_math) and v2 (boxed_math_reward_v2.py).

The stored outputs are re-scored as they are; nothing is regenerated. The result is therefore a scoring difference
on identical outputs, not a learning effect.

    python3 rescore_boxed_dumps.py --val qwen_seed1=<run>/val_dump --val ... [--train gemma_smoke=<run>/train] \\
        --group qwen=qwen_seed1,qwen_seed2,qwen_seed3 --json out.json --md out.md

--val DIR    <step>.jsonl eval dumps (fields output, gts, acc): per checkpoint n, stored acc, recomputed v1 acc
             (must equal the stored acc when the run was scored with v1), v2 acc, the change, the answers v2 credits
             that v1 did not, grouped by the whitelist steps used, and examples
--train DIR  <step>.jsonl rollout dumps: per step v1 / v2 accuracy and the prompt groups (same qid) whose completions
             all agree (all correct / all wrong) or are mixed, the only groups with a non-zero GRPO advantage
--group      mean over several --val labels per checkpoint (e.g. three seeds)
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_v2(path):
  spec = importlib.util.spec_from_file_location("boxed_math_reward_v2", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def steps_of(d):
  files = glob.glob(os.path.join(d, "*.jsonl"))
  return sorted((int(os.path.basename(f)[:-6]), f) for f in files if os.path.basename(f)[:-6].isdigit())


def flip_category(v2, pred, gold):
  value, used = v2.parse_number(pred)
  tags = set(used)
  if value is not None and str(value) != str(v2.number(gold)):
    tags.add("exact_value")          # e.g. 26.00 vs 26
  return "+".join(sorted(tags)) or "none"


def score_rows(v2, rows):
  out = []
  for r in rows:
    resp, gold = r["output"], str(r["gts"])
    rw1, acc1, fmt1, _ = v2.v1.boxed_math(resp, gold)
    rw2, acc2, _, _, _ = v2.boxed_math_v2(resp, gold)
    out.append((rw1, acc1, fmt1, rw2, acc2))
  return out


def rescore_val(v2, label, d, n_examples):
  res = []
  for step, f in steps_of(d):
    rows = [json.loads(x) for x in open(f) if x.strip()]
    sc = score_rows(v2, rows)
    stored = [float(r["acc"]) for r in rows] if all("acc" in r for r in rows) else None
    cats, ex = collections.Counter(), []
    for r, (_, a1, _, _, a2) in zip(rows, sc):
      if a2 > a1:
        pred = v2.v1.extract_boxed(r["output"])
        c = flip_category(v2, pred, str(r["gts"]))
        cats[c] += 1
        if len(ex) < n_examples:
          ex.append({"pred": pred, "gold": str(r["gts"]), "category": c})
    n = len(rows)
    acc1 = sum(x[1] for x in sc) / n
    acc2 = sum(x[4] for x in sc) / n
    res.append({
        "step": step, "n": n,
        "stored_acc": None if stored is None else sum(stored) / n,
        "v1_recomputed_mismatch": None if stored is None else sum(1 for s, x in zip(stored, sc) if s != x[1]),
        "v1_acc": acc1, "v2_acc": acc2, "delta_pp": 100 * (acc2 - acc1),
        "v2_only": sum(1 for x in sc if x[4] > x[1]), "v2_lost": sum(1 for x in sc if x[4] < x[1]),
        "fmt": sum(x[2] for x in sc) / n,
        "v2_only_by_category": dict(cats.most_common()), "examples": ex,
    })
    print(f"[{label}] step {step:>3}: n={n} v1 {100 * acc1:6.2f}% -> v2 {100 * acc2:6.2f}% "
          f"({100 * (acc2 - acc1):+.2f} pp, {res[-1]['v2_only']} answers) stored-v1 mismatch="
          f"{res[-1]['v1_recomputed_mismatch']}")
  return res


def rescore_train(v2, label, d):
  res = []
  for step, f in steps_of(d):
    rows = [json.loads(x) for x in open(f) if x.strip()]
    sc = score_rows(v2, rows)
    groups = collections.defaultdict(list)
    for r, x in zip(rows, sc):
      groups[r.get("qid", r.get("uid"))].append(x)

    def kinds(i):
      c = collections.Counter()
      for g in groups.values():
        vals = {x[i] for x in g}
        c["mixed" if len(vals) > 1 else ("all_correct" if vals == {1.0} else "all_wrong")] += 1
      return dict(c)

    n = len(rows)
    res.append({"step": step, "n": n, "groups": len(groups), "group_sizes": sorted({len(g) for g in groups.values()}),
                "v1_acc": sum(x[1] for x in sc) / n, "v2_acc": sum(x[4] for x in sc) / n,
                "v1_reward_mean": sum(x[0] for x in sc) / n, "v2_reward_mean": sum(x[3] for x in sc) / n,
                "v1_groups": kinds(1), "v2_groups": kinds(4)})
    e = res[-1]
    print(f"[{label}] train step {step}: v1 acc {100 * e['v1_acc']:.2f}% groups {e['v1_groups']} | "
          f"v2 acc {100 * e['v2_acc']:.2f}% groups {e['v2_groups']}")
  return res


def md_tables(val, groups, train):
  out = []
  for g, labels in groups.items():
    steps = sorted(set.intersection(*[{r["step"] for r in val[lb]} for lb in labels]))
    by = {lb: {r["step"]: r for r in val[lb]} for lb in labels}
    out.append(f"### {g}: eval accuracy re-scored, v1 -> v2 (%), same outputs\n")
    out.append("| step | " + " | ".join(f"{lb} v1 | {lb} v2" for lb in labels)
               + " | mean v1 | mean v2 | mean change (pp) |")
    out.append("|---:|" + "---:|---:|" * len(labels) + "---:|---:|---:|")
    for s in steps:
      a1 = [100 * by[lb][s]["v1_acc"] for lb in labels]
      a2 = [100 * by[lb][s]["v2_acc"] for lb in labels]
      cells = " | ".join(f"{x:.2f} | {y:.2f}" for x, y in zip(a1, a2))
      out.append(f"| {s} | {cells} | {statistics.mean(a1):.2f} | {statistics.mean(a2):.2f} | "
                 f"{statistics.mean(a2) - statistics.mean(a1):+.2f} |")
    s0, s1 = steps[0], steps[-1]
    for name, key in (("v1", "v1_acc"), ("v2", "v2_acc")):
      gains = [100 * (by[lb][s1][key] - by[lb][s0][key]) for lb in labels]
      out.append(f"\n{name} gain step {s0} -> {s1}: " + ", ".join(f"{lb} {x:+.2f}" for lb, x in zip(labels, gains))
                 + f"; mean {statistics.mean(gains):+.2f} pp")
    cats = collections.Counter()
    for lb in labels:
      for r in val[lb]:
        cats.update(r["v2_only_by_category"])
    tot = sum(r["v2_only"] for lb in labels for r in val[lb])
    lost = sum(r["v2_lost"] for lb in labels for r in val[lb])
    mism = sum(r["v1_recomputed_mismatch"] or 0 for lb in labels for r in val[lb])
    out.append(f"\nanswers credited only by v2: {tot} of {sum(r['n'] for lb in labels for r in val[lb])} scored; "
               f"answers v2 credits less than v1: {lost}; stored-v1 vs recomputed-v1 mismatches: {mism}")
    out.append("by whitelist step: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()) + "\n")
  for lb, rows in train.items():
    out.append(f"### {lb}: training batches re-scored (same completions)\n")
    out.append("| step | v1 acc % | v2 acc % | v1 mixed / all-correct / all-wrong groups "
               "| v2 mixed / all-correct / all-wrong groups |")
    out.append("|---:|---:|---:|---|---|")
    for e in rows:
      g1, g2 = e["v1_groups"], e["v2_groups"]
      out.append(f"| {e['step']} | {100 * e['v1_acc']:.2f} | {100 * e['v2_acc']:.2f} | "
                 f"{g1.get('mixed', 0)} / {g1.get('all_correct', 0)} / {g1.get('all_wrong', 0)} | "
                 f"{g2.get('mixed', 0)} / {g2.get('all_correct', 0)} / {g2.get('all_wrong', 0)} |")
    out.append("")
  return "\n".join(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--scorer", default=os.path.join(_HERE, "boxed_math_reward_v2.py"))
  ap.add_argument("--val", action="append", default=[], metavar="LABEL=DIR")
  ap.add_argument("--train", action="append", default=[], metavar="LABEL=DIR")
  ap.add_argument("--group", action="append", default=[], metavar="NAME=LABEL,LABEL")
  ap.add_argument("--examples", type=int, default=5)
  ap.add_argument("--json", required=True)
  ap.add_argument("--md")
  a = ap.parse_args()
  v2 = load_v2(a.scorer)
  val = {lb: rescore_val(v2, lb, d, a.examples) for lb, d in (x.split("=", 1) for x in a.val)}
  train = {lb: rescore_train(v2, lb, d) for lb, d in (x.split("=", 1) for x in a.train)}
  groups = {g: ls.split(",") for g, ls in (x.split("=", 1) for x in a.group)}
  for lb in val:
    if not any(lb in ls for ls in groups.values()):
      groups[lb] = [lb]
  out = {"scorer": {"path": a.scorer, "version": v2.SCORER_VERSION, "v1_sha256": v2.V1_SHA256},
         "inputs": {"val": a.val, "train": a.train}, "val": val, "train": train}
  with open(a.json, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
  if a.md:
    with open(a.md, "w") as f:
      f.write(md_tables(val, groups, train) + "\n")
  print("wrote", a.json, a.md or "")


if __name__ == "__main__":
  main()
