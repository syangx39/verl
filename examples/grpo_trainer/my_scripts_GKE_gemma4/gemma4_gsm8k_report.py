#!/usr/bin/env python3
"""Seed report for the Gemma-4 E2B GSM8K 8K baseline (PR #14): tables, JSON and figures from one run's evidence.

    PYTHONPATH=<matplotlib dir> python3 gemma4_gsm8k_report.py --exp EXP --label seed1 --out DIR \\
        [--qwen-rescore gsm8k_8k/scorer_v2/rescore_v1_v2.json] [--model /workspace/meta-RL/models/gemma-4-E2B-it]

Reads /workspace/meta-RL/logs/<EXP>.log (console metrics), .../gemma4_gsm8k/<EXP>/{val,train}/<step>.jsonl,
/workspace/meta-RL/logs/gemma4_gsm8k_launches.tsv (launch time) and /workspace/meta-RL/ckpt/<EXP>/ (checkpoints).

Eval (all 1,319 questions, greedy) per checkpoint:
  * v2 accuracy (training scorer) with a Wilson 95% interval, v1 accuracy (diagnostic) and boxed rate;
  * the change from the run's own step 0, paired by question: mean difference and a 95% bootstrap interval
    (10,000 resamples of questions, seed 0), wrong->right / right->wrong counts, and the wrong->right answers split
    by their step-0 state (no box / boxed but wrong). The split describes the outputs; it does not separate format
    from math, because whether an unboxed step-0 output already held the right number is not checked;
  * output length in Gemma tokens (mean / p95 / max) and truncation (outputs of >= --cap tokens).
Training per update: the console metrics (LR, grad norm, loss, entropy, rollout correction, lengths, timing), the
rollout dumps (v2 / v1 accuracy, boxed rate, all-correct / all-wrong / mixed prompt groups) and the actual interval
between updates from the dump timestamps.
Timing: internal step time is timing_s/step (generation through the rollout dump, including a checkpoint save). The
actual interval between updates is taken from consecutive rollout-dump timestamps; it also covers the work outside that
timer: validation, console / TensorBoard logging and data loading. Both are summarized over steps 20-250 excluding eval
and checkpoint steps (intervals that contain an eval are excluded), plus the launch-to-finish duration.
Figures: <label>_eval_curve.png, <label>_eval_flips.png, <label>_training.png, <label>_timing.png.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone

import numpy as np

STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
WRAPPED_RE = re.compile(r"^[A-Za-z_][\w.]*\(\s*([^,()]*?)\s*(?:,.*)?\)$")


def parse_log(path):
  steps = {}
  with open(path, errors="replace") as f:
    for ln in f:
      m = STEP_RE.search(ln)
      if not m:
        continue
      kv = steps.setdefault(int(m.group(1)), {})
      for part in m.group(2).split(" - "):
        if ":" not in part:
          continue
        k, v = part.split(":", 1)
        v = v.strip()
        mm = WRAPPED_RE.match(v)
        v = mm.group(1) if mm else v
        try:
          kv[k.strip()] = float(v)
        except ValueError:
          pass
  return steps


def by_step(d):
  out = []
  for f in glob.glob(os.path.join(d, "*.jsonl")):
    b = os.path.basename(f)[:-6]
    if b.isdigit():
      out.append((int(b), f))
  return sorted(out)


def jsonl(path):
  with open(path) as f:
    return [json.loads(x) for x in f if x.strip()]


def wilson(k, n, z=1.959964):
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return c - h, c + h


def eval_section(val_dir, tok, cap):
  files = by_step(val_dir)
  base = {float(r["qid"]): r for r in jsonl(files[0][1])}
  assert files[0][0] == 0, "no step-0 eval dump"
  qids = sorted(base)
  n = len(qids)
  rng = np.random.default_rng(0)
  idx = rng.integers(0, n, size=(10000, n))
  a0 = np.array([float(base[q]["acc"]) for q in qids])
  f0 = np.array([float(base[q]["fmt"]) for q in qids])
  out = []
  for s, f in files:
    cur = {float(r["qid"]): r for r in jsonl(f)}
    assert sorted(cur) == qids, f"eval step {s}: question IDs differ from step 0"
    a = np.array([float(cur[q]["acc"]) for q in qids])
    a1 = np.array([float(cur[q].get("acc_v1", "nan")) for q in qids])
    fm = np.array([float(cur[q]["fmt"]) for q in qids])
    d = a - a0
    boots = np.sort(d[idx].mean(axis=1))
    lens = np.array([len(tok(cur[q]["output"], add_special_tokens=False)["input_ids"]) for q in qids]) if tok else None
    k = int(a.sum())
    lo, hi = wilson(k, n)
    wr = (a0 == 0) & (a == 1)
    out.append({
        "step": s, "n": n, "acc_v2": float(a.mean()), "acc_v2_wilson95": [lo, hi], "acc_v1": float(np.nanmean(a1)),
        "boxed": float(fm.mean()), "delta_vs_step0": float(d.mean()),
        "delta_boot95": [float(boots[249]), float(boots[9749])],
        "wrong_to_right": int(wr.sum()), "right_to_wrong": int(((a0 == 1) & (a == 0)).sum()),
        "wrong_to_right_step0_no_box": int((wr & (f0 == 0)).sum()),
        "wrong_to_right_step0_boxed_wrong": int((wr & (f0 == 1)).sum()),
        "tokens_mean": float(lens.mean()) if lens is not None else None,
        "tokens_p95": int(np.percentile(lens, 95)) if lens is not None else None,
        "tokens_max": int(lens.max()) if lens is not None else None,
        "truncated": int((lens >= cap).sum()) if lens is not None else None,
        "chars_mean": float(np.mean([len(cur[q]["output"]) for q in qids])),
    })
    e = out[-1]
    print(f"eval {s:>3}: v2 {100 * e['acc_v2']:.2f}% v1 {100 * e['acc_v1']:.2f}% boxed {100 * e['boxed']:.2f}% "
          f"d {100 * e['delta_vs_step0']:+.2f} pp [{100 * e['delta_boot95'][0]:+.2f}, {100 * e['delta_boot95'][1]:+.2f}] "
          f"w->r {e['wrong_to_right']} r->w {e['right_to_wrong']} trunc {e['truncated']}", flush=True)
  return out


def train_section(train_dir, steps):
  rows_out = {}
  mtimes = {}
  for s, f in by_step(train_dir):
    mtimes[s] = os.stat(f).st_mtime
    rows = jsonl(f)
    groups = {}
    for r in rows:
      groups.setdefault(r["qid"], []).append(float(r["acc"]))
    kinds = {"mixed": 0, "all_correct": 0, "all_wrong": 0}
    for v in groups.values():
      kinds["mixed" if len(set(v)) > 1 else ("all_correct" if v[0] == 1.0 else "all_wrong")] += 1
    n = len(rows)
    rows_out[s] = {"acc_v2": sum(float(r["acc"]) for r in rows) / n,
                   "acc_v1": sum(float(r.get("acc_v1", "nan")) for r in rows) / n,
                   "boxed": sum(float(r["fmt"]) for r in rows) / n, **kinds, "rows": n}
  for s in sorted(mtimes):
    if s - 1 in mtimes:
      rows_out[s]["interval_s"] = mtimes[s] - mtimes[s - 1]
  return rows_out


def timing_section(steps, train, eval_steps, ckpt_steps, launch_utc, end_ts):
  last = max(steps)
  busy = set(eval_steps) | set(ckpt_steps)
  win = [s for s in range(20, last + 1) if s in steps and s not in busy]
  internal = [steps[s]["timing_s/step"] for s in win if "timing_s/step" in steps[s]]
  interval = [train[s]["interval_s"] for s in win if s in train and "interval_s" in train[s] and s - 1 not in busy]

  def summ(v):
    v = sorted(v)
    return {"n": len(v), "median": statistics.median(v), "mean": statistics.mean(v), "p90": v[int(0.9 * (len(v) - 1))]}

  comp = {k: statistics.median(steps[s][k] for s in win if k in steps[s])
          for k in ("timing_s/gen", "timing_s/update_actor", "timing_s/update_weights", "timing_s/dump_rollout_generations")}
  evals = {s: steps[s]["timing_s/testing"] for s in eval_steps if s in steps and "timing_s/testing" in steps[s]}
  saves = {s: steps[s]["timing_s/save_checkpoint"] for s in ckpt_steps if s in steps and "timing_s/save_checkpoint" in steps[s]}
  dur = (end_ts - launch_utc) if (launch_utc and end_ts) else None
  return {"window": f"steps 20-{last} excluding eval/checkpoint steps", "internal_step_s": summ(internal),
          "actual_interval_s": summ(interval), "internal_components_median_s": comp, "eval_s": evals,
          "checkpoint_save_s": saves, "launch_to_finish_h": dur / 3600 if dur else None,
          "gpu_hours": 64 * dur / 3600 if dur else None}


def plots(label, ev, steps, train, tim, qwen, out):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  es = [e["step"] for e in ev]
  # 1. eval curve
  fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
  acc = [100 * e["acc_v2"] for e in ev]
  err = [[100 * (e["acc_v2"] - e["acc_v2_wilson95"][0]) for e in ev], [100 * (e["acc_v2_wilson95"][1] - e["acc_v2"]) for e in ev]]
  ax[0].errorbar(es, acc, yerr=err, marker="o", capsize=3, color="C0", label=f"Gemma-4 E2B {label}: v2 accuracy (Wilson 95%)")
  ax[0].plot(es, [100 * e["acc_v1"] for e in ev], marker=".", ls=":", color="C1", label="same outputs, v1 scorer (diagnostic)")
  if qwen:
    qs = sorted(qwen)
    ax[0].plot(qs, [statistics.mean(qwen[s]) for s in qs], ls="--", color="gray",
               label="Qwen3-0.6B 8K, 3-seed mean (v1-trained, evals re-scored v2; context)")
    ax[0].fill_between(qs, [min(qwen[s]) for s in qs], [max(qwen[s]) for s in qs], color="gray", alpha=0.15)
  ax[0].set_xlabel("update")
  ax[0].set_ylabel("held-out accuracy, %  (1,319 questions, greedy)")
  ax[0].set_title("Held-out accuracy")
  ax[0].legend(fontsize=7, loc="lower right")
  ax[0].grid(alpha=0.3)
  d = [100 * e["delta_vs_step0"] for e in ev]
  derr = [[100 * (e["delta_vs_step0"] - e["delta_boot95"][0]) for e in ev], [100 * (e["delta_boot95"][1] - e["delta_vs_step0"]) for e in ev]]
  ax[1].errorbar(es, d, yerr=derr, marker="o", capsize=3, color="C2")
  ax[1].axhline(0, color="k", lw=0.8)
  ax[1].set_xlabel("update")
  ax[1].set_ylabel("v2 accuracy change vs step 0, pp")
  ax[1].set_title("Change from step 0 (paired by question, bootstrap 95%)")
  ax[1].grid(alpha=0.3)
  ax[2].plot(es, [100 * e["boxed"] for e in ev], marker="o", color="C3", label="boxed answer present, %")
  ax[2].set_xlabel("update")
  ax[2].set_ylabel("boxed rate, %", color="C3")
  ax2 = ax[2].twinx()
  ax2.plot(es, [e["tokens_mean"] for e in ev], marker="s", color="C4", label="mean output tokens")
  ax2.plot(es, [e["tokens_p95"] for e in ev], marker="^", ls="--", color="C4", label="p95 output tokens")
  ax2.set_ylabel("output tokens", color="C4")
  for i, e in enumerate(x for x in ev if x["truncated"]):
    ax2.annotate(f"{e['truncated']} at cap", (e["step"], e["tokens_p95"]), fontsize=7, color="C4",
                 xytext=(0, 7 if i % 2 == 0 else -13), textcoords="offset points", ha="center")
  ax[2].set_title("Eval format and length (labels: outputs at the 8,192-token cap)")
  h1, l1 = ax[2].get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax[2].legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
  ax[2].grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(os.path.join(out, f"{label}_eval_curve.png"), dpi=130)
  plt.close(fig)

  # 2. flips
  fig, ax = plt.subplots(figsize=(9, 4.2))
  e2 = [e for e in ev if e["step"] > 0]
  x = [e["step"] for e in e2]
  nb = [e["wrong_to_right_step0_no_box"] for e in e2]
  bw = [e["wrong_to_right_step0_boxed_wrong"] for e in e2]
  rw = [-e["right_to_wrong"] for e in e2]
  ax.bar(x, nb, width=6, color="C0", label="wrong -> right, no box at step 0")
  ax.bar(x, bw, width=6, bottom=nb, color="C9", label="wrong -> right, boxed but wrong at step 0")
  ax.bar(x, rw, width=6, color="C3", label="right -> wrong")
  ax.plot(x, [a + b + c for a, b, c in zip(nb, bw, rw)], "k.-", label="net (= change x 1,319)")
  ax.axhline(0, color="k", lw=0.8)
  ax.set_ylim(min(rw + [0]) * 1.6, max(a + b for a, b in zip(nb, bw)) * 1.45)
  ax.set_xlabel("update")
  ax.set_ylabel("questions (of 1,319)")
  ax.set_title("Per-question changes vs step 0 (descriptive: does not separate format from math)")
  ax.legend(fontsize=8, loc="upper center", ncol=2)
  ax.grid(alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(os.path.join(out, f"{label}_eval_flips.png"), dpi=130)
  plt.close(fig)

  # 3. training diagnostics
  ts = sorted(s for s in steps if s > 0 and "actor/grad_norm" in steps[s])

  def ser(k):
    return [steps[s].get(k, float("nan")) for s in ts]

  def roll(v, w=10):
    out_ = []
    for i in range(len(v)):
      seg = [x for x in v[max(0, i - w + 1):i + 1] if not math.isnan(x)]
      out_.append(sum(seg) / len(seg) if seg else float("nan"))
    return out_

  tr = [train.get(s, {}) for s in ts]
  fig, ax = plt.subplots(3, 3, figsize=(17, 11))
  a = ax[0, 0]
  a.plot(ts, [100 * t.get("acc_v2", float("nan")) for t in tr], lw=0.8, color="C0", label="v2 accuracy")
  a.plot(ts, [100 * t.get("acc_v1", float("nan")) for t in tr], lw=0.8, color="C1", label="v1 accuracy (diagnostic)")
  a.plot(ts, [100 * t.get("boxed", float("nan")) for t in tr], lw=0.8, color="C3", label="boxed rate")
  a.set_title("Training batches (T=1, 2,048 completions)")
  a.set_ylabel("%")
  a.legend(fontsize=7)
  a = ax[0, 1]
  for k, c in (("mixed", "C2"), ("all_correct", "C0"), ("all_wrong", "C3")):
    a.plot(ts, [t.get(k, float("nan")) for t in tr], lw=0.8, color=c, label=k.replace("_", "-"))
  a.set_title("Prompt groups of 16 (mixed = non-zero GRPO advantage)")
  a.set_ylabel("groups (of 128)")
  a.legend(fontsize=7)
  a = ax[0, 2]
  a.plot(ts, ser("actor/lr"), color="k")
  a.set_title("Logged LR (post-scheduler)")
  a = ax[1, 0]
  a.semilogy(ts, ser("actor/grad_norm"), lw=0.8, color="C0")
  a.axhline(1.0, color="r", ls="--", lw=0.8, label="clip = 1.0")
  a.set_title("Grad norm before clipping")
  a.legend(fontsize=7)
  a = ax[1, 1]
  a.plot(ts, ser("actor/entropy_loss"), lw=0.6, color="C4", alpha=0.6, label="per update")
  a.plot(ts, roll(ser("actor/entropy_loss")), color="C4", label="10-update mean")
  a.set_title("Token entropy of the update batch (no bonus)")
  a.legend(fontsize=7)
  a = ax[1, 2]
  a.plot(ts, ser("response_length/mean"), color="C5", label="mean (left)")
  a.set_ylabel("tokens (mean)")
  a2 = a.twinx()
  a2.plot(ts, ser("response_length/max"), color="C7", lw=0.6, label="max (right)")
  a2.set_ylabel("tokens (max)")
  caps = [round(steps[s].get("response_length/clip_ratio", 0) * 2048) for s in ts]
  for s, c in zip(ts, caps):
    if c:
      a2.annotate(f"{c} at cap", (s, steps[s]["response_length/max"]), fontsize=7, color="r", xytext=(4, -4),
                  textcoords="offset points")
  h1, l1 = a.get_legend_handles_labels()
  h2, l2 = a2.get_legend_handles_labels()
  a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
  a.set_title("Response length (training; label: completions at the 8,192 cap)")
  a = ax[2, 0]
  a.plot(ts, ser("actor/rollout_corr/k3_kl"), lw=0.8, color="C6", label="k3_kl")
  a.plot(ts, ser("actor/rollout_corr/kl"), lw=0.8, color="C8", label="kl")
  a.set_title("Trainer vs sampler log-prob mismatch")
  a.legend(fontsize=7)
  a = ax[2, 1]
  a.plot(ts, ser("actor/rollout_corr/rollout_is_max"), lw=0.8, color="C1", label="token IS max, cap 3.0 (left)")
  a.set_ylabel("IS max")
  a2 = a.twinx()
  a2.plot(ts, ser("actor/rollout_corr/rollout_is_eff_sample_size"), lw=0.8, color="C2", label="ESS (right)")
  a2.set_ylabel("ESS")
  h1, l1 = a.get_legend_handles_labels()
  h2, l2 = a2.get_legend_handles_labels()
  a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
  a.set_title("Token TIS weights")
  a = ax[2, 2]
  a.plot(ts, ser("actor/pg_loss"), lw=0.6, color="C0", alpha=0.6, label="per update")
  a.plot(ts, roll(ser("actor/pg_loss")), color="C0", label="10-update mean")
  a.axhline(0, color="k", lw=0.6)
  a.set_title("Policy-gradient loss")
  a.legend(fontsize=7)
  for a in ax.flat:
    a.set_xlabel("update")
    a.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(os.path.join(out, f"{label}_training.png"), dpi=120)
  plt.close(fig)

  # 4. timing
  fig, ax = plt.subplots(figsize=(11, 4.4))
  gen, upd, wsync = ser("timing_s/gen"), ser("timing_s/update_actor"), ser("timing_s/update_weights")
  step_t = ser("timing_s/step")
  other = [max(0.0, st - g - u - w) for st, g, u, w in zip(step_t, gen, upd, wsync)]
  ax.bar(ts, gen, width=1.0, color="C0", label="generation")
  ax.bar(ts, upd, width=1.0, bottom=gen, color="C1", label="actor update")
  b2 = [g + u for g, u in zip(gen, upd)]
  ax.bar(ts, wsync, width=1.0, bottom=b2, color="C2", label="weight sync")
  b3 = [x + w for x, w in zip(b2, wsync)]
  ax.bar(ts, other, width=1.0, bottom=b3, color="C7", label="rest of timing_s/step (checkpoint save, reward, adv, dump)")
  busy = set(tim.get("eval_s", {})) | set(tim.get("checkpoint_save_s", {}))
  iv = [train.get(s, {}).get("interval_s", float("nan")) if s - 1 not in busy else float("nan") for s in ts]
  ax.plot(ts, iv, "k.", ms=3, label="actual interval between updates (dump timestamps; intervals containing an eval omitted)")
  ax.axhline(tim["internal_step_s"]["median"], color="C1", ls="--", lw=1,
             label=f"internal step median {tim['internal_step_s']['median']:.1f} s")
  ax.axhline(tim["actual_interval_s"]["median"], color="k", ls="--", lw=1,
             label=f"actual interval median {tim['actual_interval_s']['median']:.1f} s")
  ax.set_ylim(0, max(120, tim["actual_interval_s"]["p90"] * 1.5))
  ax.set_xlabel("update")
  ax.set_ylabel("seconds")
  ax.set_title("Internal step time (timing_s/step) vs actual interval between updates\n"
               "(the gap is mostly TensorBoard writes to the GCS-fuse bucket, ~48 s per update; eval runs outside timing_s/step)",
               fontsize=10)
  ax.legend(fontsize=7, ncol=2, loc="upper right")
  ax.grid(alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(os.path.join(out, f"{label}_timing.png"), dpi=130)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--exp", required=True)
  ap.add_argument("--label", default="seed1")
  ap.add_argument("--out", required=True)
  ap.add_argument("--logs", default="/workspace/meta-RL/logs")
  ap.add_argument("--ckpt-root", default="/workspace/meta-RL/ckpt")
  ap.add_argument("--model", default="/workspace/meta-RL/models/gemma-4-E2B-it", help="tokenizer for output lengths")
  ap.add_argument("--cap", type=int, default=8192)
  ap.add_argument("--qwen-rescore", help="rescore_v1_v2.json: Qwen 3-seed eval curves under v2, plotted as context")
  ap.add_argument("--no-plots", action="store_true")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  dump = os.path.join(a.logs, "gemma4_gsm8k", a.exp)
  log_path = os.path.join(a.logs, f"{a.exp}.log")
  steps = parse_log(log_path)
  tok = None
  if a.model and os.path.isdir(a.model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
  ev = eval_section(os.path.join(dump, "val"), tok, a.cap)
  train = train_section(os.path.join(dump, "train"), steps)
  launch = None
  tsv = os.path.join(a.logs, "gemma4_gsm8k_launches.tsv")
  if os.path.exists(tsv):
    for ln in open(tsv):
      p = ln.rstrip("\n").split("\t")
      if len(p) == 3 and p[2] == a.exp:
        launch = datetime.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
  log_text = open(log_path, errors="replace").read()
  finished = "LAUNCHER_RC=" in log_text
  end_ts = os.stat(log_path).st_mtime if finished else None
  ck = sorted(int(m) for m in re.findall(r"\[GEMMA4-CKPT-STEPS\] \S+ pid=\d+ saved global_step=(\d+)", log_text))
  skipped = sorted(int(m) for m in re.findall(r"skip save at global_step=(\d+)", log_text))
  ckdir = os.path.join(a.ckpt_root, a.exp)
  ckpts = sorted(x for x in os.listdir(ckdir) if x.startswith("global_step_")) if os.path.isdir(ckdir) else []
  eval_steps = [e["step"] for e in ev]
  tim = timing_section(steps, train, [s for s in eval_steps if s > 0], ck, launch, end_ts)
  tr_steps = sorted(s for s in steps if s > 0 and "response_length/mean" in steps[s])
  gen_tokens = sum(steps[s]["response_length/mean"] * 2048 for s in tr_steps)
  rc = re.findall(r"main_ppo finished with exit_code=(\d+)", log_text)
  qwen = {}
  if a.qwen_rescore and os.path.exists(a.qwen_rescore):
    q = json.load(open(a.qwen_rescore))
    for lb, rows in q["val"].items():
      if lb.startswith("qwen"):
        for r in rows:
          qwen.setdefault(r["step"], []).append(100 * r["v2_acc"])
  summary = {"exp": a.exp, "label": a.label, "finished": finished, "driver_exit_code": int(rc[-1]) if rc else None,
             "last_step": max(tr_steps) if tr_steps else None, "eval": ev,
             "checkpoints": {"saved_by_hook": ck, "skipped_by_hook": skipped, "dirs": ckpts},
             "timing": tim, "generated_train_tokens": gen_tokens,
             "train": {str(s): {**train.get(s, {}), **{k: steps[s].get(k) for k in (
                 "actor/lr", "actor/grad_norm", "actor/pg_loss", "actor/entropy_loss", "actor/rollout_corr/k3_kl",
                 "actor/rollout_corr/kl", "actor/rollout_corr/rollout_is_max", "actor/rollout_corr/rollout_is_mean",
                 "actor/rollout_corr/rollout_is_ratio_fraction_high", "actor/rollout_corr/rollout_is_eff_sample_size",
                 "critic/score/mean", "response_length/mean", "response_length/max", "response_length/clip_ratio",
                 "timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/update_weights")}}
                       for s in tr_steps}}
  with open(os.path.join(a.out, f"{a.label}_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
  if not a.no_plots:
    plots(a.label, ev, steps, train, tim, qwen, a.out)
  t = tim
  print(f"finished={finished} exit={summary['driver_exit_code']} last_step={summary['last_step']} ckpt saved={ck} "
        f"skipped={skipped} dirs={ckpts}")
  print(f"timing: internal median {t['internal_step_s']['median']:.1f}s (n={t['internal_step_s']['n']}), actual interval "
        f"median {t['actual_interval_s']['median']:.1f}s (n={t['actual_interval_s']['n']}), components {t['internal_components_median_s']}, "
        f"launch-to-finish {t['launch_to_finish_h']} h, GPU-hours {t['gpu_hours']}, train tokens {gen_tokens:.3e}")
  print("wrote", a.out)


if __name__ == "__main__":
  sys.exit(main())
