#!/usr/bin/env python3
"""Multi-seed aggregate for the Gemma-4 E2B GSM8K 8K baseline (PR #14).

    PYTHONPATH=<matplotlib dir> python3 gemma4_gsm8k_aggregate.py --out DIR \\
        --run seed1=EXP1 --run seed2=EXP2 --run seed3=EXP3 \\
        [--evidence gsm8k_8k] [--qwen-rescore gsm8k_8k/scorer_v2/rescore_v1_v2.json]

Per run it reads the per-question eval dumps /workspace/meta-RL/logs/gemma4_gsm8k/<EXP>/val/<step>.jsonl and the
per-seed report <evidence>/<label>/<label>_summary.json written by gemma4_gsm8k_report.py. Each seed's accuracy
recomputed from the dumps must equal its report's.

Eval, at every checkpoint the runs share:
  * each seed's v2 accuracy and its paired change from its own step 0 (question-paired bootstrap 95%, from the
    per-seed report);
  * across seeds: mean, sample SD and range of the accuracy and of the change;
  * the mean change with two intervals, kept separate:
      - question bootstrap 95%: the 1,319 questions resampled jointly for all seeds (10,000 resamples, seed 0);
        it covers question sampling only;
      - seed-level t 95%: mean +- t(0.975, n-1) * SD / sqrt(n) over the n seeds' changes; it covers training-seed
        variation. With n = 3 (df = 2) it is wide, and it assumes roughly normal seed effects;
  * cross-seed agreement: questions right in every run, wrong in every run, or split, with the split questions whose
    box status also differs, and the questions whose outputs are byte-identical in every run. At step 0 all runs
    evaluate the same initial weights, so the split there measures the launch-to-launch variation of greedy eval of
    the initial model alone; it is not a noise floor for trained checkpoints.
Training: per-seed windows of entropy, length, batch accuracy and prompt-group mix, and numerics (k3_kl, token IS,
ESS, grad-norm spikes, non-finite values). Timing and cost per seed come from the per-seed reports, clock source
labeled. The steady-state timing window is the per-seed report's (default steps 20-250 without eval/checkpoint steps).
Figures: aggregate_eval.png, aggregate_training.png, aggregate_cost.png. Markdown tables go to stdout.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import statistics
import sys

import numpy as np

# Two-sided 95% Student t quantiles, by degrees of freedom.
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}
WINDOWS = ((1, 20), (21, 40), (81, 100), (161, 180), (231, 250))
NUM_KEYS = ("actor/grad_norm", "actor/pg_loss", "actor/entropy_loss", "actor/rollout_corr/k3_kl",
            "actor/rollout_corr/kl", "actor/rollout_corr/rollout_is_max",
            "actor/rollout_corr/rollout_is_ratio_fraction_high", "actor/rollout_corr/rollout_is_eff_sample_size",
            "critic/score/mean", "response_length/mean")
BATCH = 2048
GPUS = 64
COLORS = ("C0", "C1", "C2", "C4", "C5", "C6")


def jsonl(path):
  with open(path) as f:
    return [json.loads(x) for x in f if x.strip()]


def load_json(path):
  with open(path) as f:
    return json.load(f)


def load_eval(dump_dir):
  """{step: {qid: (v2 acc, boxed, sha1 of the output)}} from <dump_dir>/val/<step>.jsonl."""
  out = {}
  for f in glob.glob(os.path.join(dump_dir, "val", "*.jsonl")):
    b = os.path.basename(f)[:-6]
    if b.isdigit():
      out[int(b)] = {float(r["qid"]): (float(r["acc"]), float(r["fmt"]), hashlib.sha1(r["output"].encode()).hexdigest())
                     for r in jsonl(f)}
  return out


def spread(v):
  v = [float(x) for x in v]
  return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None, "min": min(v), "max": max(v)}


def eval_aggregate(labels, ev, reports, n_boot=10000):
  """Per shared checkpoint: per-seed values, across-seed spread, both intervals and cross-seed agreement."""
  steps = sorted(set.intersection(*(set(ev[lb]) for lb in labels)))
  assert steps and steps[0] == 0, "the runs share no step-0 eval"
  qids = sorted(ev[labels[0]][0])
  for lb in labels:
    for s in steps:
      assert sorted(ev[lb][s]) == qids, f"{lb} step {s}: question IDs differ from {labels[0]} step 0"
  acc = {s: np.array([[ev[lb][s][q][0] for q in qids] for lb in labels]) for s in steps}
  box = {s: np.array([[ev[lb][s][q][1] for q in qids] for lb in labels]) for s in steps}
  n, nq = len(labels), len(qids)
  idx = np.random.default_rng(0).integers(0, nq, size=(n_boot, nq))
  lo_i, hi_i = int(0.025 * n_boot) - 1, int(0.975 * n_boot) - 1
  rep = {lb: {e["step"]: e for e in reports[lb]["eval"]} for lb in labels}
  rows = []
  for s in steps:
    a = acc[s]
    d = a - acc[0]
    per_acc = a.mean(axis=1)
    per_d = d.mean(axis=1)
    for i, lb in enumerate(labels):
      r = rep[lb].get(s)
      assert r is not None, f"{lb} step {s}: not in the per-seed report"
      assert abs(r["acc_v2"] - per_acc[i]) < 1e-9, f"{lb} step {s}: dump accuracy {per_acc[i]} != report {r['acc_v2']}"
    boots = np.sort(d.mean(axis=0)[idx].mean(axis=1))
    sd = float(np.std(per_d, ddof=1)) if n > 1 else None
    half = T975[n - 1] * sd / math.sqrt(n) if n > 1 else None
    right_all = int((a.min(axis=0) == 1).sum())
    wrong_all = int((a.max(axis=0) == 0).sum())
    split = a.min(axis=0) != a.max(axis=0)
    box_differs = box[s].min(axis=0) != box[s].max(axis=0)
    rows.append({
        "step": s,
        "per_seed": {lb: {"acc_v2": float(per_acc[i]), "delta_vs_step0": float(per_d[i]),
                          **{k: rep[lb][s].get(k) for k in ("acc_v2_wilson95", "delta_boot95", "acc_v1", "boxed",
                                                            "wrong_to_right", "right_to_wrong", "truncated",
                                                            "tokens_mean", "tokens_p95")}}
                     for i, lb in enumerate(labels)},
        "acc_v2": spread(per_acc), "delta": spread(per_d),
        "delta_question_boot95": [float(boots[lo_i]), float(boots[hi_i])],
        "delta_seed_t95": [float(per_d.mean() - half), float(per_d.mean() + half)] if half is not None else None,
        "right_in_all": right_all, "wrong_in_all": wrong_all, "split": nq - right_all - wrong_all,
        "split_box_status_differs": int((split & box_differs).sum()),
        "identical_outputs_in_all": sum(1 for q in qids if len({ev[lb][s][q][2] for lb in labels}) == 1),
    })
  return steps, nq, rows


def train_series(report):
  return {int(k): v for k, v in report["train"].items()}


def train_aggregate(labels, reports):
  out = {}
  for lb in labels:
    tr = train_series(reports[lb])
    ts = sorted(tr)

    def vals(k, seg=None):
      return [tr[s][k] for s in (ts if seg is None else seg) if tr[s].get(k) is not None]

    win = []
    for lo, hi in WINDOWS:
      seg = [s for s in ts if lo <= s <= hi]
      if not seg:
        continue

      def m(k, scale=1.0):
        v = vals(k, seg)
        return scale * statistics.mean(v) if v else None

      win.append({"updates": f"{lo}-{hi}", "n": len(seg), "entropy": m("actor/entropy_loss"),
                  "tokens": m("response_length/mean"), "train_acc_v2": m("acc_v2", 100), "train_acc_v1": m("acc_v1", 100),
                  "boxed": m("boxed", 100), "mixed": m("mixed"), "all_correct": m("all_correct"),
                  "all_wrong": m("all_wrong")})

    def rng(k):
      v = [x for x in vals(k) if math.isfinite(x)]
      return [min(v), max(v)] if v else None

    g = [(s, tr[s]["actor/grad_norm"]) for s in ts if tr[s].get("actor/grad_norm") is not None]
    spikes = sorted(((v, s) for s, v in g if v > 1.0), reverse=True)
    out[lb] = {
        "updates": len(ts), "last_update": ts[-1] if ts else None, "windows": win,
        "nonfinite_values": sum(1 for s in ts for k in NUM_KEYS
                                if tr[s].get(k) is not None and not math.isfinite(tr[s][k])),
        "missing_values": sum(1 for s in ts for k in NUM_KEYS if tr[s].get(k) is None),
        "k3_kl_range": rng("actor/rollout_corr/k3_kl"), "kl_range": rng("actor/rollout_corr/kl"),
        "is_max_range": rng("actor/rollout_corr/rollout_is_max"),
        "is_fraction_high_range": rng("actor/rollout_corr/rollout_is_ratio_fraction_high"),
        "ess_range": rng("actor/rollout_corr/rollout_is_eff_sample_size"),
        "grad_norm_median": statistics.median(v for _, v in g) if g else None,
        "grad_norm_above_clip": len(spikes), "grad_norm_top": [[s, v] for v, s in spikes[:5]],
        "train_cap_hits": sum(round((tr[s].get("response_length/clip_ratio") or 0.0) * BATCH) for s in ts),
    }
  return out


def window_steps(report):
  """The per-seed report's steady-state timing window: steps start..end without eval and checkpoint steps."""
  tr = train_series(report)
  t = report["timing"]
  busy = {e["step"] for e in report["eval"] if e["step"] > 0} | set(report["checkpoints"].get("saved_by_hook", []))
  parts = t["window"].split()[1].split("-")  # "steps 20-250 excluding ..."
  lo, hi = int(parts[0]), int(parts[1])
  return [s for s in range(lo, hi + 1) if s in tr and s not in busy]


def timing_aggregate(labels, reports):
  out = {}
  for lb in labels:
    r = reports[lb]
    t = r["timing"]
    out[lb] = {k: t.get(k) for k in ("window", "interval_source", "internal_step_s", "actual_interval_s",
                                     "paired_gap_s", "paired_steps", "launch_to_finish_h", "gpu_hours",
                                     "internal_components_median_s", "checkpoint_save_s")}
    out[lb]["eval_total_s"] = sum(t.get("eval_s", {}).values())
    out[lb]["generated_train_tokens"] = r.get("generated_train_tokens")
  return out


def cumulative(report, key, scale=1.0):
  """{step: running sum of train[step][key] * scale} from update 1; None from the first missing value on."""
  tr = train_series(report)
  out, tot = {0: 0.0}, 0.0
  for s in sorted(tr):
    v = tr[s].get(key)
    if v is None or tot is None:
      tot = None
    else:
      tot += float(v) * scale
    out[s] = tot
  return out


def qwen_curves(path):
  if not path or not os.path.exists(path):
    return {}
  q = load_json(path)
  out = {}
  for lb, rows in q["val"].items():
    if lb.startswith("qwen"):
      out[lb] = {r["step"]: 100 * r["v2_acc"] for r in rows}
  return out


def roll(v, w=10):
  out = []
  for i in range(len(v)):
    seg = [x for x in v[max(0, i - w + 1):i + 1] if not math.isnan(x)]
    out.append(sum(seg) / len(seg) if seg else float("nan"))
  return out


def plots(labels, steps, rows, reports, qwen, out):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  n = len(labels)
  # 1. eval
  fig, ax = plt.subplots(1, 3, figsize=(18, 4.9))
  a = ax[0]
  for i, lb in enumerate(labels):
    a.plot(steps, [100 * r["per_seed"][lb]["acc_v2"] for r in rows], marker=".", lw=1, color=COLORS[i],
           label=f"Gemma-4 E2B {lb}")
  a.plot(steps, [100 * r["acc_v2"]["mean"] for r in rows], color="k", lw=2, label=f"Gemma {n}-seed mean")
  if qwen:
    qs = sorted(set.intersection(*(set(v) for v in qwen.values())))
    a.plot(qs, [statistics.mean(v[s] for v in qwen.values()) for s in qs], ls="--", color="gray",
           label=f"Qwen3-0.6B 8K, {len(qwen)}-seed mean and range (v1-trained, evals re-scored v2; context)")
    a.fill_between(qs, [min(v[s] for v in qwen.values()) for s in qs], [max(v[s] for v in qwen.values()) for s in qs],
                   color="gray", alpha=0.15)
  a.set_xlabel("update")
  a.set_ylabel("held-out v2 accuracy, %  (1,319 questions, greedy)")
  a.set_title("Held-out accuracy, every seed")
  a.legend(fontsize=7, loc="center right")
  a.grid(alpha=0.3)
  a = ax[1]
  off = np.linspace(-2.5, 2.5, n) if n > 1 else np.zeros(1)
  later = rows[1:]
  for i, lb in enumerate(labels):
    p = [r["per_seed"][lb] for r in later]
    a.errorbar([r["step"] + off[i] for r in later], [100 * x["delta_vs_step0"] for x in p],
               yerr=[[100 * (x["delta_vs_step0"] - x["delta_boot95"][0]) for x in p],
                     [100 * (x["delta_boot95"][1] - x["delta_vs_step0"]) for x in p]],
               fmt=".", capsize=2, lw=0.8, color=COLORS[i], label=f"{lb} (question-paired bootstrap 95%)")
  a.plot(steps, [100 * r["delta"]["mean"] for r in rows], color="k", lw=2, label=f"{n}-seed mean")
  if n > 1:
    a.fill_between([r["step"] for r in later], [100 * r["delta_seed_t95"][0] for r in later],
                   [100 * r["delta_seed_t95"][1] for r in later], color="k", alpha=0.12,
                   label=f"seed-level t 95% of the mean (df = {n - 1})")
  a.axhline(0, color="k", lw=0.8)
  a.set_xlabel("update")
  a.set_ylabel("v2 accuracy change vs own step 0, pp")
  a.set_title("Change from each seed's own step 0")
  a.legend(fontsize=7, loc="lower right")
  a.grid(alpha=0.3)
  a = ax[2]
  sp = [r["split"] for r in rows]
  sb = [r["split_box_status_differs"] for r in rows]
  a.bar(steps, sb, width=7, color="C3", alpha=0.75, label="box present in some runs, absent in others")
  a.bar(steps, [x - y for x, y in zip(sp, sb)], width=7, bottom=sb, color="C0", alpha=0.75,
        label="same box status, different correctness")
  for s, v in zip(steps, sp):
    a.annotate(str(v), (s, v), fontsize=7, ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
  a.text(0.98, 0.86, "Step 0: every run evaluates the same initial weights,\nso its split is launch-to-launch eval "
         "variation\nof the initial model. Later steps add training-seed\nvariation; the eval variation of trained "
         "weights\nwas not measured separately.", transform=a.transAxes, ha="right", va="top", fontsize=7,
         bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
  a.set_ylim(0, max(sp) * 1.8 + 1)
  a.set_xlabel("update")
  a.set_ylabel(f"questions (of {rows[0]['right_in_all'] + rows[0]['wrong_in_all'] + rows[0]['split']:,})")
  a.set_title(f"Questions whose correctness differs across the {n} runs")
  a.legend(fontsize=7, loc="upper left")
  a.grid(alpha=0.3, axis="y")
  fig.tight_layout()
  fig.savefig(os.path.join(out, "aggregate_eval.png"), dpi=130)
  plt.close(fig)

  # 2. training
  panels = (("acc_v2", 100.0, "Training batch v2 accuracy, % (10-update mean)", "roll"),
            ("mixed", 1.0, "Mixed-correctness prompt groups, of 128 (10-update mean)", "roll"),
            ("actor/entropy_loss", 1.0, "Token entropy of the update batch (10-update mean)", "roll"),
            ("response_length/mean", 1.0, "Mean response tokens, training (10-update mean)", "roll"),
            ("actor/grad_norm", 1.0, "Grad norm before clipping (per update; clip 1.0)", "log"),
            ("actor/rollout_corr/k3_kl", 1.0, "Trainer vs sampler k3_kl (per update)", "raw"))
  fig, ax = plt.subplots(2, 3, figsize=(18, 9))
  for (k, sc, title, kind), a in zip(panels, ax.flat):
    for i, lb in enumerate(labels):
      tr = train_series(reports[lb])
      ts = sorted(tr)
      v = [sc * tr[s][k] if tr[s].get(k) is not None else float("nan") for s in ts]
      if kind == "log":
        a.semilogy(ts, v, lw=0.7, color=COLORS[i], label=lb)
      elif kind == "roll":
        a.plot(ts, roll(v), lw=1.1, color=COLORS[i], label=lb)
      else:
        a.plot(ts, v, lw=0.7, color=COLORS[i], label=lb)
    if kind == "log":
      a.axhline(1.0, color="r", ls="--", lw=0.8, label="clip = 1.0")
    a.set_title(title, fontsize=10)
    a.set_xlabel("update")
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
  fig.suptitle("Training diagnostics per seed (same recipe and LR schedule: warmup 10, cosine to 0 at 250)", fontsize=11)
  fig.tight_layout()
  fig.savefig(os.path.join(out, "aggregate_training.png"), dpi=120)
  plt.close(fig)

  # 3. cost and timing
  fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
  for i, lb in enumerate(labels):
    r = reports[lb]
    gh = cumulative(r, "interval_s", GPUS / 3600.0)
    tok = cumulative(r, "response_length/mean", BATCH)
    pts = [(gh.get(row["step"]), tok.get(row["step"]), 100 * row["per_seed"][lb]["acc_v2"]) for row in rows]
    gap = r["timing"].get("paired_gap_s") or {}
    gap_txt = f"paired gap median {gap['median']:.1f} s/update" if gap.get("n") else "no paired gap"
    ok = [(x, y) for x, _, y in pts if x is not None]
    if ok:
      ax[0].plot(*zip(*ok), marker=".", lw=1, color=COLORS[i],
                 label=f"{lb} ({r['timing']['interval_source']}; {gap_txt})")
    ok = [(x / 1e6, y) for _, x, y in pts if x is not None]
    if ok:
      ax[1].plot(*zip(*ok), marker=".", lw=1, color=COLORS[i], label=lb)
  ax[0].set_xlabel(f"GPU-hours after the step-0 eval ({GPUS} GPUs x summed intervals)")
  ax[0].set_ylabel("held-out v2 accuracy, %")
  ax[0].set_title("Accuracy vs elapsed GPU-hours (startup excluded)")
  ax[0].legend(fontsize=7, loc="lower right")
  ax[1].set_xlabel("generated training tokens, millions (cumulative)")
  ax[1].set_title("Accuracy vs generated training tokens")
  ax[1].legend(fontsize=7, loc="lower right")
  a = ax[2]
  data, ticks, cols = [], [], []
  for i, lb in enumerate(labels):
    r = reports[lb]
    tr = train_series(r)
    win = window_steps(r)
    data.append([tr[s]["timing_s/step"] for s in win if tr[s].get("timing_s/step") is not None])
    data.append([tr[s]["interval_s"] for s in win if tr[s].get("interval_s") is not None])
    ticks += [f"{lb}\ninternal", f"{lb}\ninterval"]
    cols += [COLORS[i], COLORS[i]]
  bp = a.boxplot(data, whis=(10, 90), showfliers=False, patch_artist=True, medianprops={"color": "k"})
  for j, (patch, c) in enumerate(zip(bp["boxes"], cols)):
    patch.set_facecolor(c)
    patch.set_alpha(0.35 if j % 2 == 0 else 0.75)
  a.set_xticks(range(1, len(ticks) + 1))
  a.set_xticklabels(ticks, fontsize=8)
  a.set_ylabel("seconds per update")
  a.set_title("timing_s/step (internal) vs actual interval, steady window\n"
              "box: quartiles and median; whiskers: p10-p90", fontsize=10)
  for x in ax:
    x.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(os.path.join(out, "aggregate_cost.png"), dpi=130)
  plt.close(fig)


def pct(x, nd=2):
  return "—" if x is None else f"{100 * x:.{nd}f}"


def tables(labels, rows, train, timing, reports, qwen):
  n = len(labels)
  lines = [f"### Held-out v2 accuracy (%), {n} seeds", "",
           "| Step | " + " | ".join(labels) + " | Mean ± SD | Mean change vs step 0, pp | question bootstrap 95% | "
           f"seed-level t 95% (df {n - 1}) | Split questions (box status differs) | Identical outputs |",
           "|---:|" + "---:|" * n + "---|---:|---|---|---:|---:|"]
  for r in rows:
    sd = r["acc_v2"]["sd"]
    ci_q = r["delta_question_boot95"]
    ci_t = r["delta_seed_t95"]
    lines.append(
        f"| {r['step']} | " + " | ".join(pct(r["per_seed"][lb]["acc_v2"]) for lb in labels) +
        f" | {pct(r['acc_v2']['mean'])} ± {pct(sd)} | " +
        ("—" if r["step"] == 0 else f"{100 * r['delta']['mean']:+.2f}") + " | " +
        ("—" if r["step"] == 0 else f"[{100 * ci_q[0]:+.2f}, {100 * ci_q[1]:+.2f}]") + " | " +
        ("—" if r["step"] == 0 or ci_t is None else f"[{100 * ci_t[0]:+.2f}, {100 * ci_t[1]:+.2f}]") +
        f" | {r['split']} ({r['split_box_status_differs']}) | {r['identical_outputs_in_all']} |")
  last = rows[-1]
  lines += ["", f"### Endpoint (step {last['step']}) per seed", "",
            "| Seed | Step 0 | Step " + str(last["step"]) + " | Change, pp [question-paired 95%] | Wrong→right | "
            "Right→wrong | Boxed 0 → end | v1 acc 0 → end |", "|---|---:|---:|---|---:|---:|---|---|"]
  for lb in labels:
    p0, p = rows[0]["per_seed"][lb], last["per_seed"][lb]
    lines.append(f"| {lb} | {pct(p0['acc_v2'])} | {pct(p['acc_v2'])} | {100 * p['delta_vs_step0']:+.2f} "
                 f"[{100 * p['delta_boot95'][0]:+.2f}, {100 * p['delta_boot95'][1]:+.2f}] | {p['wrong_to_right']} | "
                 f"{p['right_to_wrong']} | {pct(p0['boxed'])} → {pct(p['boxed'])} | {pct(p0['acc_v1'])} → {pct(p['acc_v1'])} |")
  if qwen:
    q0 = [v[0] for v in qwen.values() if 0 in v]
    qe = [v[last["step"]] for v in qwen.values() if last["step"] in v]
    if q0 and qe:
      lines.append(f"| Qwen3-0.6B, {len(qe)}-seed mean (v1-trained, re-scored v2; context) | {statistics.mean(q0):.2f} | "
                   f"{statistics.mean(qe):.2f} | {statistics.mean(qe) - statistics.mean(q0):+.2f} (seeds: "
                   + " / ".join(f"{x:.2f}" for x in qe) + ") | | | | |")
  lines += ["", "### Training windows (means over the updates in each window)", "",
            "| Seed | Updates | Entropy | Response tokens | Train v2 acc | Train v1 acc | Boxed | Mixed / all-correct / "
            "all-wrong groups |", "|---|---|---:|---:|---:|---:|---:|---|"]
  for lb in labels:
    for w in train[lb]["windows"]:
      lines.append(f"| {lb} | {w['updates']} | {w['entropy']:.3f} | {w['tokens']:.0f} | {w['train_acc_v2']:.1f}% | "
                   f"{w['train_acc_v1']:.1f}% | {w['boxed']:.1f}% | {w['mixed']:.1f} / {w['all_correct']:.1f} / "
                   f"{w['all_wrong']:.1f} |")
  lines += ["", "### Numerics", "",
            "| Seed | Updates | Non-finite / missing | k3_kl range | Token IS max range | IS fraction above cap, max | "
            "ESS min | Grad norm median | Above clip 1.0 | Largest grad norms (update: value) | Train completions at cap |",
            "|---|---:|---|---|---|---:|---:|---:|---:|---|---:|"]
  for lb in labels:
    t = train[lb]
    lines.append(
        f"| {lb} | {t['updates']} | {t['nonfinite_values']} / {t['missing_values']} | "
        f"{t['k3_kl_range'][0]:.2e}–{t['k3_kl_range'][1]:.2e} | {t['is_max_range'][0]:.2f}–{t['is_max_range'][1]:.2f} | "
        f"{t['is_fraction_high_range'][1]:.1e} | {t['ess_range'][0]:.4f} | {t['grad_norm_median']:.2f} | "
        f"{t['grad_norm_above_clip']} | " + ", ".join(f"{s}: {v:.2f}" for s, v in t["grad_norm_top"]) +
        f" | {t['train_cap_hits']} |")

  def f3(d):
    return "—" if not d or not d.get("n") else f"{d['mean']:.1f} / {d['median']:.1f} / {d['p90']:.1f}"

  lines += ["", "### Timing and cost (steady window from each per-seed report)", "",
            "| Seed | Interval clock | Internal step, mean / median / p90 s | Actual interval, mean / median / p90 s | "
            "Paired gap, mean / median / p90 s | Launch to finish | GPU-hours | Generated train tokens |",
            "|---|---|---|---|---|---:|---:|---:|"]
  for lb in labels:
    t = timing[lb]
    lth = t["launch_to_finish_h"]
    lines.append(f"| {lb} | {t['interval_source']} | {f3(t['internal_step_s'])} | {f3(t['actual_interval_s'])} | "
                 f"{f3(t['paired_gap_s'])} | " + ("—" if lth is None else f"{lth:.2f} h") + " | " +
                 ("—" if t["gpu_hours"] is None else f"{t['gpu_hours']:.0f}") +
                 f" | {t['generated_train_tokens']:.3e} |")
  lines += ["", "### Checkpoints", "", "| Seed | Saved by the hook | Skipped | Directories |", "|---|---|---|---|"]
  for lb in labels:
    c = reports[lb]["checkpoints"]
    lines.append(f"| {lb} | {c.get('saved_by_hook')} | {c.get('skipped_by_hook')} | {', '.join(c.get('dirs', []))} |")
  return "\n".join(lines)


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--run", action="append", required=True, help="LABEL=EXP, once per seed, in display order")
  ap.add_argument("--out", required=True)
  ap.add_argument("--evidence", default=os.path.join(here, "gsm8k_8k"),
                  help="directory holding <label>/<label>_summary.json from gemma4_gsm8k_report.py")
  ap.add_argument("--logs", default="/workspace/meta-RL/logs")
  ap.add_argument("--qwen-rescore", help="rescore_v1_v2.json: Qwen 3-seed eval curves under v2, plotted as context")
  ap.add_argument("--no-plots", action="store_true")
  a = ap.parse_args()
  runs = [x.split("=", 1) for x in a.run]
  labels = [lb for lb, _ in runs]
  assert len(set(labels)) == len(labels), "duplicate labels"
  os.makedirs(a.out, exist_ok=True)
  reports, ev = {}, {}
  for lb, exp in runs:
    reports[lb] = load_json(os.path.join(a.evidence, lb, f"{lb}_summary.json"))
    assert reports[lb]["exp"] == exp, f"{lb}: report is for {reports[lb]['exp']}, not {exp}"
    ev[lb] = load_eval(os.path.join(a.logs, "gemma4_gsm8k", exp))
  unfinished = [lb for lb in labels if not reports[lb].get("finished") or reports[lb].get("driver_exit_code") != 0]
  steps, nq, rows = eval_aggregate(labels, ev, reports)
  train = train_aggregate(labels, reports)
  timing = timing_aggregate(labels, reports)
  qwen = qwen_curves(a.qwen_rescore)
  summary = {"runs": dict(runs), "unfinished_or_failed": unfinished, "eval_steps": steps, "questions": nq,
             "eval": rows, "train": train, "timing": timing,
             "checkpoints": {lb: reports[lb]["checkpoints"] for lb in labels},
             "statistics": {"question_bootstrap": "10,000 resamples of questions, jointly for all seeds, seed 0",
                            "seed_t95": f"mean +- t(0.975, {len(labels) - 1}) * SD / sqrt({len(labels)})"}}
  with open(os.path.join(a.out, "aggregate_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
  if not a.no_plots:
    plots(labels, steps, rows, reports, qwen, a.out)
  if unfinished:
    print(f"WARNING: unfinished or failed runs: {unfinished}; shared eval steps {steps[0]}-{steps[-1]}")
  print(tables(labels, rows, train, timing, reports, qwen))
  print("wrote", a.out)


if __name__ == "__main__":
  sys.exit(main())
