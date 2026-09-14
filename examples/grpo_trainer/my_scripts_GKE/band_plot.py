#!/usr/bin/env python3
"""GB200 reference band across seeds (Level 3).

Reads N TensorBoard dirs (one per seed, same recipe), and for each eval metric
computes, at every eval step, min / max / mean across seeds. Produces:
  <out>.png   two panels (GSM8K, OMI2 val): per-seed curves + min-max band + mean
  <out>.json  the band per checkpoint (for the parity package / TPU comparison)
  stdout      per-checkpoint table and band-width statistics

The band is DESCRIPTIVE (a range over 3 seeds, not a confidence interval); the
parity criterion is stated separately in the rulebook (e.g. TPU inside
[min - delta, max + delta] at every checkpoint, with delta from the seed-to-seed
spread below).

Usage:
  python3 band_plot.py --tb <tb_dir_seed1> <tb_dir_seed2> <tb_dir_seed3> --out <path_without_ext>
  optional: --extra <tb_dir> ... (e.g. a TPU run or an ablation) drawn on top for comparison
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

METRICS = [("val-core/gsm8k/acc/mean@1", "GSM8K test acc (greedy)"),
           ("val-core/omi2_val1k/acc/mean@1", "OMI2 held-out val acc (greedy)")]
AUX = ["actor/entropy", "response_length/mean", "actor/grad_norm"]


def load(tb_dir):
  """Tags are keyed with '@' replaced by '_' (some writers sanitize '@'); lookups do the same."""
  a = ea.EventAccumulator(tb_dir, size_guidance={ea.SCALARS: 0})
  a.Reload()
  tags = a.Tags().get("scalars", [])
  return {t.replace("@", "_"): {e.step: e.value for e in a.Scalars(t)} for t in tags}


def key(tag):
  return tag.replace("@", "_")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tb", nargs="+", required=True, help="TB dirs, one per seed")
  ap.add_argument("--labels", nargs="*", default=None)
  ap.add_argument("--extra", nargs="*", default=[], help="TB dirs to overlay (not part of the band)")
  ap.add_argument("--extra_labels", nargs="*", default=None)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  runs = [load(p) for p in args.tb]
  labels = args.labels or [f"seed{i + 1}" for i in range(len(runs))]
  extras = [load(p) for p in args.extra]
  extra_labels = args.extra_labels or [os.path.basename(p.rstrip("/"))[:40] for p in args.extra]

  band = {}
  fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 5))
  for ax, (tag, title) in zip(axes, METRICS):
    k = key(tag)
    steps = sorted(set.intersection(*[set(r.get(k, {})) for r in runs]))
    if not steps:
      ax.set_title(f"{title}: no data")
      continue
    M = np.array([[r[k][s] for s in steps] for r in runs])         # seeds x steps
    lo, hi, mean = M.min(0), M.max(0), M.mean(0)
    band[tag] = [{"step": int(s), "min": float(l), "max": float(h), "mean": float(m),
                  "per_seed": [float(v) for v in M[:, i]]}
                 for i, (s, l, h, m) in enumerate(zip(steps, lo, hi, mean))]
    ax.fill_between(steps, lo, hi, color="#9ecae1", alpha=0.5, label=f"min-max band ({len(runs)} seeds)")
    for i, lab in enumerate(labels):
      ax.plot(steps, M[i], lw=1, alpha=0.8, label=lab)
    ax.plot(steps, mean, color="k", lw=2, label="mean")
    for e, lab in zip(extras, extra_labels):
      es = sorted(s for s in e.get(k, {}) if s in set(steps))
      if es:
        ax.plot(es, [e[k][s] for s in es], color="#d62728", lw=2, ls="--", marker="o", ms=4, label=lab)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    width = hi - lo
    print(f"\n{title}")
    print(f"  {'step':>5} {'min':>7} {'max':>7} {'mean':>7} {'width':>7}   per seed")
    for row in band[tag]:
      print(f"  {row['step']:>5} {row['min']:7.3f} {row['max']:7.3f} {row['mean']:7.3f} {row['max'] - row['min']:7.3f}   "
            + " ".join(f"{v:.3f}" for v in row["per_seed"]))
    print(f"  band width: median {np.median(width):.3f}, p90 {np.percentile(width, 90):.3f}, max {width.max():.3f}  "
          f"| final mean {mean[-1]:.3f} [{lo[-1]:.3f}, {hi[-1]:.3f}]  | gain 0->{steps[-1]} per seed: "
          + ", ".join(f"{M[i, -1] - M[i, 0]:+.3f}" for i in range(len(runs))))
    for e, lab in zip(extras, extra_labels):
      es = [s for s in steps if s in e.get(k, {})]
      if es:
        inside = sum(1 for s in es if lo[steps.index(s)] <= e[k][s] <= hi[steps.index(s)])
        print(f"  overlay {lab}: inside band at {inside}/{len(es)} checkpoints")

  # aux: seed-to-seed spread of training diagnostics (for the rulebook)
  print("\ntraining diagnostics, seed-to-seed spread (mean over steps of max-min across seeds):")
  for tag in AUX:
    k = key(tag)
    steps = sorted(set.intersection(*[set(r.get(k, {})) for r in runs]))
    if steps:
      M = np.array([[r[k][s] for s in steps] for r in runs])
      print(f"  {tag:28s} mean {M.mean():9.4f}  spread {np.mean(M.max(0) - M.min(0)):9.4f}")

  fig.suptitle(f"GB200 reference band: {', '.join(labels)}")
  fig.tight_layout()
  fig.savefig(args.out + ".png", dpi=120)
  with open(args.out + ".json", "w") as f:
    json.dump({"seeds": args.tb, "labels": labels, "band": band}, f, indent=1)
  print(f"\nsaved {args.out}.png and {args.out}.json")


if __name__ == "__main__":
  main()
