#!/usr/bin/env python3
"""Phase 0 pre-check figure: six panels from one verl run.

Inputs
  --tb   <dir>   TensorBoard event directory of the run (TB_DIR from the launcher)
  --agg  <dir>   ${REWARD_DUMP_DIR}/${EXPERIMENT_NAME}  (agg_pid*.jsonl; optional)
  --out  <png>   output figure (default phase0.png)
  --ma   <int>   moving-average window in steps (default 5)

Panels
  1. train acc vs step        -- agg jsonl (acc), per-window + moving mean;
                                 critic/score/mean from TB overlaid as a check
                                 (score = acc + 0.1*fmt, so score - 0.1*fmt should sit on acc)
  2. eval vs step             -- val-core/<ds>/reward/mean@1 and val-aux/<ds>/acc/mean@1
  3. solve_all / solve_none   -- agg jsonl (fraction of GRPO groups all-correct / all-wrong)
                                 falls back to fmt rate if groups were not identifiable
  4. response length          -- response_length/mean and response_length/clip_ratio
  5. pg_clipfrac / grad_norm  -- actor/pg_clipfrac, actor/grad_norm
  6. rollout-vs-trainer logp  -- training/rollout_probs_diff_mean / _max

Summary printed to stdout: first/last 5-step means and an OLS slope per 100 steps
with a naive 95% CI for train acc and each eval series.

Run it wherever the two directories are readable (head pod, or a laptop after
`gsutil -m rsync -r gs://.../tensorboard_log/<project>/<exp> ./tb`).
Needs: tensorboard, matplotlib, numpy.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
  raise SystemExit("pip install matplotlib") from e

from tensorboard.backend.event_processing import event_accumulator as ea


# ------------------------------------------------------------------ loaders
def load_tb(tb_dir):
  """Return {tag: (steps ndarray, values ndarray)} for every scalar tag."""
  acc = ea.EventAccumulator(tb_dir, size_guidance={ea.SCALARS: 0})
  acc.Reload()
  out = {}
  for tag in acc.Tags().get("scalars", []):
    ev = acc.Scalars(tag)
    out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
  return out


def load_agg(agg_dir):
  """Merge agg_pid*.jsonl by window index -> dict of per-window arrays."""
  if not agg_dir or not os.path.isdir(agg_dir):
    return None
  files = glob.glob(os.path.join(agg_dir, "agg_pid*.jsonl"))
  if not files:
    return None
  by_win = defaultdict(lambda: {"n": 0, "acc": 0.0, "fmt": 0.0, "chars": 0.0,
                                "g": 0, "sa": 0.0, "sn": 0.0})
  for fn in files:
    with open(fn, encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        r = json.loads(line)
        w = by_win[r["window"]]
        n = r["n"]
        w["n"] += n
        w["acc"] += r["acc"] * n
        w["fmt"] += r["fmt"] * n
        w["chars"] += r["mean_chars"] * n
        if r.get("n_groups") and r.get("solve_all") is not None:
          g = r["n_groups"]
          w["g"] += g
          w["sa"] += r["solve_all"] * g
          w["sn"] += r["solve_none"] * g
  wins = sorted(by_win)
  n = np.array([by_win[w]["n"] for w in wins], dtype=float)
  out = {
      "window": np.array(wins),
      "n": n,
      "acc": np.array([by_win[w]["acc"] for w in wins]) / n,
      "fmt": np.array([by_win[w]["fmt"] for w in wins]) / n,
      "chars": np.array([by_win[w]["chars"] for w in wins]) / n,
  }
  g = np.array([by_win[w]["g"] for w in wins], dtype=float)
  if g.sum() > 0:
    with np.errstate(invalid="ignore", divide="ignore"):
      out["solve_all"] = np.array([by_win[w]["sa"] for w in wins]) / g
      out["solve_none"] = np.array([by_win[w]["sn"] for w in wins]) / g
  out["n_files"] = len(files)
  return out


# ------------------------------------------------------------------ helpers
def moving_mean(y, k):
  if len(y) < k or k <= 1:
    return y
  c = np.cumsum(np.insert(y, 0, 0.0))
  mm = (c[k:] - c[:-k]) / k
  return np.concatenate([np.full(k - 1, np.nan), mm])


def ols_slope(x, y):
  """Slope per 100 steps with naive 95% CI (no HAC correction)."""
  x = np.asarray(x, float)
  y = np.asarray(y, float)
  m = np.isfinite(x) & np.isfinite(y)
  x, y = x[m], y[m]
  if len(x) < 3:
    return np.nan, np.nan
  a, b = np.polyfit(x, y, 1)
  resid = y - (a * x + b)
  se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
  return 100 * a, 100 * 1.96 * se


def first_last(y, k=5):
  y = np.asarray(y, float)
  y = y[np.isfinite(y)]
  if len(y) == 0:
    return np.nan, np.nan
  k = min(k, len(y))
  return y[:k].mean(), y[-k:].mean()


def tb_get(tb, tag):
  return tb.get(tag, (np.array([]), np.array([])))


def tb_find(tb, prefix, suffix):
  """All tags of the form <prefix><ds><suffix>, returned as {ds: (steps, vals)}.

  TensorBoard writers sanitize '@' to '_' in tag names, so a verl tag written as
  val-aux/gsm8k/acc/mean@1 shows up as val-aux/gsm8k/acc/mean_1. Accept both.
  """
  out = {}
  variants = {suffix, suffix.replace("@", "_")}
  for tag in tb:
    if not tag.startswith(prefix):
      continue
    for suf in variants:
      if tag.endswith(suf):
        out[tag[len(prefix):-len(suf)]] = tb[tag]
        break
  return out


# ------------------------------------------------------------------ main
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tb", required=True)
  ap.add_argument("--agg", default=None)
  ap.add_argument("--out", default="phase0.png")
  ap.add_argument("--ma", type=int, default=5)
  ap.add_argument("--title", default="")
  args = ap.parse_args()

  tb = load_tb(args.tb)
  agg = load_agg(args.agg)
  if not tb:
    raise SystemExit(f"no scalar tags found under {args.tb}")

  fig, axes = plt.subplots(2, 3, figsize=(17, 9))
  axes = axes.ravel()
  summary = []

  # ---- 1. train acc ------------------------------------------------------
  ax = axes[0]
  s_step, s_val = tb_get(tb, "critic/score/mean")
  if agg is not None:
    x = agg["window"]
    y = agg["acc"]
    ax.plot(x, y, color="#9ecae1", lw=1, label="acc per step (agg)")
    ax.plot(x, moving_mean(y, args.ma), color="#08519c", lw=2, label=f"acc {args.ma}-step mean")
    if len(s_step):
      # score - 0.1*fmt should coincide with acc if windows align with steps
      fmt_i = np.interp(s_step, x, agg["fmt"]) if len(x) else 0.0
      ax.plot(s_step, s_val - 0.1 * fmt_i, color="#e6550d", lw=1, ls="--",
              label="TB score - 0.1*fmt (cross-check)")
    lo, hi = first_last(y)
    sl, ci = ols_slope(x, y)
    summary.append(f"train acc   : first5={lo:.3f} last5={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}")
    ax.set_title("train acc (fraction correct, 2048 samples/step)")
  elif len(s_step):
    ax.plot(s_step, s_val, color="#9ecae1", lw=1, label="critic/score/mean")
    ax.plot(s_step, moving_mean(s_val, args.ma), color="#08519c", lw=2, label=f"{args.ma}-step mean")
    lo, hi = first_last(s_val)
    sl, ci = ols_slope(s_step, s_val)
    summary.append(f"train score : first5={lo:.3f} last5={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}")
    ax.set_title("train score = acc + 0.1*fmt (no agg jsonl found)")
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 2. eval -----------------------------------------------------------
  ax = axes[1]
  plotted = False
  for ds, (st, vals) in sorted(tb_find(tb, "val-aux/", "/acc/mean@1").items()):
    ax.plot(st, vals, marker="o", lw=2, label=f"{ds} acc (greedy)")
    lo, hi = first_last(vals, 1)
    sl, ci = ols_slope(st, vals)
    summary.append(f"eval {ds:<8}: first={lo:.3f} last={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}  (n_evals={len(st)})")
    plotted = True
  for ds, (st, vals) in sorted(tb_find(tb, "val-core/", "/reward/mean@1").items()):
    ax.plot(st, vals, marker="s", lw=1, ls="--", alpha=0.7, label=f"{ds} reward (=acc+0.1fmt)")
    plotted = True
  if not plotted:
    ax.text(0.5, 0.5, "no val-* tags\n(test_freq=-1?)", ha="center", va="center", transform=ax.transAxes)
  ax.set_title("eval vs step (greedy, n=1)")
  ax.set_ylim(0, 1)
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 3. solve_all / solve_none ----------------------------------------
  ax = axes[2]
  if agg is not None and "solve_all" in agg:
    ax.plot(agg["window"], agg["solve_all"], color="#31a354", lw=1.5, label="solve_all (group all-correct)")
    ax.plot(agg["window"], agg["solve_none"], color="#de2d26", lw=1.5, label="solve_none (group all-wrong)")
    zero_std = agg["solve_all"] + agg["solve_none"]
    ax.plot(agg["window"], zero_std, color="k", lw=1, ls=":", label="sum = frac_zero_std")
    lo, hi = first_last(zero_std)
    summary.append(f"frac_zero_std: first5={lo:.3f} last5={hi:.3f}")
    ax.set_ylim(0, 1)
    ax.set_title("GRPO groups with zero advantage")
  elif agg is not None:
    ax.plot(agg["window"], agg["fmt"], color="#756bb1", lw=1.5, label="fmt rate")
    ax.set_ylim(0, 1)
    ax.set_title("fmt rate (no group ids in extra_info -> no solve_* stats)")
  else:
    ax.text(0.5, 0.5, "no agg jsonl", ha="center", va="center", transform=ax.transAxes)
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 4. response length ----------------------------------------------
  ax = axes[3]
  st, v = tb_get(tb, "response_length/mean")
  if len(st):
    ax.plot(st, v, color="#08519c", lw=1.5, label="response_length/mean")
    ax.set_ylabel("tokens")
  st2, v2 = tb_get(tb, "response_length/clip_ratio")
  if len(st2):
    ax2 = ax.twinx()
    ax2.plot(st2, v2, color="#e6550d", lw=1.5, label="clip_ratio (hit 8192 cap)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("cap-hit fraction")
    ax2.legend(loc="upper right", fontsize=8)
    lo, hi = first_last(v2)
    summary.append(f"cap-hit      : first5={lo:.3f} last5={hi:.3f}")
  ax.set_title("response length / cap hits")
  ax.set_xlabel("training step")
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 5. clipfrac / grad norm ------------------------------------------
  ax = axes[4]
  st, v = tb_get(tb, "actor/pg_clipfrac")
  if len(st):
    ax.plot(st, v, color="#08519c", lw=1.5, label="actor/pg_clipfrac")
    ax.axhspan(0.05, 0.15, color="green", alpha=0.08, label="healthy 5-15%")
    ax.set_ylabel("clipfrac")
    lo, hi = first_last(v)
    summary.append(f"pg_clipfrac  : first5={lo:.3f} last5={hi:.3f}")
  st2, v2 = tb_get(tb, "actor/grad_norm")
  if len(st2):
    ax2 = ax.twinx()
    ax2.plot(st2, v2, color="#e6550d", lw=1, alpha=0.8, label="actor/grad_norm")
    ax2.set_ylabel("grad norm")
    ax2.legend(loc="upper right", fontsize=8)
  ax.set_title("PPO clip fraction / gradient norm")
  ax.set_xlabel("training step")
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 6. rollout vs trainer logp --------------------------------------
  ax = axes[5]
  got = False
  for tag, c in (("training/rollout_probs_diff_mean", "#de2d26"),
                 ("training/rollout_probs_diff_max", "#fd8d3c")):
    st, v = tb_get(tb, tag)
    if len(st):
      ax.plot(st, v, color=c, lw=1.5, label=tag.split("/")[-1])
      got = True
      if tag.endswith("mean"):
        lo, hi = first_last(v)
        summary.append(f"rollout/trainer |dlogp| mean: first5={lo:.4f} last5={hi:.4f}")
  if not got:
    ax.text(0.5, 0.5, "no training/rollout_probs_diff_*\n(calculate_log_probs=False?)",
            ha="center", va="center", transform=ax.transAxes)
  ax.set_yscale("log")
  ax.set_title("trainer vs rollout log-prob mismatch (GPU floor)")
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3, which="both")

  fig.suptitle(args.title or os.path.basename(os.path.normpath(args.tb)), fontsize=13)
  fig.tight_layout()
  fig.savefig(args.out, dpi=130)

  print(f"saved {args.out}")
  if agg is not None:
    print(f"agg: {agg['n_files']} pid files, {len(agg['window'])} windows, "
          f"total samples={int(agg['n'].sum())} (expect steps*2048)")
  print("\n".join(summary))


if __name__ == "__main__":
  main()