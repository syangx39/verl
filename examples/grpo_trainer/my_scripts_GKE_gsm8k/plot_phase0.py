#!/usr/bin/env python3
"""Phase 0 pre-check figure: six panels from one verl run.

Inputs
  --tb   <dir>   TensorBoard event directory of the run (TB_DIR from the launcher)
  --rollout <dir>  trainer.rollout_data_dir of the run (<step>.jsonl per step; optional)
  --out  <png>   output figure (default phase0.png)
  --ma   <int>   moving-average window in steps (default 5)

Panels
  1. train acc vs step        -- rollout dump: acc (task signal), the ACTUAL reward
                                 (dump mean score) and TB critic/score/mean as cross-check;
                                 no assumption about reward composition
  2. eval vs step             -- val-core/<ds>/reward/mean@1 and val-aux/<ds>/acc/mean@1
  3. zero-advantage groups    -- rollout dump, groups by uid: frac_zero_std of the ACTUAL
                                 reward, solve_all/solve_none on acc, non-acc-signal groups
                                 (format and/or length penalty), mean length penalty
  4. response length          -- response_length/mean and response_length/clip_ratio
  5. entropy / grad_norm      -- actor/entropy (log scale), actor/grad_norm; clipfrac in summary
  6. rollout-vs-trainer       -- probability-space MAE (training/rollout_probs_diff_*) and
                                 log-domain rollout_corr/kl, log_ppl_abs_diff if logged;
                                 plus actor/kl_loss = policy-vs-reference KL when use_kl_loss

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


def load_rollout(rollout_dir, require_uid=True, groups=256, group_size=8):
  expect_rows = groups * group_size
  """Per-step stats from trainer.rollout_data_dir (<step>.jsonl, one line per sample).

  Groups = the n=8 completions of one prompt. Grouped by the trainer's `uid`
  (verl uuid per prompt; needs patch_verl_dump_uid.py), else by the reward's
  `qid` (unique row id from build v4), else by input string. A step whose
  groups are not exactly 256 x 8 is skipped for group stats and listed.

  Three distinct "zero-advantage" quantities (Henry's point 4):
    solve_all / solve_none   : group's ANSWER correctness all 1 / all 0
    frac_zero_std_reward     : group's actual TRAINING reward has zero std ->
                               zero GRPO advantage (the KL term can still give gradient)
    format_only_groups       : acc identical inside the group but reward std > 0
                               -> the only signal in that group is format
  """
  if not rollout_dir or not os.path.isdir(rollout_dir):
    return None
  files = []
  for fn in glob.glob(os.path.join(rollout_dir, "*.jsonl")):
    try:
      files.append((int(os.path.splitext(os.path.basename(fn))[0]), fn))
    except ValueError:
      continue
  if not files:
    return None
  files.sort()
  keys = ["step", "n", "acc", "fmt", "score", "length_penalty", "chars", "n_groups", "bad_groups",
          "solve_all", "solve_none", "zero_std_reward", "nonacc_signal",
          "mv_timeout", "mv_exc", "mv_lenrej"]
  out = {k: [] for k in keys}
  anomalies = []
  for step, fn in files:
    groups_ = defaultdict(list)     # key -> list of (acc, score)
    n = acc = fmt = chars = tmo = exc = lrj = 0
    ssum = lpsum = 0.0
    with open(fn, encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        r = json.loads(line)
        sc = float(r.get("score", 0.0))
        a = float(r.get("acc", 1.0 if sc >= 1.0 else 0.0))
        fm = float(r.get("fmt", 0.0))
        if r.get("uid"):                                   # verl group uuid (fork patch)
          key = ("uid", r["uid"])
        elif require_uid:
          raise SystemExit(f"{fn}: rollout dump has no 'uid' field -- the verl fork is not patched "
                           f"(patch_verl_dump_uid.py). Pass --allow_no_uid only for pre-patch pilot dumps.")
        elif r.get("qid") is not None and r["qid"] >= 0:    # unique row id from build v4
          key = ("qid", int(r["qid"]))
        else:
          key = ("input", r.get("input", ""))
        groups_[key].append((a, sc))
        n += 1; acc += a; fmt += fm; chars += len(r.get("output", ""))
        ssum += sc; lpsum += float(r.get("length_penalty", 0.0))
        tmo += float(r.get("mv_timeout", 0)); exc += float(r.get("mv_exc", 0)); lrj += float(r.get("mv_lenrej", 0))
    if n == 0:
      continue
    sizes = [len(g) for g in groups_.values()]
    bad = sum(1 for sz in sizes if sz != group_size)
    if bad or len(groups_) != groups or n != expect_rows:
      # per review: a step whose groups are not exactly 256 x 8 is SKIPPED for the
      # group statistics (not silently included); it is listed in the summary.
      anomalies.append((step, n, len(groups_), bad))
      continue
    gs = list(groups_.values())
    def frac(pred):
      return sum(1 for g in gs if pred(g)) / len(gs) if gs else np.nan
    out["step"].append(step); out["n"].append(n)
    out["acc"].append(acc / n); out["fmt"].append(fmt / n); out["chars"].append(chars / n)
    out["score"].append(ssum / n); out["length_penalty"].append(lpsum / n)
    out["n_groups"].append(len(gs)); out["bad_groups"].append(bad)
    out["solve_all"].append(frac(lambda g: all(a >= 1.0 for a, _ in g)))
    out["solve_none"].append(frac(lambda g: all(a < 1.0 for a, _ in g)))
    out["zero_std_reward"].append(frac(lambda g: np.std([sc for _, sc in g]) == 0.0))
    # groups whose ONLY signal is non-accuracy (format bonus and/or length penalty)
    out["nonacc_signal"].append(frac(lambda g: len({a for a, _ in g}) == 1 and np.std([sc for _, sc in g]) > 0.0))
    out["mv_timeout"].append(tmo / n); out["mv_exc"].append(exc / n); out["mv_lenrej"].append(lrj / n)
  res = {k: np.array(v, dtype=float) for k, v in out.items()}
  res["anomalies"] = anomalies
  if len(res["step"]) == 0:
    print(f"!! rollout dump: all {len(anomalies)} steps skipped (groups not 256x8) -- no group statistics. "
          f"first anomalies: {anomalies[:5]}")
    return None
  kinds = {k[0] for k in groups_}
  res["grouped_by"] = "uid" if "uid" in kinds else ("qid" if "qid" in kinds else "input")
  return res


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
  ap.add_argument("--rollout", default=None)
  ap.add_argument("--groups", type=int, default=256, help="prompts per step")
  ap.add_argument("--group_size", type=int, default=8, help="completions per prompt")
  ap.add_argument("--allow_no_uid", action="store_true",
                  help="permit grouping by qid/input for dumps written before the uid patch (pilot runs only)")
  ap.add_argument("--out", default="phase0.png")
  ap.add_argument("--ma", type=int, default=5)
  ap.add_argument("--title", default="")
  args = ap.parse_args()

  tb = load_tb(args.tb)
  agg = load_rollout(args.rollout, require_uid=not args.allow_no_uid, groups=args.groups, group_size=args.group_size)
  if not tb:
    raise SystemExit(f"no scalar tags found under {args.tb}")

  fig, axes = plt.subplots(2, 3, figsize=(17, 9))
  axes = axes.ravel()
  summary = []

  # ---- 1. train acc + reward ---------------------------------------------
  # acc is the task signal; the ACTUAL training reward (acc + fmt_w*fmt + length_penalty)
  # is plotted separately from the dump and cross-checked against TB critic/score/mean.
  # No assumption about reward composition is made here.
  ax = axes[0]
  s_step, s_val = tb_get(tb, "critic/score/mean")
  if agg is not None:
    x = agg["step"]
    ax.plot(x, agg["acc"], color="#9ecae1", lw=1, label="acc per step (rollout dump)")
    ax.plot(x, moving_mean(agg["acc"], args.ma), color="#08519c", lw=2, label=f"acc {args.ma}-step mean")
    ax.plot(x, agg["score"], color="#e6550d", lw=1, ls="--", label="actual reward per step (dump mean score)")
    if len(s_step):
      ax.plot(s_step, s_val, color="#fd8d3c", lw=1, ls=":", label="TB critic/score/mean (cross-check)")
    lo, hi = first_last(agg["acc"])
    sl, ci = ols_slope(x, agg["acc"])
    summary.append(f"train acc   : first5={lo:.3f} last5={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}")
    lo, hi = first_last(agg["score"])
    summary.append(f"train reward: first5={lo:.3f} last5={hi:.3f}  (actual training reward)")
    lo, hi = first_last(agg["length_penalty"])
    summary.append(f"length_pen  : first5={lo:+.3f} last5={hi:+.3f}  (mean per sample, <= 0)")
    ax.set_title(f"train acc vs actual reward ({args.groups * args.group_size} samples/step)")
  elif len(s_step):
    ax.plot(s_step, s_val, color="#9ecae1", lw=1, label="critic/score/mean")
    ax.plot(s_step, moving_mean(s_val, args.ma), color="#08519c", lw=2, label=f"{args.ma}-step mean")
    lo, hi = first_last(s_val)
    sl, ci = ols_slope(s_step, s_val)
    summary.append(f"train score : first5={lo:.3f} last5={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}")
    ax.set_title("train reward (no rollout dump found)")
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 2. eval -----------------------------------------------------------
  ax = axes[1]
  # verl files the "core" metric under val-core/ and the rest under val-aux/;
  # which key is core depends on the reward's return keys (acc if present,
  # else reward). Look under both prefixes so the panel works either way.
  plotted = False
  acc_series, rew_series, fmt_series = {}, {}, {}
  for pre in ("val-core/", "val-aux/"):
    acc_series.update(tb_find(tb, pre, "/acc/mean@1"))
    rew_series.update(tb_find(tb, pre, "/reward/mean@1"))
    fmt_series.update(tb_find(tb, pre, "/fmt/mean@1"))
  for ds, (st, vals) in sorted(acc_series.items()):
    if ds.startswith("num_turns"):
      continue
    ax.plot(st, vals, marker="o", lw=2, label=f"{ds} acc (greedy)")
    lo, hi = first_last(vals, 1)
    sl, ci = ols_slope(st, vals)
    summary.append(f"eval {ds:<8}: first={lo:.3f} last={hi:.3f}  slope/100steps={sl:+.4f} ±{ci:.4f}  (n_evals={len(st)})")
    plotted = True
  for ds, (st, vals) in sorted(rew_series.items()):
    ax.plot(st, vals, marker="s", lw=1, ls="--", alpha=0.7, label=f"{ds} reward (actual score)")
    plotted = True
  for ds, (st, vals) in sorted(fmt_series.items()):
    ax.plot(st, vals, marker="^", lw=1, ls=":", alpha=0.7, label=f"{ds} fmt rate")
    plotted = True
  if not plotted:
    ax.text(0.5, 0.5, "no val-* tags\n(test_freq=-1?)", ha="center", va="center", transform=ax.transAxes)
  ax.set_title("eval vs step (greedy, n=1)")
  ax.set_ylim(0, 1)
  ax.set_xlabel("training step")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # ---- 3. zero-advantage groups (three distinct quantities) ------------
  ax = axes[2]
  if agg is not None and np.isfinite(agg["zero_std_reward"]).any():
    ax.plot(agg["step"], agg["zero_std_reward"], color="k", lw=1.8, label="frac_zero_std (actual reward) = zero GRPO advantage")
    ax.plot(agg["step"], agg["solve_none"], color="#de2d26", lw=1.2, label="solve_none (acc all 0)")
    ax.plot(agg["step"], agg["solve_all"], color="#31a354", lw=1.2, label="solve_all (acc all 1)")
    ax.plot(agg["step"], agg["nonacc_signal"], color="#e6550d", lw=1.5, ls="--",
            label="non-acc-signal groups (acc same, reward differs: format/length)")
    ax.plot(agg["step"], agg["fmt"], color="#756bb1", lw=1, ls=":", label="train fmt rate")
    ax.plot(agg["step"], -agg["length_penalty"], color="#636363", lw=1, ls="-.", label="-mean length_penalty")
    for name in ("zero_std_reward", "nonacc_signal", "fmt"):
      lo, hi = first_last(agg[name])
      summary.append(f"{name:<14}: first5={lo:.3f} last5={hi:.3f}")
    ax.set_ylim(0, 1)
    ax.set_title(f"zero-advantage groups (grouped by {agg['grouped_by']})")
  else:
    ax.text(0.5, 0.5, "no rollout dump\n(trainer.rollout_data_dir unset?)",
            ha="center", va="center", transform=ax.transAxes)
  ax.set_xlabel("training step")
  ax.legend(fontsize=7)
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

  # ---- 5. entropy / grad norm (clipfrac goes to the summary) ------------
  # Entropy was the earliest collapse signal in the v4 pilot (rising from step ~30,
  # exploding past step ~100), so it gets the panel; pg_clipfrac is 0 by design at
  # mu=1 and only informative in multi-update ablations -> printed, not plotted.
  ax = axes[4]
  st, v = tb_get(tb, "actor/entropy")
  if len(st):
    ax.plot(st, v, color="#08519c", lw=1.5, label="actor/entropy")
    ax.set_ylabel("entropy (nats/token)")
    ax.set_yscale("log")
    lo, hi = first_last(v)
    summary.append(f"entropy      : first5={lo:.3f} last5={hi:.3f}  (x{hi / max(lo, 1e-9):.1f})  [trainer entropy at the rollout temperature; compare across runs only at equal T]")
    if hi > 3 * lo:
      summary.append("  !! entropy grew > 3x from the start -- collapse signature")
  st2, v2 = tb_get(tb, "actor/grad_norm")
  if len(st2):
    ax2 = ax.twinx()
    ax2.plot(st2, v2, color="#e6550d", lw=1, alpha=0.8, label="actor/grad_norm")
    ax2.set_ylabel("grad norm")
    ax2.legend(loc="upper right", fontsize=8)
    lo, hi = first_last(v2)
    summary.append(f"grad_norm    : first5={lo:.3f} last5={hi:.3f}  max={np.nanmax(v2):.3f}")
  st3, v3 = tb_get(tb, "actor/pg_clipfrac")
  if len(st3):
    lo, hi = first_last(v3)
    summary.append(f"pg_clipfrac  : first5={lo:.3f} last5={hi:.3f}  (0 by design at mu=1)")
  ax.set_title("policy entropy / gradient norm")
  ax.set_xlabel("training step")
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(alpha=0.3, which="both")

  # ---- 6. rollout vs trainer mismatch ---------------------------------
  # verl's training/rollout_probs_diff_* is mean |exp(logp_trainer) - exp(logp_rollout)|
  # -- a PROBABILITY-space MAE, not nats. The fork also logs log-domain
  # quantities (rollout_corr/kl, rollout_corr/log_ppl_abs_diff); plot both,
  # labelled honestly. Do not compare the probability MAE with nats numbers
  # from other reports.
  ax = axes[5]
  got = False
  # Two different KLs live here, deliberately labelled apart:
  #   rollout_corr/kl  = TRAINER vs ROLLOUT engine on the same weights (numerics / sampling)
  #   actor/kl_loss    = CURRENT POLICY vs REFERENCE model (k3 / low_var_kl), the term
  #                      that use_kl_loss adds to the actor loss -- policy drift
  for tag, c, lab in (("training/rollout_probs_diff_mean", "#de2d26", "mean |p_trainer - p_rollout|  (probability MAE)"),
                      ("training/rollout_probs_diff_max", "#fd8d3c", "max |p_trainer - p_rollout|"),
                      ("rollout_corr/kl", "#08519c", "rollout_corr/kl  = trainer vs rollout (log domain)"),
                      ("rollout_corr/log_ppl_abs_diff", "#6baed6", "rollout_corr/log_ppl_abs_diff  (log domain)"),
                      ("actor/kl_loss", "#31a354", "actor/kl_loss  = policy vs REFERENCE (k3), drift from init")):
    st, v = tb_get(tb, tag)
    if len(st):
      ax.plot(st, v, color=c, lw=1.5, label=lab)
      got = True
      lo, hi = first_last(v)
      summary.append(f"{tag:<34}: first5={lo:.4f} last5={hi:.4f}")
  if not got:
    ax.text(0.5, 0.5, "no mismatch tags\n(calculate_log_probs=False?)",
            ha="center", va="center", transform=ax.transAxes)
  ax.set_yscale("log")
  ax.set_title("trainer-vs-rollout mismatch  |  policy-vs-reference KL (actor/kl_loss)")
  ax.set_xlabel("training step")
  ax.legend(fontsize=7)
  ax.grid(alpha=0.3, which="both")

  fig.suptitle(args.title or os.path.basename(os.path.normpath(args.tb)), fontsize=13)
  fig.tight_layout()
  fig.savefig(args.out, dpi=130)

  print(f"saved {args.out}")
  if agg is not None:
    print(f"rollout dump: {len(agg['step'])} steps, grouped by {agg['grouped_by']}, samples/step "
          f"min={int(agg['n'].min())} max={int(agg['n'].max())} (expect {args.groups * args.group_size}), groups/step ~{int(np.nanmedian(agg['n_groups']))} (expect {args.groups})")
    if agg["anomalies"]:
      print(f"  !! {len(agg['anomalies'])} steps SKIPPED (groups not {args.groups}x{args.group_size}) (step, n, n_groups, bad_groups): {agg['anomalies'][:5]} ...")
    if agg["grouped_by"] != "uid":
      print("  !! grouped by", agg["grouped_by"], "-- apply patch_verl_dump_uid.py so dumps carry the trainer uid")
    print(f"  math_verify flags (rate over all samples): timeout={agg['mv_timeout'].mean():.5f} "
          f"exc={agg['mv_exc'].mean():.5f} lenrej={agg['mv_lenrej'].mean():.5f}")
  print("\n".join(summary))


if __name__ == "__main__":
  main()