#!/usr/bin/env python3
"""Step-time breakdown for a verl V1 separate_async run (and optionally colocated runs) from TensorBoard.

Panels: (1) per-step timing_s/step with eval/checkpoint steps marked, plus a steady-state median line;
        (2) stacked breakdown per step: update_actor, update_weights, adv, old_log_prob, gen (wait for the sampler);
        (3) cumulative wall-clock vs step for every --tb run, with the time each run first reaches --target accuracy.
Usage: plot_step_time.py --tb <run_tb> [<other_tb> ...] --labels A B ... --out fig.png [--target 0.80] [--exclude_every 20 50]
"""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load(tb):
    a = EventAccumulator(tb, size_guidance={"scalars": 0}); a.Reload()
    tags = set(a.Tags()["scalars"])
    def S(t):
        if t not in tags: return {}
        d = {}
        for e in a.Scalars(t): d[e.step] = (e.wall_time, e.value)      # last write wins
        return {k: v[1] for k, v in d.items()}, {k: v[0] for k, v in d.items()}
    return S, tags

ap = argparse.ArgumentParser()
ap.add_argument("--tb", nargs="+", required=True); ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--target", type=float, default=0.80)
ap.add_argument("--exclude_every", nargs="*", type=int, default=[20, 50], help="eval/ckpt periods excluded from the steady-state stat")
ap.add_argument("--val_tag", default="val-core/gsm8k_boxed_test/acc/mean@1")
args = ap.parse_args(); assert len(args.tb) == len(args.labels)

fig, ax = plt.subplots(1, 3, figsize=(21, 5))
parts = [("timing_s/update_actor", "update_actor"), ("timing_s/update_weights", "update_weights (sync)"), ("timing_s/adv", "adv"),
         ("timing_s/old_log_prob", "old_log_prob"), ("timing_s/gen", "gen = wait for sampler")]
summary = []
for i, (tb, lab) in enumerate(zip(args.tb, args.labels)):
    S, tags = load(tb)
    st, wt = S("timing_s/step") if "timing_s/step" in tags else ({}, {})
    if not st: print(f"{lab}: no timing_s/step"); continue
    steps = sorted(st); v = np.array([st[s] for s in steps])
    steady = [s for s in steps if s >= 20 and all(s % p for p in args.exclude_every)]
    med = float(np.median([st[s] for s in steady])); p90 = float(np.percentile([st[s] for s in steady], 90))
    c = f"C{i}"
    if i == 0:   # panel 1 + 2 for the first run only
        excl = [s for s in steps if s not in steady and s >= 20]
        ax[0].plot(steps, v, color=c, lw=1, label=f"{lab}: timing_s/step")
        ax[0].scatter(excl, [st[s] for s in excl], color="#d62728", s=14, zorder=3, label="eval / checkpoint steps")
        ax[0].axhline(med, color=c, ls="--", lw=1, label=f"steady median {med:.2f}s (p90 {p90:.2f}s, n={len(steady)})")
        ax[0].set_ylim(0, min(v.max(), med * 4)); ax[0].set_title("step time"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("s"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        bottom = np.zeros(len(steps))
        for tag, name in parts:
            d = S(tag)[0] if tag in tags else {}
            y = np.array([d.get(s, 0.0) for s in steps]); ax[1].bar(steps, y, bottom=bottom, width=1.0, label=f"{name} (median {np.median([d.get(s,0.0) for s in steady]):.2f}s)"); bottom += y
        ax[1].set_ylim(0, med * 2.5); ax[1].set_title("step time breakdown (stacked; eval/ckpt not shown)"); ax[1].set_xlabel("training step"); ax[1].set_ylabel("s"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # panel 3: cumulative wall clock from the first step's wall_time, + time to target
    t0 = wt[steps[0]] - st[steps[0]]; cum = [(wt[s] - t0) / 60 for s in steps]
    ax[2].plot(steps, cum, color=c, lw=1.8, label=f"{lab}: {cum[-1]:.0f} min to step {steps[-1]}")
    val, vwt = S(args.val_tag) if args.val_tag in tags else ({}, {})
    hit = [s for s in sorted(val) if val[s] >= args.target]
    if hit:
        s = hit[0]; t = (vwt[s] - t0) / 60; ax[2].scatter([s], [t], color=c, s=50, zorder=3); ax[2].annotate(f"{args.target:.2f} @ step {s}, {t:.0f} min", (s, t), textcoords="offset points", xytext=(6, -12), fontsize=8, color=c)
    summary.append((lab, med, p90, cum[-1], hit[0] if hit else None))
ax[2].set_title(f"cumulative wall clock (first eval >= {args.target:.2f} marked)"); ax[2].set_xlabel("training step"); ax[2].set_ylabel("minutes"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=130)
for lab, med, p90, tot, hit in summary: print(f"{lab:36s} steady step median {med:.2f}s p90 {p90:.2f}s | total {tot:.1f} min | first eval >= {args.target}: step {hit}")
print("saved", args.out)