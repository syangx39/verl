
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
ap.add_argument("--steady_from", type=int, default=20, help="first step of the steady-state window (use 1 for short runs)")
ap.add_argument("--target_rule", choices=["first", "sustained2"], default="sustained2",
                help="time-to-target: first eval >= target, or (rulebook convention) the second of two consecutive evals >= target, i.e. the confirming evaluation")
ap.add_argument("--breakdown_idx", nargs=2, type=int, default=[0, 1], help="indices (into --tb) of the two runs whose stacked breakdown is drawn")
ap.add_argument("--wait_idx", nargs="*", type=int, default=[0], help="indices of runs whose timing_s/gen is the trainer's WAIT for the sampler (disaggregated); others: generation")
ap.add_argument("--start_epoch", nargs="*", default=[], help="per run: file with the launch epoch (run_dir/start_epoch.txt) -> end-to-end wall clock incl. startup and the step-0 eval; omitted runs start the clock at their first training step")
args = ap.parse_args(); assert len(args.tb) == len(args.labels)

fig, ax = plt.subplots(1, 4, figsize=(27, 5))
parts = [("timing_s/update_actor", "update_actor"), ("timing_s/update_weights", "update_weights (sync)"), ("timing_s/adv", "adv"),
         ("timing_s/old_log_prob", "old_log_prob"), ("timing_s/gen", "gen = wait for sampler")]
summary = []
for i, (tb, lab) in enumerate(zip(args.tb, args.labels)):
    S, tags = load(tb)
    st, wt = S("timing_s/step") if "timing_s/step" in tags else ({}, {})
    if not st: print(f"{lab}: no timing_s/step"); continue
    steps = sorted(st); v = np.array([st[s] for s in steps])
    steady = [s for s in steps if s >= args.steady_from and all(s % p for p in args.exclude_every)]
    assert steady, f"{lab}: steady-state window is empty (steps>={args.steady_from} excluding multiples of {args.exclude_every}); lower --steady_from"
    med = float(np.median([st[s] for s in steady])); p90 = float(np.percentile([st[s] for s in steady], 90)); vmax = float(v.max())
    c = f"C{i}"
    # panel 1: per-step step time of every run (eval/ckpt steps of the first run marked)
    clip = np.percentile([st[s] for s in steady], 99) * 1.5; vv = np.minimum(v, clip)
    ax[0].plot(steps, vv, color=c, lw=1 if i == 0 else 0.8, alpha=1 if i == 0 else 0.7, label=f"{lab}: median {med:.2f}s, p90 {p90:.2f}s (steady, n={len(steady)}); max step {vmax:.0f}s (clipped at {clip:.0f}s)")
    ax[0].axhline(med, color=c, ls="--", lw=0.8)
    if i == 0:
        excl = [s for s in steps if s not in steady and s >= 20]
        ax[0].scatter(excl, [min(st[s], vv.max()) for s in excl], color="#d62728", s=14, zorder=3, label="eval / checkpoint steps (clipped)")
    # panels 2 and 3: stacked breakdown for the first two runs. NB: for the disaggregated run timing_s/gen is the
    # trainer's WAIT for the sampler; for a colocated run it is the generation itself.
    if i in args.breakdown_idx:
        axb = ax[1 + args.breakdown_idx.index(i)]; bottom = np.zeros(len(steps)); is_wait = i in args.wait_idx
        for tag, name in parts:
            d = S(tag)[0] if tag in tags else {}
            if not d: continue
            y = np.minimum(np.array([d.get(s, 0.0) for s in steps]), med * 2.5)
            shown = name if (is_wait or not tag.endswith("/gen")) else "gen = generation"
            axb.bar(steps, y, bottom=bottom, width=1.0, label=f"{shown}: median {np.median([d.get(s,0.0) for s in steady]):.2f}s"); bottom += y
        axb.set_ylim(0, med * 2.5); axb.set_title(f"breakdown: {lab}  (gen = {'wait for sampler' if is_wait else 'generation'})", fontsize=10)
        axb.set_xlabel("training step"); axb.set_ylabel("s"); axb.legend(fontsize=7); axb.grid(alpha=0.3)
    # panel 3: cumulative wall clock from the first step's wall_time, + time to target
    if i < len(args.start_epoch) and args.start_epoch[i]:
        t0 = float(open(args.start_epoch[i]).read().strip()); origin = "launch (end-to-end)"
    else:
        t0 = wt[steps[0]] - st[steps[0]]; origin = "first training step (excl. startup and step-0 eval)"
    cum = [(wt[s] - t0) / 60 for s in steps]
    ax[3].plot(steps, cum, color=c, lw=1.8, label=f"{lab}: {cum[-1]:.0f} min to step {steps[-1]}; clock from {origin}")
    val, vwt = S(args.val_tag) if args.val_tag in tags else ({}, {})
    vs = sorted(val)
    if args.target_rule == "first":
        hit = [s for s in vs if val[s] >= args.target]
    else:   # rulebook convention: the SECOND of two consecutive evaluations >= target (the confirming one)
        hit = [b for a, b in zip(vs, vs[1:]) if val[a] >= args.target and val[b] >= args.target]
    if hit:
        s = hit[0]; t = (vwt[s] - t0) / 60; ax[3].scatter([s], [t], color=c, s=50, zorder=3); ax[3].annotate(f"{args.target:.2f} ({args.target_rule}) @ step {s}, {t:.0f} min", (s, t), textcoords="offset points", xytext=(6, -12), fontsize=8, color=c)
    summary.append((lab, med, p90, cum[-1], hit[0] if hit else None, origin))
ax[0].set_title("step time (all runs; clipped at 1.5x p99 of steady steps)"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("s"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
ax[3].set_title(f"cumulative wall clock ({args.target_rule} >= {args.target:.2f} marked; see legend for the clock origin)"); ax[3].set_xlabel("training step"); ax[3].set_ylabel("minutes"); ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(args.out, dpi=130)
for lab, med, p90, tot, hit, origin in summary: print(f"{lab:36s} steady step median {med:.2f}s p90 {p90:.2f}s (steps>={args.steady_from}) | total {tot:.1f} min from {origin} | {args.target_rule} >= {args.target}: step {hit}")
print("saved", args.out)
