#!/usr/bin/env python3
"""Step-time report from a verl driver log, plus the wall clock per step from the run's TensorBoard events.
usage: step_time_report.py DRIVER_LOG [LO HI] [--tb DIR]
  default window: steps 4..last logged step; --tb defaults to <dir of DRIVER_LOG>/tensorboard when it exists.
Per step: train = update_actor + old_log_prob, other = step - gen - train; medians are taken AFTER the per-step split.
old_log_prob may be absent when the old-logprob forward is bypassed (single-forward runs): it then counts as 0.
timing_s/step is measured inside the trainer loop, so work done between steps (logging, TensorBoard writes, data loading)
is outside it. The TPU side reports wall time per global step, so the wall clock between consecutive steps (TensorBoard
event timestamps of timing_s/step) and its gap to timing_s/step are printed as well."""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("lo", nargs="?", type=int); ap.add_argument("hi", nargs="?", type=int)
ap.add_argument("--tb", default=None, help="TensorBoard dir of the run (default: <dir of DRIVER_LOG>/tensorboard)")
a = ap.parse_args()

L = open(a.log).read()
def series(tag):
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"step:(\d+) .*?" + re.escape(tag) + r":([0-9.eE+-]+)", L)}
req = ("timing_s/step", "timing_s/gen", "timing_s/update_actor", "response_length/mean", "prompt_length/mean", "response_length/clip_ratio", "critic/score/mean")
d = {t: series(t) for t in req}
olp = series("timing_s/old_log_prob")
assert d["timing_s/step"], "no timing_s/step in the log"
lo = a.lo if a.lo is not None else 4
hi = a.hi if a.hi is not None else max(d["timing_s/step"])
w = list(range(lo, hi + 1))
missing = {t: [s for s in w if s not in v] for t, v in d.items() if any(s not in v for s in w)}
assert not missing, f"metrics missing in the window {lo}-{hi}: { {t: v[:5] for t, v in missing.items()} }"
olp_note = ""
if not olp:
    olp = {s: 0.0 for s in w}; olp_note = " [old_log_prob not logged: bypassed]"
else:
    gaps = [s for s in w if s not in olp]
    assert not gaps, f"old_log_prob missing at steps {gaps[:5]}"
st, gen, ua = d["timing_s/step"], d["timing_s/gen"], d["timing_s/update_actor"]
train = {s: ua[s] + olp[s] for s in w}
other = {s: st[s] - gen[s] - train[s] for s in w}
med = lambda x: float(np.median([x[s] for s in w]))
v = np.array([st[s] for s in w])
print(f"steps {lo}-{hi} (n={len(w)}): step median {med(st):.1f} s (mean {v.mean():.1f}, p90 {np.percentile(v, 90):.1f}, max {v.max():.1f}) | "
      f"rollout {med(gen):.1f} s ({100 * med(gen) / med(st):.0f}%) | train {med(train):.1f} s (update {med(ua):.1f} + old_logp {med(olp):.1f}){olp_note} | other {med(other):.1f} s")
k = min(10, len(w))
sc = d["critic/score/mean"]
print(f"prompt mean {med(d['prompt_length/mean']):.1f} tok | output mean {med(d['response_length/mean']):.0f} tok | at cap {100 * med(d['response_length/clip_ratio']):.1f}% | "
      f"score first{k} {np.mean([sc[s] for s in w[:k]]):.3f} -> last{k} {np.mean([sc[s] for s in w[-k:]]):.3f} | steps logged {min(st)}-{max(st)}")
if len(w) <= 20:
    print("per step:", " ".join(f"{s}:{st[s]:.0f}" for s in w))
else:
    blocks = [w[i:i + 25] for i in range(0, len(w), 25)]
    print("median per 25 steps:", " ".join(f"{b[0]}-{b[-1]}:{np.median([st[s] for s in b]):.1f}" for b in blocks))

# ---- wall clock per step from the TensorBoard event timestamps ----
tb = a.tb or os.path.join(os.path.dirname(os.path.abspath(a.log)), "tensorboard")
ev = glob.glob(os.path.join(tb, "**", "events.out.tfevents*"), recursive=True)
if not ev:
    print(f"wall clock: no TensorBoard events under {tb}; skipped")
else:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        EventAccumulator = None
        print("wall clock: tensorboard is not importable here; skipped")
    if EventAccumulator is not None:
        wt = {}
        for evdir in sorted({os.path.dirname(p) for p in ev}):
            acc = EventAccumulator(evdir, size_guidance={"scalars": 0}); acc.Reload()
            if "timing_s/step" in acc.Tags().get("scalars", []):
                for e in acc.Scalars("timing_s/step"):
                    wt[e.step] = e.wall_time
        pairs = [s for s in w if s in wt and s - 1 in wt]
        if not pairs:
            print("wall clock: no consecutive steps with TensorBoard timestamps in the window; skipped")
        else:
            dw = np.array([wt[s] - wt[s - 1] for s in pairs]); gap = np.array([wt[s] - wt[s - 1] - st[s] for s in pairs])
            print(f"wall clock between steps (TensorBoard timestamps, n={len(pairs)}): median {np.median(dw):.1f} s (p90 {np.percentile(dw, 90):.1f}) | "
                  f"gap to timing_s/step: median {np.median(gap):.1f} s, max {gap.max():.1f} s")
