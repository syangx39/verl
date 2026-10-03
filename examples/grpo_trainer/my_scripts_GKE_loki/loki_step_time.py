#!/usr/bin/env python3
"""Loki step-time report from a verl driver log: median over steps 4-11 (verl steps are 1-based; 1-3 warmup) of timing_s/step and the phase split
(rollout = timing_s/gen; train = update_actor + old_log_prob; other = step - rollout - train), plus prompt/output length and cap-hit rate."""
import re, sys, numpy as np
L = open(sys.argv[1]).read(); lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (4, 11)   # verl steps are 1-based; 1-3 warmup
def series(tag): return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"step:(\d+) .*?" + re.escape(tag) + r":([0-9.eE+-]+)", L)}
need = {t: series(t) for t in ("timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/old_log_prob", "response_length/mean", "prompt_length/mean", "response_length/clip_ratio")}
w = list(range(lo, hi + 1))
missing = {t: [s for s in w if s not in d] for t, d in need.items() if any(s not in d for s in w)}
assert not missing, f"metrics missing in the window {lo}-{hi}: {missing}"
st, gen, ua, olp, rl, pl, cap = (need[t] for t in ("timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/old_log_prob", "response_length/mean", "prompt_length/mean", "response_length/clip_ratio"))
train = {s: ua[s] + olp[s] for s in w}; other = {s: st[s] - gen[s] - train[s] for s in w}       # per-step phases first, then medians
med = lambda d: float(np.median([d[s] for s in w]))
print(f"steps {lo}-{hi} (n={len(w)}): step median {med(st):.1f} s (p90 {np.percentile([st[s] for s in w], 90):.1f}) | rollout {med(gen):.1f} s ({100*med(gen)/med(st):.0f}%) | train {med(train):.1f} s (update {med(ua):.1f} + old_logp {med(olp):.1f}) | other {med(other):.1f} s")
print(f"prompt mean {med(pl):.1f} tok | output mean {med(rl):.0f} tok | at cap {100*med(cap):.1f}% | steps logged {min(st)}-{max(st)}")
print("per step:", " ".join(f"{s}:{st[s]:.0f}" for s in w))