#!/usr/bin/env python3
"""Loki step-time report from a verl driver log: median over steps 3-10 (steps 0-2 warmup) of timing_s/step and the phase split
(rollout = timing_s/gen; train = update_actor + old_log_prob; other = step - rollout - train), plus prompt/output length and cap-hit rate."""
import re, sys, numpy as np
L = open(sys.argv[1]).read(); lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (3, 10)
def series(tag): return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"step:(\d+) .*?" + re.escape(tag) + r":([0-9.eE+-]+)", L)}
st, gen, ua, olp = series("timing_s/step"), series("timing_s/gen"), series("timing_s/update_actor"), series("timing_s/old_log_prob")
rl, pl, cap = series("response_length/mean"), series("prompt_length/mean"), series("response_length/clip_ratio")
w = [s for s in range(lo, hi + 1) if s in st]; assert w, "no steps in window"
med = lambda d: float(np.median([d.get(s, 0.0) for s in w]))
step, r, t = med(st), med(gen), med(ua) + med(olp)
print(f"steps {w[0]}-{w[-1]} (n={len(w)}): step {step:.1f} s | rollout {r:.1f} s ({100*r/step:.0f}%) | train {t:.1f} s (update {med(ua):.1f} + old_logp {med(olp):.1f}) | other {step-r-t:.1f} s")
print(f"prompt mean {med(pl):.1f} tok | output mean {med(rl):.0f} tok | at cap {100*med(cap):.1f}% | steps logged {sorted(st)[0]}-{sorted(st)[-1]}")