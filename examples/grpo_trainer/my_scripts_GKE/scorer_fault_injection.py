#!/usr/bin/env python3
"""Controlled tests for maxtext_math_reward.py failure handling (both stacks).

Verifies the frozen rule: for a per-sample verification timeout or exception,
acc = 0, the length penalty is retained, the sample is retained, and the
corresponding flag is set. Deterministic scoring is covered by
scorer_fixture.jsonl; this script covers the non-deterministic branches.

Usage:
  REWARD_FMT_WEIGHT=0 REWARD_OVERLONG_BUFFER=1024 python3 scorer_fault_injection.py /path/to/maxtext_math_reward.py
Exit code 0 = all checks passed.
"""
import importlib.util
import json
import os
import sys
import threading

os.environ.setdefault("REWARD_DUMP_DIR", "")
path = sys.argv[1] if len(sys.argv) > 1 else "maxtext_math_reward.py"
spec = importlib.util.spec_from_file_location("r", path)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

GT = json.dumps(["\\frac{1}{2}", "\\frac{1}{2}"])
COMP = "<reasoning>x</reasoning><answer>1/2</answer>"      # needs math_verify (1/2 vs \frac{1}{2})
CAP = int(os.environ.get("REWARD_MAX_RESP_LEN", "8192"))
PEN = float(os.environ.get("REWARD_OVERLONG_PENALTY", "1.0"))
FMT_W = float(os.environ.get("REWARD_FMT_WEIGHT", "0.1"))
BUF = int(os.environ.get("REWARD_OVERLONG_BUFFER", "0"))
fails = []


def check(name, cond, detail=""):
  print(("OK   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
  if not cond:
    fails.append(name)


def respawn_all():
  """Fork fresh workers so they inherit the current (monkeypatched) parse()."""
  for _ in range(r._MV_PROCS):
    w = r._mv_idle.get()
    w.kill()
    r._mv_idle.put(r._mv_spawn_worker())


def score(n):
  return r.compute_score("test", COMP, GT, extra_info={"index": 0, "response_len": n})


# 0. healthy baseline from a worker thread (verl calls the scorer from threads)
res = {}
t = threading.Thread(target=lambda: res.__setitem__("o", score(100)))
t.start(); t.join()
o = res["o"]
check("baseline: equivalence scored from a worker thread", o["acc"] == 1.0 and abs(o["score"] - (1.0 + FMT_W)) < 1e-9, str(o))

expected_pen_at_cap = -PEN if BUF > 0 else 0.0

# 1. injected verifier exception
real_parse = r.parse
def bad_parse(*a, **k):
  raise ValueError("injected verifier exception")
r.parse = bad_parse
respawn_all()
o = score(CAP)
check("exception: acc == 0", o["acc"] == 0.0, str(o))
check("exception: mv_exc == 1, mv_timeout == 0", o["mv_exc"] == 1.0 and o["mv_timeout"] == 0.0)
check("exception: length penalty retained at cap", abs(o["length_penalty"] - expected_pen_at_cap) < 1e-9, f"{o['length_penalty']} vs {expected_pen_at_cap}")
check("exception: score == fmt_w*fmt + 0 + penalty", abs(o["score"] - (FMT_W * o["fmt"] + expected_pen_at_cap)) < 1e-9, str(o["score"]))

# 2. injected verifier timeout exception (math_verify's own TimeoutException, a BaseException) -- exercises
#    the exception-classification path only; the real watchdog is exercised in test 3
from math_verify.errors import TimeoutException
def slow_parse(*a, **k):
  raise TimeoutException("injected verifier timeout")
r.parse = slow_parse
respawn_all()
o = score(CAP)
check("timeout: acc == 0", o["acc"] == 0.0, str(o))
check("timeout: mv_timeout == 1, mv_exc == 0", o["mv_timeout"] == 1.0 and o["mv_exc"] == 0.0)
check("timeout: length penalty retained at cap", abs(o["length_penalty"] - expected_pen_at_cap) < 1e-9)

# 3. TRUE hang: a verifier that blocks. The patched parse() is not wrapped by math_verify's own
#    signal timeout, so only the outer watchdog (poll timeout) can end it: expect mv_timeout=1,
#    the hung worker killed and AUTOMATICALLY replaced, score at cap == -1, and -- the point of the
#    recovery check -- the auto-replaced worker itself serving the next call correctly.
import time as _time
def hanging_parse(*a, **k):
  _time.sleep(120)
saved_timeout = r._MV_TIMEOUT
r._MV_TIMEOUT = 1.0                     # parent poll window = 1 s + 1 s grace; keeps the test short
r.parse = hanging_parse
respawn_all()                           # all workers now carry the hanging parse
r.parse = real_parse                    # restore in the PARENT only: any auto-replacement forks a healthy worker
others = [r._mv_idle.get() for _ in range(r._MV_PROCS - 1)]   # park the rest; exactly one hanging worker is idle
before = r.mv_stats()
t0 = _time.time()
o = score(CAP)                          # served by the hanging worker -> watchdog -> auto replacement
elapsed = _time.time() - t0
after = r.mv_stats()
check("hang: watchdog fired (call returned within ~5 s)", elapsed < 5.0, f"{elapsed:.1f}s")
check("hang: acc == 0, mv_timeout == 1, mv_exc == 0", o["acc"] == 0.0 and o["mv_timeout"] == 1.0 and o["mv_exc"] == 0.0, str(o))
check("hang: length penalty retained at cap -> score == -1 (with fmt_w=0)", abs(o["length_penalty"] - expected_pen_at_cap) < 1e-9 and abs(o["score"] - (FMT_W * o["fmt"] + expected_pen_at_cap)) < 1e-9, str(o["score"]))
check("hang: hung worker auto-replaced (replaced +1, exactly one idle)", after["replaced"] == before["replaced"] + 1 and after["idle"] == 1, str(after))
o = score(100)                          # the only idle worker is the auto-replacement -> it must be healthy
check("hang: recovery -- the AUTO-REPLACED worker serves the next call correctly", o["acc"] == 1.0 and o["mv_timeout"] == 0.0 and o["mv_exc"] == 0.0, str(o))
for x in others:                        # return the parked (still hanging-parse) workers, then rebuild the pool cleanly
  r._mv_idle.put(x)
r._MV_TIMEOUT = saved_timeout
respawn_all()
check("hang: pool rebuilt", r.mv_stats()["idle"] == r._MV_PROCS and score(100)["acc"] == 1.0)

# 4. dead worker: malformed message kills one worker; next call on it must count mv_exc and replace it
r.parse = real_parse
respawn_all()
w = r._mv_idle.get()
w.conn.send((1, 2, 3))
w.proc.join(timeout=5)
check("dead worker: process exited after malformed message", not w.proc.is_alive())
r._mv_idle.put(w)
others = [r._mv_idle.get() for _ in range(r._MV_PROCS - 1)]
o = score(100)
for x in others:
  r._mv_idle.put(x)
check("dead worker: call counted as mv_exc, acc == 0", o["mv_exc"] == 1.0 and o["acc"] == 0.0, str(o))
o2 = score(100)
check("dead worker: replaced, next call healthy", o2["acc"] == 1.0 and r.mv_stats()["idle"] == r._MV_PROCS, str(r.mv_stats()))

print("\nstats:", r.mv_stats())
print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)