#!/usr/bin/env python3
"""Collapse guard (v2) for Phase 0 runs.

Polls the run's node-local TensorBoard dir (+ the rollout dump for per-step
train accuracy) and STOPS the training driver only on the agreed joint rule:

  STOP  = acc_bad AND (entropy_bad OR cap_bad)
  WARN  = any single signal (entropy / cap / grad / acc) abnormal on its own

  entropy_bad  actor/entropy > ENT_MULT x baseline for ENT_STEPS consecutive steps
  cap_bad      response_length/clip_ratio > CAP_FRAC for CAP_STEPS consecutive steps
  acc_bad      per-step train acc (rollout dump) < ACC_MULT x baseline for ACC_STEPS
               consecutive steps, evaluated only after WARMUP steps
  grad (warn)  actor/grad_norm > GRAD_MAX on >= GRAD_STEPS of the last 20 steps

baseline = mean of the first BASE_STEPS logged steps. No rule uses the training
reward (it is signed once a length penalty is on). The guard never changes LR or
rolls back; it only warns/stops. The stop is scoped to ONE driver: --pid (SIGTERM,
then SIGKILL after --grace seconds). Without --pid the guard is report-only.

Outputs: <tb_dir>/COLLAPSE_WARN.txt (appended), <tb_dir>/COLLAPSE_ABORT.txt.
Usage (started by the launcher):
  python3 collapse_guard.py --tb <TB_DIR> --rollout <ROLLOUT_DUMP_DIR> --pid <DRIVER_PID> [--poll 60]
"""

import argparse
import glob
import json
import os
import signal
import time

from tensorboard.backend.event_processing import event_accumulator as ea


def load_tb(tb_dir):
  acc = ea.EventAccumulator(tb_dir, size_guidance={ea.SCALARS: 0})
  acc.Reload()
  return {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in acc.Tags().get("scalars", [])}


def rollout_acc(rollout_dir, max_files=40):
  files = []
  for fn in glob.glob(os.path.join(rollout_dir, "*.jsonl")):
    try:
      files.append((int(os.path.splitext(os.path.basename(fn))[0]), fn))
    except ValueError:
      pass
  files.sort()
  out = []
  for step, fn in files[-max_files:]:
    n = a = 0
    try:
      with open(fn, encoding="utf-8") as f:
        for line in f:
          if line.strip():
            n += 1
            a += float(json.loads(line).get("acc", 0.0))
    except (OSError, ValueError):
      continue
    if n:
      out.append((step, a / n))
  return out


def tail_run(series, pred):
  n = 0
  for _, v in reversed(series):
    if pred(v):
      n += 1
    else:
      break
  return n


def alive(pid):
  try:
    os.kill(pid, 0)
    return True
  except OSError:
    return False


def stop_driver(pid, grace):
  os.kill(pid, signal.SIGTERM)
  t0 = time.time()
  while time.time() - t0 < grace:
    if not alive(pid):
      return "SIGTERM"
    time.sleep(2)
  os.kill(pid, signal.SIGKILL)
  return "SIGKILL"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tb", required=True)
  ap.add_argument("--rollout", default=None)
  ap.add_argument("--pid", type=int, default=None, help="driver PID to stop (report-only if omitted)")
  ap.add_argument("--grace", type=int, default=30)
  ap.add_argument("--poll", type=int, default=60)
  ap.add_argument("--base_steps", type=int, default=5)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--ent_mult", type=float, default=3.0)
  ap.add_argument("--ent_steps", type=int, default=3)
  ap.add_argument("--cap_frac", type=float, default=0.9)
  ap.add_argument("--cap_steps", type=int, default=5)
  ap.add_argument("--acc_mult", type=float, default=0.5)
  ap.add_argument("--acc_steps", type=int, default=10)
  ap.add_argument("--grad_max", type=float, default=5.0)
  ap.add_argument("--grad_steps", type=int, default=3)
  ap.add_argument("--once", action="store_true", help="evaluate once on the current data and exit (tests)")
  args = ap.parse_args()

  print(f"[guard] watching {args.tb} (+{args.rollout}) every {args.poll}s; driver pid={args.pid}", flush=True)
  base = {}
  warned = set()
  while True:
    if not args.once:
      time.sleep(args.poll)
    if args.pid and not alive(args.pid):
      print("[guard] driver exited; guard done", flush=True)
      return
    if not os.path.isdir(args.tb):
      if args.once:
        return
      continue
    try:
      m = load_tb(args.tb)
    except Exception as e:  # noqa: BLE001
      print(f"[guard] tb read error: {e}", flush=True)
      if args.once:
        return
      continue
    ent = m.get("actor/entropy", [])
    cap = m.get("response_length/clip_ratio", [])
    gn = m.get("actor/grad_norm", [])
    acc = rollout_acc(args.rollout) if args.rollout and os.path.isdir(args.rollout) else []
    if not ent:
      if args.once:
        return
      continue
    step = ent[-1][0]
    for name, series in (("ent", ent), ("acc", acc)):
      if name not in base and len(series) >= args.base_steps:
        base[name] = sum(v for _, v in series[:args.base_steps]) / args.base_steps

    ent_k = tail_run(ent, lambda v: v > args.ent_mult * base["ent"]) if "ent" in base else 0
    cap_k = tail_run(cap, lambda v: v > args.cap_frac) if cap else 0
    acc_k = tail_run(acc, lambda v: v < args.acc_mult * base["acc"]) if ("acc" in base and step > args.warmup) else 0
    grad_n = sum(v > args.grad_max for _, v in gn[-20:]) if gn else 0
    ent_bad, cap_bad, acc_bad, grad_bad = (ent_k >= args.ent_steps, cap_k >= args.cap_steps,
                                           acc_k >= args.acc_steps, grad_n >= args.grad_steps)

    status = (f"[guard] step {step}: entropy {ent[-1][1]:.3f}" +
              (f" (x{ent[-1][1] / base['ent']:.1f})" if "ent" in base else "") +
              (f" cap {cap[-1][1]:.2f}" if cap else "") +
              (f" acc {acc[-1][1]:.3f}" + (f" (x{acc[-1][1] / max(base['acc'], 1e-9):.2f})" if "acc" in base else "") if acc else "") +
              (f" grad {gn[-1][1]:.3f}" if gn else ""))
    print(status, flush=True)

    for key, bad, txt in (("ent", ent_bad, f"entropy > {args.ent_mult}x baseline for {ent_k} steps"),
                          ("cap", cap_bad, f"cap-hit > {args.cap_frac} for {cap_k} steps"),
                          ("grad", grad_bad, f"grad_norm > {args.grad_max} on {grad_n} of last 20 steps"),
                          ("acc", acc_bad, f"train acc < {args.acc_mult}x baseline for {acc_k} steps")):
      if bad and key not in warned:
        warned.add(key)
        msg = f"WARN step {step}: {txt}\n"
        print("[guard] " + msg.strip(), flush=True)
        with open(os.path.join(args.tb, "COLLAPSE_WARN.txt"), "a") as f:
          f.write(msg)
      elif not bad and key in warned:
        warned.discard(key)

    if acc_bad and (ent_bad or cap_bad):
      parts = []
      if ent_bad:
        parts.append(f"entropy x{ent[-1][1] / base['ent']:.1f}")
      if cap_bad:
        parts.append(f"cap-hit {cap[-1][1]:.2f}")
      reason = (f"train acc {acc[-1][1]:.3f} < {args.acc_mult}x baseline {base['acc']:.3f} for {acc_k} steps"
                f" AND {' / '.join(parts)}")
      msg = f"COLLAPSE at step {step}: {reason}\n{status}\n"
      print("[guard] " + msg, flush=True)
      with open(os.path.join(args.tb, "COLLAPSE_ABORT.txt"), "w") as f:
        f.write(msg)
      if args.pid:
        how = stop_driver(args.pid, args.grace)
        print(f"[guard] driver {args.pid} stopped via {how}", flush=True)
      return
    if args.once:
      return


if __name__ == "__main__":
  main()