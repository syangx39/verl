#!/usr/bin/env python3
"""Collapse guard for Phase 0 runs.

Polls the run's (node-local) TensorBoard directory and aborts the training job
when the policy shows the collapse signature seen in the v4 pilot, so a dead run
does not burn 5 hours:

  entropy   actor/entropy > ENT_MULT x baseline for ENT_STEPS consecutive steps
  acc       per-step train acc (from the rollout dump) < ACC_MULT x baseline for
            ACC_STEPS consecutive steps (after WARMUP)
  cap-hit   response_length/clip_ratio > CAP_FRAC for CAP_STEPS consecutive steps
  score     critic/score/mean < SCORE_MULT x baseline for SCORE_STEPS consecutive
            steps (only after WARMUP steps, to let early format learning settle)
  grad      actor/grad_norm > GRAD_MAX on GRAD_STEPS steps within the last 20

baseline = mean of the first BASE_STEPS logged steps of each metric.

Rules are OR-ed (any one fires); each requires its own run of consecutive steps.
The guard only STOPS; it never changes LR or rolls back.

On trigger: writes <tb_dir>/COLLAPSE_ABORT.txt with the reason and the last
values, then `pkill -f verl.trainer.main_ppo` (the driver in this pod; Ray tears
the job's actors down with it). Prints a one-line status every poll.

Usage (started by the launcher in the background):
  python3 collapse_guard.py --tb <TB_DIR> [--poll 60] [--dry_run]
"""

import argparse
import os
import subprocess
import time
from collections import deque

from tensorboard.backend.event_processing import event_accumulator as ea


def load(tb_dir):
  acc = ea.EventAccumulator(tb_dir, size_guidance={ea.SCALARS: 0})
  acc.Reload()
  tags = acc.Tags().get("scalars", [])
  return {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in tags}


def rollout_acc(rollout_dir, max_files=40):
  """Mean acc per step from the newest <step>.jsonl files (cheap: reads only acc fields)."""
  import glob
  import json
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


def consecutive_tail(series, pred):
  """Number of trailing points satisfying pred."""
  n = 0
  for _, v in reversed(series):
    if pred(v):
      n += 1
    else:
      break
  return n


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tb", required=True)
  ap.add_argument("--rollout", default=None, help="trainer.rollout_data_dir; enables the per-step ACC rule")
  ap.add_argument("--acc_mult", type=float, default=0.5)
  ap.add_argument("--acc_steps", type=int, default=10)
  ap.add_argument("--poll", type=int, default=60)
  ap.add_argument("--base_steps", type=int, default=5)
  ap.add_argument("--warmup", type=int, default=20)
  ap.add_argument("--ent_mult", type=float, default=3.0)
  ap.add_argument("--ent_steps", type=int, default=3)
  ap.add_argument("--cap_frac", type=float, default=0.9)
  ap.add_argument("--cap_steps", type=int, default=5)
  ap.add_argument("--score_mult", type=float, default=0.5)
  ap.add_argument("--score_steps", type=int, default=10)
  ap.add_argument("--grad_max", type=float, default=5.0)
  ap.add_argument("--grad_steps", type=int, default=3)
  ap.add_argument("--dry_run", action="store_true", help="report, do not kill")
  args = ap.parse_args()

  print(f"[guard] watching {args.tb} every {args.poll}s (dry_run={args.dry_run})", flush=True)
  baselines = {}
  while True:
    time.sleep(args.poll)
    if not os.path.isdir(args.tb):
      continue
    try:
      m = load(args.tb)
    except Exception as e:  # noqa: BLE001
      print(f"[guard] read error: {e}", flush=True)
      continue
    ent = m.get("actor/entropy", [])
    cap = m.get("response_length/clip_ratio", [])
    sc = m.get("critic/score/mean", [])
    gn = m.get("actor/grad_norm", [])
    if not ent:
      continue
    step = ent[-1][0]
    for name, series in (("ent", ent), ("score", sc)):
      if name not in baselines and len(series) >= args.base_steps:
        baselines[name] = sum(v for _, v in series[:args.base_steps]) / args.base_steps
    reason = None
    if "ent" in baselines:
      k = consecutive_tail(ent, lambda v: v > args.ent_mult * baselines["ent"])
      if k >= args.ent_steps:
        reason = f"entropy {ent[-1][1]:.3f} > {args.ent_mult}x baseline {baselines['ent']:.3f} for {k} steps"
    if reason is None and cap:
      k = consecutive_tail(cap, lambda v: v > args.cap_frac)
      if k >= args.cap_steps:
        reason = f"cap-hit {cap[-1][1]:.2f} > {args.cap_frac} for {k} steps"
    if reason is None and "score" in baselines and step > args.warmup:
      k = consecutive_tail(sc, lambda v: v < args.score_mult * baselines["score"])
      if k >= args.score_steps:
        reason = f"score {sc[-1][1]:.3f} < {args.score_mult}x baseline {baselines['score']:.3f} for {k} steps"
    if reason is None and args.rollout and os.path.isdir(args.rollout):
      acc = rollout_acc(args.rollout)
      if acc and "acc" not in baselines and len(acc) >= args.base_steps:
        baselines["acc"] = sum(v for _, v in acc[:args.base_steps]) / args.base_steps
      if acc and "acc" in baselines and step > args.warmup:
        k = consecutive_tail(acc, lambda v: v < args.acc_mult * baselines["acc"])
        if k >= args.acc_steps:
          reason = f"train acc {acc[-1][1]:.3f} < {args.acc_mult}x baseline {baselines['acc']:.3f} for {k} steps"
    if reason is None and gn:
      recent = [v for _, v in gn[-20:]]
      if sum(v > args.grad_max for v in recent) >= args.grad_steps:
        reason = f"grad_norm > {args.grad_max} on {sum(v > args.grad_max for v in recent)} of last 20 steps"
    status = (f"[guard] step {step}: entropy {ent[-1][1]:.3f}"
              f"{' (x%.1f)' % (ent[-1][1] / baselines['ent']) if 'ent' in baselines else ''}"
              f" cap {cap[-1][1]:.2f} score {sc[-1][1]:.3f} grad {gn[-1][1]:.3f}" if cap and sc and gn else f"[guard] step {step}")
    print(status, flush=True)
    if reason:
      msg = f"COLLAPSE at step {step}: {reason}\n{status}\n"
      print("[guard] " + msg, flush=True)
      with open(os.path.join(args.tb, "COLLAPSE_ABORT.txt"), "w") as f:
        f.write(msg)
      if not args.dry_run:
        subprocess.run("pkill -f 'verl.trainer.main_ppo' || true", shell=True)
        print("[guard] driver killed; the launcher's EXIT trap mirrors TB and exits", flush=True)
      return


if __name__ == "__main__":
  main()