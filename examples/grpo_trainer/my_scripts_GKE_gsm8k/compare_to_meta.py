#!/usr/bin/env python3
"""Overlay our verl reproduction against Meta's reference curves.

Meta's TensorBoard export is expected as CSV files with columns step,value (one per metric),
e.g. meta/reward_accuracy.csv (train-batch accuracy per step), meta/eval_accuracy.csv (every 20 steps).
Ours come from the run's TensorBoard dir and rollout dump.

Prints per-step / per-checkpoint differences with the noise bands implied by the batch sizes
(train: 128 prompts/step -> sigma ~ sqrt(p(1-p)/128) ~ 0.035 at p=0.8; eval: 512 questions -> ~0.02).
Usage:
  python3 compare_to_meta.py --tb <TB_DIR> --rollout <ROLLOUT_DUMP_DIR> --meta_train meta/reward_accuracy.csv \
      [--meta_eval meta/eval_accuracy.csv] [--eval_tag val-core/gsm8k_boxed_test512/acc/mean@1] --out <png>
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as ea


def read_csv(p, col):
  """Meta's TensorBoard exports: one row per step, named metric columns (reward/accuracy, eval/accuracy, ...)."""
  pts = {}
  with open(p) as f:
    rd = csv.DictReader(f)
    if col not in rd.fieldnames:
      raise SystemExit(f"{p} has no column {col!r}; columns: {rd.fieldnames}")
    for row in rd:
      if row.get(col, "") != "":
        pts[int(float(row["step"]))] = float(row[col])
  if not pts:
    raise SystemExit(f"{p}: column {col!r} has no values")
  return pts


def ours_rollout_stats(rollout_dir, groups, group_size):
  """Per COMPLETE step: mean acc, mean raw reward, and the fraction of groups whose TRAINING rewards (score =
  raw + overlong penalty, the quantity advantages are computed from) are all identical -- Meta's reward/frac_zero_std."""
  out, skipped = {}, []
  for f in glob.glob(os.path.join(rollout_dir, "*.jsonl")):
    try:
      s = int(os.path.basename(f)[:-6])
    except ValueError:
      continue
    n = a = rr = 0.0; by_uid = {}; ok = True
    try:
      for l in open(f):
        if not l.strip():
          continue
        r = json.loads(l)
        if "acc" not in r or "uid" not in r or not np.isfinite(float(r["acc"])):
          ok = False; break
        n += 1; a += float(r["acc"]); rr += float(r.get("reward_raw", r.get("score", 0.0)))
        by_uid.setdefault(r["uid"], []).append(float(r["score"]))      # TRAINING reward (raw + overlong penalty) -> what advantages see
    except (OSError, ValueError):
      ok = False
    if ok and n == groups * group_size and len(by_uid) == groups and all(len(v) == group_size for v in by_uid.values()):
      zero = sum(1 for v in by_uid.values() if max(v) == min(v)) / groups
      out[s] = {"acc": a / n, "reward_raw": rr / n, "frac_zero_std": zero}
    else:
      skipped.append(s)
  if skipped:
    print(f"!! {len(skipped)} rollout steps skipped as incomplete/invalid (expect {groups}x{group_size}): {sorted(skipped)[:8]}...")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tb", required=True); ap.add_argument("--rollout", required=True)
  ap.add_argument("--meta_train", required=True, help="reference/train_metrics.csv"); ap.add_argument("--meta_eval", default=None, help="reference/eval_metrics.csv")
  ap.add_argument("--meta_train_col", default="reward/accuracy"); ap.add_argument("--meta_eval_col", default="eval/accuracy")
  ap.add_argument("--eval_tag", default="val-core/gsm8k_boxed_test512/acc/mean@1")
  ap.add_argument("--out", required=True)
  ap.add_argument("--groups", type=int, default=128); ap.add_argument("--group_size", type=int, default=16)
  a = ap.parse_args()
  acc = ea.EventAccumulator(a.tb, size_guidance={ea.SCALARS: 0}); acc.Reload()
  tags = acc.Tags()["scalars"]
  stats = ours_rollout_stats(a.rollout, a.groups, a.group_size)
  ours_tr = {k: v["acc"] for k, v in stats.items()}
  meta_tr = read_csv(a.meta_train, a.meta_train_col)
  if not ours_tr:
    raise SystemExit("no complete rollout steps on our side")
  if not set(ours_tr) & set(meta_tr):
    raise SystemExit("no common training steps between ours and Meta's CSV -- check step numbering (both 1-based)")
  ev_tag = a.eval_tag if a.eval_tag in tags else (a.eval_tag.replace("@", "_") if a.eval_tag.replace("@", "_") in tags else None)  # some writers sanitize '@'
  ours_ev = {e.step: e.value for e in acc.Scalars(ev_tag)} if ev_tag else {}
  if not ours_ev:
    print(f"!! eval tag {a.eval_tag!r} not found in {a.tb}; available val tags: {[t for t in tags if 'val-core' in t][:6]}")
  meta_ev = read_csv(a.meta_eval, a.meta_eval_col) if a.meta_eval else {}
  # secondary channels (printed, not gated): completion length, cap-hit ratio, zero-std groups
  def tb(tag):
    return {e.step: e.value for e in acc.Scalars(tag)} if tag in tags else {}
  extras = [("completions/length_mean", tb("response_length/mean")),
            ("completions/clipped_ratio", tb("response_length/clip_ratio")),
            ("reward/frac_zero_std", {k: v["frac_zero_std"] for k, v in stats.items()}),
            ("reward/mean", tb("critic/score/mean"))]
  for col, ours in extras:
    try:
      m = read_csv(a.meta_train, col)
    except SystemExit:
      continue
    kk = sorted(set(m) & set(ours))
    if kk:
      dd = np.array([ours[k] - m[k] for k in kk])
      print(f"{col:<28} common {len(kk):>3} | ours-meta mean {dd.mean():+.4f} | first5 ours {np.mean([ours[k] for k in kk[:5]]):.3f} meta {np.mean([m[k] for k in kk[:5]]):.3f} | last5 ours {np.mean([ours[k] for k in kk[-5:]]):.3f} meta {np.mean([m[k] for k in kk[-5:]]):.3f}")

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  ks = sorted(set(ours_tr) & set(meta_tr))
  d = np.array([ours_tr[k] - meta_tr[k] for k in ks])
  axes[0].plot(sorted(meta_tr), [meta_tr[k] for k in sorted(meta_tr)], color="#d62728", lw=1.5, label="Meta reference (train-batch acc)")
  axes[0].plot(sorted(ours_tr), [ours_tr[k] for k in sorted(ours_tr)], color="#1f77b4", lw=1.5, label="ours (verl, rollout dump acc)")
  axes[0].set_title("train-batch accuracy per step (T=1, 128 prompts x 16)"); axes[0].set_xlabel("step"); axes[0].legend(); axes[0].grid(alpha=0.3)
  print(f"train acc: {len(ks)} common steps | ours-meta mean {d.mean():+.4f} sd {d.std():.4f} max|d| {np.abs(d).max():.3f} | "
        f"first5 ours {np.mean([ours_tr[k] for k in ks[:5]]):.3f} meta {np.mean([meta_tr[k] for k in ks[:5]]):.3f} | "
        f"last5 ours {np.mean([ours_tr[k] for k in ks[-5:]]):.3f} meta {np.mean([meta_tr[k] for k in ks[-5:]]):.3f}")
  print(f"  (reference only, not an acceptance rule) binomial 2-sigma for one step with {a.groups} prompts at p=0.8: ±{2*np.sqrt(0.16/a.groups):.3f}; "
        f"steps with |d| beyond it: {int((np.abs(d) > 2*np.sqrt(0.16/a.groups)).sum())}/{len(ks)}")
  if meta_ev and ours_ev:
    ke = sorted(set(ours_ev) & set(meta_ev)); de = np.array([ours_ev[k] - meta_ev[k] for k in ke])
    axes[1].plot(sorted(meta_ev), [meta_ev[k] for k in sorted(meta_ev)], "o-", color="#d62728", label="Meta eval (512, greedy)")
    axes[1].plot(sorted(ours_ev), [ours_ev[k] for k in sorted(ours_ev)], "o-", color="#1f77b4", label="ours eval (test512, greedy)")
    print(f"eval acc: {len(ke)} common checkpoints | ours-meta per checkpoint: " + ", ".join(f"{k}:{v:+.3f}" for k, v in zip(ke, de)) +
          f" | (reference only) binomial 2-sigma for 512 questions at p=0.8: ±{2*np.sqrt(0.16/512):.3f}")
  else:
    axes[1].plot(sorted(ours_ev), [ours_ev[k] for k in sorted(ours_ev)], "o-", color="#1f77b4", label="ours eval (test512, greedy)")
    print("eval: Meta eval CSV not provided; ours only:", {k: round(v, 3) for k, v in sorted(ours_ev.items())})
  axes[1].set_title("eval accuracy (greedy)"); axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(alpha=0.3)
  fig.tight_layout(); fig.savefig(a.out, dpi=120); print("saved", a.out)


if __name__ == "__main__":
  main()