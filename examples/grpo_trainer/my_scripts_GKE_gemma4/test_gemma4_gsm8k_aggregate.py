#!/usr/bin/env python3
"""Tests for gemma4_gsm8k_aggregate (multi-seed statistics) and gemma4_gsm8k_report.step_intervals (clock sources).

    python3 test_gemma4_gsm8k_aggregate.py      # needs numpy; no matplotlib (the plots are not exercised)

The aggregate cases build three small synthetic runs (val dumps + per-seed report summaries) in a temporary directory.
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gemma4_gsm8k_aggregate as agg  # noqa: E402
import gemma4_gsm8k_report as rep  # noqa: E402

# Per seed and step: (acc, fmt, output) for 6 questions. At step 0 all seeds share the initial weights; seed 3 differs
# from seeds 1-2 on question 4 only, where its output has no box (eval variation). At step 20 seed 2 misses question 0
# and seed 3 misses question 5, both with a box.
Q = 6
RUNS = {
    "seed1": {0: [(1, 1, "a"), (0, 0, "b"), (0, 1, "c"), (1, 1, "d"), (1, 1, "e"), (0, 0, "f")],
              20: [(1, 1, "a1"), (1, 1, "b1"), (0, 1, "c1"), (1, 1, "d1"), (1, 1, "e1"), (1, 1, "f1")]},
    "seed2": {0: [(1, 1, "a"), (0, 0, "b"), (0, 1, "c"), (1, 1, "d"), (1, 1, "e"), (0, 0, "f")],
              20: [(0, 1, "a2"), (1, 1, "b2"), (0, 1, "c2"), (1, 1, "d2"), (1, 1, "e2"), (1, 1, "f2")]},
    "seed3": {0: [(1, 1, "a"), (0, 0, "b"), (0, 1, "c"), (1, 1, "d"), (0, 0, "e'"), (0, 0, "f")],
              20: [(1, 1, "a3"), (1, 1, "b3"), (0, 1, "c3"), (1, 1, "d3"), (1, 1, "e3"), (0, 1, "f3")]},
}


def write_run(root, label, steps, acc_override=None):
  exp = f"exp_{label}"
  val = os.path.join(root, "logs", "gemma4_gsm8k", exp, "val")
  os.makedirs(val, exist_ok=True)
  ev = []
  for s, rows in steps.items():
    with open(os.path.join(val, f"{s}.jsonl"), "w") as f:
      for q, (a, b, o) in enumerate(rows):
        f.write(json.dumps({"qid": q, "acc": a, "fmt": b, "output": o}) + "\n")
    acc = sum(r[0] for r in rows) / Q
    if acc_override and s in acc_override:
      acc = acc_override[s]
    ev.append({"step": s, "acc_v2": acc, "acc_v2_wilson95": [0, 1], "delta_boot95": [0, 0], "acc_v1": 0.5,
               "boxed": sum(r[1] for r in rows) / Q, "wrong_to_right": 0, "right_to_wrong": 0, "truncated": 0,
               "tokens_mean": 1.0, "tokens_p95": 1})
  train = {str(s): {"acc_v2": 0.9, "acc_v1": 0.5, "boxed": 1.0, "mixed": 20, "all_correct": 100, "all_wrong": 8,
                    "actor/entropy_loss": 0.15, "response_length/mean": 500.0, "actor/grad_norm": 0.3 + (s == 5) * 3,
                    "actor/rollout_corr/k3_kl": 4e-4, "actor/rollout_corr/kl": 4e-4,
                    "actor/rollout_corr/rollout_is_max": 1.6, "actor/rollout_corr/rollout_is_ratio_fraction_high": 0.0,
                    "actor/rollout_corr/rollout_is_eff_sample_size": 0.999, "actor/pg_loss": -0.01,
                    "critic/score/mean": 0.9, "response_length/clip_ratio": 0.0, "timing_s/step": 30.0,
                    "interval_s": 30.1}
           for s in range(1, 21)}
  summary = {"exp": exp, "label": label, "finished": True, "driver_exit_code": 0, "eval": ev, "train": train,
             "checkpoints": {"saved_by_hook": [], "skipped_by_hook": [], "dirs": []},
             "generated_train_tokens": 20 * 500.0 * 2048,
             "timing": {"window": "steps 1-20 excluding eval/checkpoint steps", "interval_source": "coordinator_clock",
                        "internal_step_s": {"n": 19, "mean": 30.0, "median": 30.0, "p90": 30.0},
                        "actual_interval_s": {"n": 19, "mean": 30.1, "median": 30.1, "p90": 30.1},
                        "paired_gap_s": {"n": 19, "mean": 0.1, "median": 0.1, "p90": 0.1}, "paired_steps": 19,
                        "launch_to_finish_h": 0.2, "gpu_hours": 12.8, "internal_components_median_s": {},
                        "checkpoint_save_s": {}, "eval_s": {"20": 10.0}}}
  ed = os.path.join(root, "evidence", label)
  os.makedirs(ed, exist_ok=True)
  with open(os.path.join(ed, f"{label}_summary.json"), "w") as f:
    json.dump(summary, f)
  return exp


class AggregateTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = self.tmp.name

  def tearDown(self):
    self.tmp.cleanup()

  def load(self):
    labels = list(RUNS)
    exps = {lb: write_run(self.root, lb, RUNS[lb]) for lb in labels}
    ev = {lb: agg.load_eval(os.path.join(self.root, "logs", "gemma4_gsm8k", exps[lb])) for lb in labels}
    reports = {lb: agg.load_json(os.path.join(self.root, "evidence", lb, f"{lb}_summary.json")) for lb in labels}
    return labels, exps, ev, reports

  def test_eval_statistics(self):
    labels, _, ev, reports = self.load()
    steps, nq, rows = agg.eval_aggregate(labels, ev, reports, n_boot=2000)
    self.assertEqual((steps, nq), ([0, 20], Q))
    r0, r20 = rows
    # Step 0: seed 3 lost question 4 because it has no box; the outputs of questions 0-3 and 5 are identical.
    self.assertEqual((r0["right_in_all"], r0["wrong_in_all"], r0["split"]), (2, 3, 1))
    self.assertEqual(r0["split_box_status_differs"], 1)
    self.assertEqual(r0["identical_outputs_in_all"], 5)
    self.assertAlmostEqual(r0["acc_v2"]["mean"], (3 + 3 + 2) / 3 / Q)
    # Step 20: per-seed accuracy 5/6, 4/6, 4/6; changes from each seed's own step 0: +2/6, +1/6, +2/6.
    d = [2 / 6, 1 / 6, 2 / 6]
    for lb, x in zip(labels, d):
      self.assertAlmostEqual(r20["per_seed"][lb]["delta_vs_step0"], x)
    m, sd = sum(d) / 3, math.sqrt(sum((x - sum(d) / 3) ** 2 for x in d) / 2)
    self.assertAlmostEqual(r20["delta"]["mean"], m)
    self.assertAlmostEqual(r20["delta"]["sd"], sd)
    half = 4.3027 * sd / math.sqrt(3)
    self.assertAlmostEqual(r20["delta_seed_t95"][0], m - half)
    self.assertAlmostEqual(r20["delta_seed_t95"][1], m + half)
    lo, hi = r20["delta_question_boot95"]
    self.assertLessEqual(lo, m)
    self.assertGreaterEqual(hi, m)
    # Questions 0 and 5 split at step 20; question 5 differs in correctness but not in box status.
    self.assertEqual((r20["split"], r20["split_box_status_differs"], r20["identical_outputs_in_all"]), (2, 0, 0))

  def test_bootstrap_is_reproducible(self):
    labels, _, ev, reports = self.load()
    a = agg.eval_aggregate(labels, ev, reports, n_boot=2000)[2][1]["delta_question_boot95"]
    b = agg.eval_aggregate(labels, ev, reports, n_boot=2000)[2][1]["delta_question_boot95"]
    self.assertEqual(a, b)

  def test_report_mismatch_fails(self):
    labels = list(RUNS)
    exps = {lb: write_run(self.root, lb, RUNS[lb], acc_override={20: 0.5} if lb == "seed2" else None) for lb in labels}
    ev = {lb: agg.load_eval(os.path.join(self.root, "logs", "gemma4_gsm8k", exps[lb])) for lb in labels}
    reports = {lb: agg.load_json(os.path.join(self.root, "evidence", lb, f"{lb}_summary.json")) for lb in labels}
    with self.assertRaisesRegex(AssertionError, "seed2 step 20: dump accuracy"):
      agg.eval_aggregate(labels, ev, reports, n_boot=100)

  def test_question_ids_must_match(self):
    labels, exps, _, _ = self.load()
    with open(os.path.join(self.root, "logs", "gemma4_gsm8k", exps["seed3"], "val", "20.jsonl"), "a") as f:
      f.write(json.dumps({"qid": 99, "acc": 1, "fmt": 1, "output": "x"}) + "\n")
    ev = {lb: agg.load_eval(os.path.join(self.root, "logs", "gemma4_gsm8k", exps[lb])) for lb in labels}
    reports = {lb: agg.load_json(os.path.join(self.root, "evidence", lb, f"{lb}_summary.json")) for lb in labels}
    with self.assertRaisesRegex(AssertionError, "question IDs differ"):
      agg.eval_aggregate(labels, ev, reports, n_boot=100)

  def test_train_and_main(self):
    labels, exps, _, reports = self.load()
    tr = agg.train_aggregate(labels, reports)
    self.assertEqual(tr["seed1"]["grad_norm_above_clip"], 1)
    self.assertEqual(tr["seed1"]["grad_norm_top"][0][0], 5)
    self.assertEqual((tr["seed1"]["nonfinite_values"], tr["seed1"]["missing_values"]), (0, 0))
    self.assertEqual(agg.window_steps(reports["seed1"]), list(range(1, 20)))
    out = os.path.join(self.root, "out")
    argv = ["x", "--out", out, "--evidence", os.path.join(self.root, "evidence"), "--logs",
            os.path.join(self.root, "logs"), "--no-plots"] + sum((["--run", f"{lb}={exps[lb]}"] for lb in labels), [])
    buf = io.StringIO()
    old = sys.argv
    try:
      sys.argv = argv
      with contextlib.redirect_stdout(buf):
        agg.main()
    finally:
      sys.argv = old
    s = agg.load_json(os.path.join(out, "aggregate_summary.json"))
    self.assertEqual(s["eval_steps"], [0, 20])
    self.assertEqual(s["unfinished_or_failed"], [])
    text = buf.getvalue()
    self.assertIn("| 20 | 83.33 | 66.67 | 66.67 | 72.22 ± 9.62 | +27.78 |", text)
    self.assertIn("Endpoint (step 20) per seed", text)

  def test_wrong_exp_for_label_fails(self):
    labels, exps, _, _ = self.load()
    argv = ["x", "--out", os.path.join(self.root, "out"), "--evidence", os.path.join(self.root, "evidence"),
            "--logs", os.path.join(self.root, "logs"), "--no-plots", "--run", f"seed1={exps['seed2']}"]
    old = sys.argv
    try:
      sys.argv = argv
      with self.assertRaisesRegex(AssertionError, "report is for"):
        agg.main()
    finally:
      sys.argv = old


class ReportStepIntervalsTest(unittest.TestCase):

  def test_union_of_file_and_console(self):
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "metrics.jsonl")
      with open(p, "w") as f:
        for s, v in ((1, 31.0), (2, 29.5)):
          f.write(json.dumps({"step": s, "data": {"wallclock/interval_s": v}}) + "\n")
        f.write('{"step": 3, "data": {"wallclock/inter')  # partial trailing line (stale GCS-fuse read)
      steps = {1: {"wallclock/interval_s": 31.0}, 2: {"wallclock/interval_s": 29.5}, 3: {"wallclock/interval_s": 30.25}}
      iv, src, check = rep.step_intervals(steps, {}, "", p)
    self.assertEqual(src, "coordinator_clock")
    self.assertEqual(iv, {1: 31.0, 2: 29.5, 3: 30.25})
    self.assertEqual(check, {"metrics_jsonl": 2, "console": 3, "both": 2, "max_abs_diff_s": 0.0})

  def test_tqdm_fallback(self):
    log = ("Training Progress:   0%|          | 0/250 [00:00<?, ?it/s]\n"
           "Training Progress:   0%|          | 1/250 [01:39<6:50:00, 99.00s/it]\n"
           "Training Progress:   1%|          | 2/250 [02:20<5:00:00, 70.00s/it]\n")
    iv, src, check = rep.step_intervals({1: {}, 2: {}}, {}, log, None)
    self.assertEqual((src, iv, check), ("tqdm_elapsed", {1: 99, 2: 41}, {"tqdm_lines": 3}))


if __name__ == "__main__":
  unittest.main()
