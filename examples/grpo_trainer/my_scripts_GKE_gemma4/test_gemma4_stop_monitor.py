#!/usr/bin/env python3
"""Self-test for gemma4_stop_monitor.py (PR #12 section 4.1 stop rules). Standard library only.

Synthetic runs use small batches (--prompts 4: 4 groups x 8 = 32 rows per step, 3.125 pp per row; A6: 4 x 16) and a
200-question eval set. Cases:
  helpers   : consecutive-step counting (None and step gaps break a run), metric-line parsing (np / tensor wrappers)
  clean     : no rule triggers and no alert on a healthy run; a split metric line (unreadable k3_kl, no final key)
              raises an alert and leaves that step unevaluable instead of reading it as zero
  train_acc : matched rule triggers on the 3rd consecutive step < -10 pp; not with a recovery in between; not when the
              middle step's dump is partial (unevaluable, reported); not when prompt identity differs (alert)
  A5 offset : a persistent initial offset (b) alone never triggers; a further drop from step 6 does; steps 1-5 never do
  A6        : 16-completion groups compared with an 8-completion reference; a 15-row group is never evaluated
  missing   : > 10 % responses without </answer> on 3 consecutive steps triggers; 9.4 % does not
  k3_kl     : > 0.003 on 3 consecutive steps triggers; a missing metric line breaks the run
  nonfinite : a nan / inf actor or rollout_corr metric triggers at that step
  heldout   : -5 pp with CI < 0 triggers; -2 pp with an interval spanning 0 does not; -1.5 pp does not
  guard     : a first-forward record that is not PASS triggers
  live      : live mode with --act SIGTERMs only the dummy driver of the monitored experiment and writes
              stop_trigger.json; a finished run (LAUNCHER_RC) without triggers exits 0 with a final status
  real logs : if the committed driver logs exist, the metric rules replay A3b (LR 4e-6) to a k3_kl trigger at step 57
              and A0 / A1 / A2 / A3a to no trigger

Usage:  python3 test_gemma4_stop_monitor.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemma4_stop_monitor as M  # noqa: E402

P, N, EVAL = 4, 8, 200
FAILS = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + ("" if ok else f" :: {detail}"))
    if not ok:
        FAILS.append(name)


def write_train(
    root, step, n_correct, n=N, prompts=P, n_missing=0, rows=None, change_input_qid=None, drop_row_of_group=None
):
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    out = []
    i = 0
    for j in range(prompts):
        qid = 1000 * step + j
        for _ in range(n):
            acc = 1.0 if i < n_correct else 0.0
            ans = "" if i >= prompts * n - n_missing else "<answer>1</answer>"
            inp = f"user\nQuestion {qid}?\nmodel\n" + (" (changed)" if change_input_qid == j else "")
            out.append(
                {
                    "input": inp,
                    "output": f"<reasoning>r</reasoning>{ans}",
                    "gts": json.dumps([str(qid)] * 2),
                    "score": acc + 0.1,
                    "step": step,
                    "acc": acc,
                    "fmt": 1.0,
                    "qid": float(qid),
                }
            )
            i += 1
    if drop_row_of_group is not None:  # one group loses a row and another gains one: same total, broken groups
        g = [r for r in out if r["qid"] == float(1000 * step + drop_row_of_group)]
        out.remove(g[0])
        out.append(dict(out[0]))
    text = "".join(json.dumps(r) + "\n" for r in out[: rows if rows is not None else len(out)])
    with open(os.path.join(root, "train", f"{step}.jsonl"), "w") as f:
        f.write(text)


def write_val(root, step, accs):
    os.makedirs(os.path.join(root, "val"), exist_ok=True)
    with open(os.path.join(root, "val", f"{step}.jsonl"), "w") as f:
        for q, a in enumerate(accs):
            f.write(
                json.dumps(
                    {
                        "input": f"Eval {q}?",
                        "gts": json.dumps(["1", "1"]),
                        "acc": float(a),
                        "fmt": 1.0,
                        "qid": float(q),
                        "output": "x",
                    }
                )
                + "\n"
            )


def write_log(path, metrics_by_step, tail=""):
    with open(path, "w") as f:
        f.write("+ bash run_gemma4_e2b_rl05_parity.sh\nTraining Progress:   0%|          | 0/100\r")
        for k, kv in sorted(metrics_by_step.items()):
            parts = " - ".join(f"{key}:{val}" for key, val in kv.items())
            f.write(f"(TaskRunner pid=123, ip=10.0.0.1) step:{k} - {parts}\n")
        f.write(tail)


def healthy_metrics(steps, k3=0.0008):
    return {
        k: {
            "actor/entropy": "np.float64(0.25)",
            "actor/grad_norm": 0.2,
            "actor/pg_loss": -0.01,
            "rollout_corr/k3_kl": f"np.float64({k3})",
            "actor/lr": 2e-06,
            "critic/score/mean": 0.4,
            "perf/throughput": 800.0,
        }
        for k in steps
    }


def make_run(
    tmp,
    name,
    ref_correct,
    arm_correct,
    steps=10,
    n=N,
    missing=None,
    metrics=None,
    partial=(),
    changed=(),
    broken=(),
    val=None,
    log_tail="",
):
    """ref_correct / arm_correct: step -> number of correct rows (default 16 = 50 %)."""
    ref, arm = os.path.join(tmp, name, "ref"), os.path.join(tmp, name, "arm")
    for k in range(1, steps + 1):
        write_train(ref, k, ref_correct.get(k, 16))
        write_train(
            arm,
            k,
            arm_correct.get(k, 16 * n // N),
            n=n,
            n_missing=(missing or {}).get(k, 0),
            rows=(P * n - 1) if k in partial else None,
            change_input_qid=1 if k in changed else None,
            drop_row_of_group=2 if k in broken else None,
        )
    if val:
        for s, accs in val.items():
            write_val(arm, s, accs)
    log = os.path.join(tmp, name, "driver.log")
    write_log(log, metrics if metrics is not None else healthy_metrics(range(1, steps + 1)), log_tail)
    return ref, arm, log


def replay(tmp, name, arm_label, ref, arm, log, n=None, guard_dir=None):
    out = os.path.join(tmp, name, "replay.json")
    argv = [
        "--exp",
        name,
        "--arm",
        arm_label,
        "--replay",
        "--dump-dir",
        arm,
        "--log",
        log,
        "--ref-dump-dir",
        ref,
        "--prompts",
        str(P),
        "--eval-rows",
        str(EVAL),
        "--guard-dir",
        guard_dir or os.path.join(tmp, name, "guard"),
        "--out-json",
        out,
    ]
    if n:
        argv += ["--group-size", str(n)]
    devnull = open(os.devnull, "w")
    old = sys.stdout
    sys.stdout = devnull
    try:
        rc = M.main(argv)
    finally:
        sys.stdout = old
    with open(out) as f:
        st = json.load(f)
    return rc, st


def trig(st):
    return [(t["rule"], t["step"]) for t in st["triggers"]]


def test_helpers():
    fc = M.first_consecutive
    check("consecutive: 3 in a row", fc({1: True, 2: True, 3: True}) == 3)
    check("consecutive: broken by False", fc({1: True, 2: True, 3: False, 4: True, 5: True}) is None)
    check("consecutive: broken by None", fc({1: True, 2: None, 3: True, 4: True}) is None)
    check("consecutive: broken by a step gap", fc({1: True, 2: True, 4: True}) is None)
    check("consecutive: later run", fc({1: True, 2: False, 3: True, 4: True, 5: True, 6: True}) == 5)
    kv = M.parse_metric_line(
        "actor/grad_norm:np.float64(nan) - rollout_corr/k3_kl:tensor(inf, device='cuda:0') - "
        "actor/lr:2e-06 - name:abc - critic/score/mean:np.float32(0.25)"
    )
    ok = (
        str(kv.get("actor/grad_norm")) == "nan"
        and kv.get("rollout_corr/k3_kl") == float("inf")
        and kv.get("actor/lr") == 2e-6
        and "name" not in kv
        and kv.get("critic/score/mean") == 0.25
    )
    check("parse: np / tensor wrappers, nan / inf kept, strings skipped", ok, str(kv))
    d, lo, hi = M.paired_bootstrap_pp([-1.0] * 100 + [0.0] * 4900, B=200, seed=1)
    check("bootstrap: exact point estimate -2.0 pp", d == -2.0 and lo <= d <= hi, f"{d} [{lo}, {hi}]")


def test_rules(tmp):
    ref, arm, log = make_run(tmp, "clean", {}, {})
    rc, st = replay(tmp, "clean", "A4", ref, arm, log)
    check(
        "clean A4 run: no trigger, no alert",
        rc == 0 and not st["triggers"] and st["last_complete_train_dump"] == 10 and not st["alerts"],
        f"rc={rc} {trig(st)} {st['alerts']}",
    )

    met = healthy_metrics(range(1, 11))
    del met[4]["perf/throughput"]
    met[4]["rollout_corr/k3_kl"] = "0.0008\x1b[36m(TaskRunner pid=1, ip=x)\x1b[0m"  # glued line: value unreadable
    ref, arm, log = make_run(tmp, "split_line", {}, {}, metrics=met)
    rc, st = replay(tmp, "split_line", "A4", ref, arm, log)
    check(
        "split metric line: alert, k3_kl unevaluable at that step (not zero)",
        any(a["key"] == "metric_incomplete_4" for a in st["alerts"])
        and st["per_step"]["4"]["k3_kl"] is None
        and not st["triggers"],
        f"{st['alerts']} {st['per_step'].get('4')}",
    )

    drop = {4: 12, 5: 12, 6: 12}  # 37.5 % vs 50 %: -12.5 pp
    ref, arm, log = make_run(tmp, "acc_drop", {}, drop)
    rc, st = replay(tmp, "acc_drop", "A4", ref, arm, log)
    check("A4 train_acc: triggers at the 3rd consecutive step", trig(st) == [("train_acc", 6)], f"{trig(st)}")

    ref, arm, log = make_run(tmp, "acc_recover", {}, {4: 12, 5: 12, 6: 14, 7: 12})
    rc, st = replay(tmp, "acc_recover", "A4", ref, arm, log)
    check("A4 train_acc: -6.25 pp in between breaks the run", not st["triggers"], f"{trig(st)}")

    ref, arm, log = make_run(tmp, "acc_partial", {}, drop, partial=(5,))
    rc, st = replay(tmp, "acc_partial", "A4", ref, arm, log)
    check(
        "A4 train_acc: partial middle dump is unevaluable, no trigger",
        not st["triggers"] and "5" in st["not_complete"]["train"] and st["per_step"]["5"]["acc"] is None,
        f"{trig(st)} {st['not_complete']}",
    )

    ref, arm, log = make_run(tmp, "acc_identity", {}, drop, changed=(5,))
    rc, st = replay(tmp, "acc_identity", "A4", ref, arm, log)
    check(
        "A4 train_acc: prompt identity mismatch -> unevaluable + alert",
        not st["triggers"] and any(a["key"] == "identity_5" for a in st["alerts"]),
        f"{trig(st)} {st['alerts']}",
    )

    off = {k: 10 for k in range(1, 11)}  # 31.25 % vs 50 %: -18.75 pp from step 1
    ref, arm, log = make_run(tmp, "a5_offset", {}, off)
    rc, st = replay(tmp, "a5_offset", "A5", ref, arm, log)
    check(
        "A5: persistent initial offset alone never triggers",
        not st["triggers"] and abs(st["a5_baseline_pp"] + 18.75) < 1e-9,
        f"{trig(st)} b={st['a5_baseline_pp']}",
    )
    rc, st = replay(tmp, "a5_offset", "A4", ref, arm, log)
    check(
        "same data under the matched rule triggers at step 3 (contrast)", trig(st) == [("train_acc", 3)], f"{trig(st)}"
    )

    off2 = dict(off)
    off2.update({7: 3, 8: 3, 9: 3})  # 9.375 %: d = -40.6, d - b = -21.9
    ref, arm, log = make_run(tmp, "a5_drop", {}, off2)
    rc, st = replay(tmp, "a5_drop", "A5", ref, arm, log)
    check("A5: further drop from step 6 triggers on the 3rd step", trig(st) == [("train_acc", 9)], f"{trig(st)}")

    ref, arm, log = make_run(tmp, "a5_early", {}, {k: 3 for k in range(1, 6)})
    rc, st = replay(tmp, "a5_early", "A5", ref, arm, log)
    check("A5: drops in steps 1-5 only never trigger", not st["triggers"], f"{trig(st)}")

    ref, arm, log = make_run(tmp, "a6_ok", {}, {}, n=16)
    rc, st = replay(tmp, "a6_ok", "A6", ref, arm, log)
    check(
        "A6: 4 x 16 groups vs a 4 x 8 reference are evaluated",
        not st["triggers"] and st["per_step"]["3"]["d"] == 0.0,
        f"{trig(st)} {st['per_step'].get('3')}",
    )
    rc, st = replay(tmp, "a6_ok", "A4", ref, arm, log, n=8)
    check(
        "A6 dumps read as n=8 are never complete",
        not st["triggers"] and st["last_complete_train_dump"] is None,
        f"{trig(st)}",
    )
    ref, arm, log = make_run(tmp, "a6_drop", {}, {4: 24, 5: 24, 6: 24}, n=16)  # 37.5 %
    rc, st = replay(tmp, "a6_drop", "A6", ref, arm, log)
    check("A6: -12.5 pp on 3 consecutive steps triggers", trig(st) == [("train_acc", 6)], f"{trig(st)}")
    ref, arm, log = make_run(tmp, "a6_broken", {}, {4: 24, 5: 24, 6: 24}, n=16, broken=(5,))
    rc, st = replay(tmp, "a6_broken", "A6", ref, arm, log)
    check(
        "A6: a 15-row group makes the step invalid (alert, unevaluable)",
        not st["triggers"] and any(a["key"] == "train_invalid_5" for a in st["alerts"]),
        f"{trig(st)} {st['alerts']}",
    )

    ref, arm, log = make_run(tmp, "miss", {}, {}, missing={2: 4, 3: 4, 4: 4})
    rc, st = replay(tmp, "miss", "A4", ref, arm, log)
    check("missing </answer> 12.5 % x3 triggers", trig(st) == [("missing_answer", 4)], f"{trig(st)}")
    ref, arm, log = make_run(tmp, "miss_low", {}, {}, missing={2: 3, 3: 3, 4: 3})
    rc, st = replay(tmp, "miss_low", "A4", ref, arm, log)
    check("missing </answer> 9.4 % x3 does not trigger", not st["triggers"], f"{trig(st)}")

    met = healthy_metrics(range(1, 11))
    for k in (3, 4, 5):
        met[k]["rollout_corr/k3_kl"] = "np.float64(0.004)"
    ref, arm, log = make_run(tmp, "k3", {}, {}, metrics=met)
    rc, st = replay(tmp, "k3", "A4", ref, arm, log)
    check("k3_kl > 0.003 x3 triggers", trig(st) == [("k3_kl", 5)], f"{trig(st)}")
    del met[4]
    ref, arm, log = make_run(tmp, "k3_gap", {}, {}, metrics=met)
    rc, st = replay(tmp, "k3_gap", "A4", ref, arm, log)
    check("k3_kl: a missing metric line breaks the run", not st["triggers"], f"{trig(st)}")

    met = healthy_metrics(range(1, 11))
    met[3]["actor/grad_norm"] = "np.float64(nan)"
    met[6]["rollout_corr/chi2_token"] = "inf"
    ref, arm, log = make_run(tmp, "nonfinite", {}, {}, metrics=met)
    rc, st = replay(tmp, "nonfinite", "A4", ref, arm, log)
    check("nonfinite metric triggers at its step", trig(st) == [("nonfinite", 3)], f"{trig(st)}")

    base = [1] * 100 + [0] * 100
    lose10 = [0] * 10 + [1] * 90 + [0] * 100  # -5 pp
    mixed = [0] * 34 + [1] * 66 + [1] * 30 + [0] * 70  # gained 30, lost 34: -2 pp, CI spans 0
    small = [0] * 3 + [1] * 97 + [0] * 100  # -1.5 pp
    ref, arm, log = make_run(tmp, "heldout", {}, {}, val={0: base, 20: small, 40: lose10})
    rc, st = replay(tmp, "heldout", "A4", ref, arm, log)
    h = st["heldout"]
    check(
        "heldout: -5 pp with CI < 0 triggers at that eval; -1.5 pp does not",
        trig(st) == [("heldout", 40)] and abs(h["40"]["delta_pp"] + 5) < 1e-9 and abs(h["20"]["delta_pp"] + 1.5) < 1e-9,
        f"{trig(st)} {h}",
    )
    ref, arm, log = make_run(tmp, "heldout_ci", {}, {}, val={0: base, 20: mixed})
    rc, st = replay(tmp, "heldout_ci", "A4", ref, arm, log)
    h = st["heldout"]["20"]
    check(
        "heldout: -2 pp but interval spans 0 -> no trigger",
        not st["triggers"] and h["delta_pp"] == -2.0 and h["ci95_pp"][1] >= 0,
        f"{trig(st)} {h}",
    )

    ref, arm, log = make_run(tmp, "guard", {}, {})
    gd = os.path.join(tmp, "guard", "guard")
    os.makedirs(gd, exist_ok=True)
    for r in range(2):
        with open(os.path.join(gd, f"first_forward_actor_rank{r:04d}.json"), "w") as f:
            json.dump(
                {"rank": r, "status": "PASS" if r == 0 else "FAIL", "restored": [] if r == 0 else ["inv_freq"]}, f
            )
    rc, st = replay(tmp, "guard", "A4", ref, arm, log, guard_dir=gd)
    check("guard: a failing first-forward record triggers", trig(st) == [("guard", None)], f"{trig(st)}")


def dummy(exp):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(600)",
            "-m",
            "verl.trainer.main_ppo",
            f"trainer.experiment_name={exp}",
        ]
    )


def test_live(tmp):
    target, other = dummy("exp_live_target"), dummy("exp_live_other")
    time.sleep(0.5)
    try:
        pids = M.find_driver_pids("exp_live_target")
        check(
            "process match: exact experiment only",
            pids == [target.pid],
            f"{pids} target={target.pid} other={other.pid}",
        )
        ref, arm, log = make_run(tmp, "live", {}, {4: 12, 5: 12, 6: 12})
        out = os.path.join(tmp, "live", "monitor")
        p = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "gemma4_stop_monitor.py"),
                "--exp",
                "exp_live_target",
                "--arm",
                "A4",
                "--act",
                "--dump-dir",
                arm,
                "--log",
                log,
                "--ref-dump-dir",
                ref,
                "--prompts",
                str(P),
                "--eval-rows",
                str(EVAL),
                "--guard-dir",
                os.path.join(tmp, "live", "g"),
                "--out-dir",
                out,
                "--poll-s",
                "0.2",
                "--final-wait-s",
                "0",
                "--kill-wait-s",
                "10",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        time.sleep(0.5)
        with open(os.path.join(out, "stop_trigger.json")) as f:
            ev = json.load(f)
        ok = (
            p.returncode == 3
            and target.poll() is not None
            and other.poll() is None
            and ev["rule"]["rule"] == "train_acc"
            and ev["rule"]["step"] == 6
            and ev["stop"]["pids"] == [target.pid]
            and ev["stop"]["exited"] is True
            and ev["arm_status"] == "early-stopped/incomplete"
            and ev["last_completed_before_stop"]["last_complete_train_dump"] == 10
        )
        check(
            "live --act: SIGTERM only the monitored driver, evidence written",
            ok,
            f"rc={p.returncode} target={target.poll()} other={other.poll()} {json.dumps(ev)[:400]} {p.stderr[-300:]}",
        )
        with open(os.path.join(out, "status.json")) as f:
            st = json.load(f)
        check(
            "live: final status marks the arm early-stopped",
            st.get("arm_status") == "early-stopped/incomplete",
            str(st.get("arm_status")),
        )
    finally:
        for proc in (target, other):
            if proc.poll() is None:
                proc.kill()

    ref, arm, log = make_run(tmp, "live_done", {}, {}, log_tail="LAUNCHER_RC=0\n")
    out = os.path.join(tmp, "live_done", "monitor")
    p = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "gemma4_stop_monitor.py"),
            "--exp",
            "exp_live_done",
            "--arm",
            "A4",
            "--act",
            "--dump-dir",
            arm,
            "--log",
            log,
            "--ref-dump-dir",
            ref,
            "--prompts",
            str(P),
            "--eval-rows",
            str(EVAL),
            "--guard-dir",
            os.path.join(tmp, "live_done", "g"),
            "--out-dir",
            out,
            "--poll-s",
            "0.2",
            "--final-wait-s",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    with open(os.path.join(out, "status.json")) as f:
        st = json.load(f)
    check(
        "live: finished run without triggers exits 0 with a final status",
        p.returncode == 0
        and st.get("final")
        and st.get("arm_status") == "finished"
        and not os.path.exists(os.path.join(out, "stop_trigger.json")),
        f"rc={p.returncode} {p.stderr[-300:]}",
    )


def test_real_logs():
    logs = os.path.join(HERE, "logs")
    runs = {
        "gemma4_e2b_rl05fp32_a3lr4e6_16n64g_s100_seed1_1023_1116": [("k3_kl", 57)],
        "gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753": [],
        "gemma4_e2b_rl05fp32_16n64g_s200_seed1_1022_2019": [],
        "gemma4_e2b_rl05fp32_a1fmt0_16n64g_s100_seed1_1023_0058": [],
        "gemma4_e2b_rl05fp32_a2wrap1_16n64g_s100_seed1_1023_0423": [],
    }
    for exp, want in runs.items():
        path = os.path.join(logs, exp + ".log")
        if not os.path.exists(path):
            print(f"SKIP real log {exp} (not present)")
            continue
        tail = M.LogTail(path)
        metrics = {}
        for line in tail.read_new():
            m = M.STEP_RE.search(line)
            if m:
                metrics.setdefault(int(m.group(1)), {}).update(M.parse_metric_line(m.group(2)))
        st = {"train": {}, "ref": {}, "metrics": metrics, "val": {}, "guard_failures": [], "log_guard_failed": False}
        ev = M.evaluate(st, "matched", 100)
        got = [(t["rule"], t["step"]) for t in ev["triggers"]]
        check(
            f"real log metric rules {exp[-28:]}: {want or 'no trigger'}",
            got == want and ev["last_step"] == 100,
            f"{got} last={ev['last_step']}",
        )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="g4_stop_monitor_test_")
    test_helpers()
    test_rules(tmp)
    test_live(tmp)
    test_real_logs()
    print(f"{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
