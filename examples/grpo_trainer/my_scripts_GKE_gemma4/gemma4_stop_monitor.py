#!/usr/bin/env python3
"""Predeclared early-stop monitor for the PR #12 Gemma-4 screens (RECIPE_OPTIMIZATION_PLAN.md section 4.1).

Runs beside one training run on the Ray head pod, reads only complete evidence and, with --act, terminates that run's
driver (and nothing else) as soon as an applicable rule triggers. Standard library only.

Rules. A "3 consecutive" rule counts consecutive *evaluated* training steps. A step whose evidence is missing,
incomplete or unreadable is unevaluable: it breaks the streak and is reported as an observability alert, never as a
pass.

  nonfinite       any non-finite loss / gradient / log-prob metric in a step's metric line (keys under actor/ or
                  rollout_corr/, or containing "loss")                                -> stop at that step
  guard           a first-forward [GEMMA4-ROPE] record that is not PASS or restored a buffer, or
                  "[GEMMA4-ROPE] guard FAILED" in the driver log                        -> stop
  k3_kl           rollout_corr/k3_kl > 0.003 on 3 consecutive steps
  missing_answer  share of responses without "</answer>" > 10 % on 3 consecutive steps
  train_acc       matched (A4/A6): d_k = acc_arm(k) - acc_ref(k) < -10 pp on 3 consecutive steps, where both runs
                  trained on the same 256 prompt instances at step k (qid -> input / ground truth identity checked)
                  offset (A5): b = median(d_1..d_5), frozen once those five steps are evaluated; from step 6,
                  d_k < -10 pp and d_k - b < -10 pp on 3 consecutive steps
  heldout         at a complete eval s > 0: acc(s) - acc(0) <= -2 pp and the question-paired bootstrap 95 % upper bound
                  of that difference < 0 (2,000 resamples)

Evidence:
  training dumps  <dump-dir>/train/<k>.jsonl (verl trainer.rollout_data_dir). Complete only with exactly
                  prompts x group_size rows, `prompts` qid groups of exactly group_size rows, one input and one ground
                  truth per group, step field == k, acc in [0, 1] and the required fields. A partial file is "not yet
                  complete", never a low-accuracy batch or a missing-answer event.
  eval dumps      <dump-dir>/val/<s>.jsonl: exactly --eval-rows unique qids, same qid -> input / ground truth as step 0.
  metric lines    "step:<k> - key:value - ..." lines of the console logger in the driver log, read incrementally.
                  The monitor records when each step's dump and metric line first become visible, so metric latency is
                  measured, not assumed. A metric line missing while its dump is complete raises an alert after
                  --metric-lag-alert-s.

Outputs (in --out-dir): status.json (rewritten when it changes, at least every 5 min), monitor.jsonl (one event per
line), stop_trigger.json (only when a rule triggers: rule, step, values, threshold, times, stop request, PIDs, last
completed update before and after the stop, `ray status` GPU lines).

Usage on the head pod (live; --act enables termination):
    nohup python3 gemma4_stop_monitor.py --exp EXP --arm A4 --act > /dev/null 2>&1 &
Replay over finished runs (never acts):
    python3 gemma4_stop_monitor.py --exp EXP --arm A4 --replay [--max-step 100] [--out-json x.json]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

LOGS = "/workspace/meta-RL/logs"
A3A = "gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753"

PROMPTS_PER_STEP = 256
EVAL_ROWS = 5000
K3_MAX = 0.003
MISSING_ANSWER_MAX_PCT = 10.0
ACC_DROP_PP = 10.0
HELDOUT_DROP_PP = 2.0
CONSECUTIVE = 3
A5_BASELINE_STEPS = 5
BOOTSTRAP = 2000
ANSWER_END = "</answer>"

ARMS = {  # arm -> (rollout.n, training-accuracy rule)
    "A4": (8, "matched"),
    "A5": (8, "offset"),
    "A6": (16, "matched"),
}

REQUIRED_TRAIN = ("input", "output", "gts", "score", "step", "acc", "qid")
REQUIRED_VAL = ("input", "gts", "acc", "qid")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
WRAPPED_RE = re.compile(r"^[A-Za-z_][\w.]*\(\s*([^,()]*?)\s*(?:,.*)?\)$")  # np.float64(x) / tensor(x, device=...)
GUARD_FAILED = "[GEMMA4-ROPE] guard FAILED"
LAST_KEY = "perf/throughput:"  # last key of verl's per-step console line at the runtime pin


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8", "replace")).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- evidence readers


class NotReady(Exception):
    """Evidence that is not (yet) complete: partial write, missing file, unreadable."""


class Invalid(Exception):
    """Evidence that is complete-looking but violates the batch / eval contract."""


def _read_jsonl(path: str) -> list:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise NotReady(f"unreadable: {e.strerror or e}") from None
    if not data:
        raise NotReady("empty file")
    if not data.endswith(b"\n"):
        raise NotReady("no trailing newline (partial write)")
    rows = []
    for i, line in enumerate(data.split(b"\n")):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                raise NotReady(f"line {i + 1} is not JSON (partial write)") from None
    return rows


def _acc(r, where: str) -> float:
    try:
        a = float(r["acc"])
    except (TypeError, ValueError):
        raise Invalid(f"{where}: acc {r.get('acc')!r} is not a number") from None
    if not (math.isfinite(a) and 0.0 <= a <= 1.0):
        raise Invalid(f"{where}: acc {a!r} outside [0, 1]")
    return a


def load_train_dump(path: str, step: int, group_size: int, prompts: int = PROMPTS_PER_STEP) -> dict:
    """Validate one training dump; raise NotReady / Invalid, else return its summary."""
    rows = _read_jsonl(path)
    want = prompts * group_size
    if len(rows) < want:
        raise NotReady(f"{len(rows)} rows, expected {want}")
    if len(rows) > want:
        raise Invalid(f"{len(rows)} rows, expected {want}")
    groups: dict = {}
    n_correct = n_missing = 0
    for r in rows:
        miss = [k for k in REQUIRED_TRAIN if k not in r]
        if miss:
            raise Invalid(f"row without {miss}")
        if str(r["step"]) != str(step):
            raise Invalid(f"step field {r['step']!r} != {step}")
        n_correct += _acc(r, f"step {step}")
        n_missing += ANSWER_END not in str(r["output"])
        groups.setdefault(str(r["qid"]), []).append(r)
    if len(groups) != prompts:
        raise Invalid(f"{len(groups)} prompt groups, expected {prompts}")
    ident = {}
    for q, g in groups.items():
        if len(g) != group_size:
            raise Invalid(f"qid {q}: {len(g)} rows, expected {group_size}")
        ins, gts = {_h(r["input"]) for r in g}, {_h(r["gts"]) for r in g}
        if len(ins) != 1 or len(gts) != 1:
            raise Invalid(f"qid {q} mixes inputs or ground truths inside its group")
        ident[q] = (ins.pop(), gts.pop())
    return {
        "rows": len(rows),
        "groups": len(groups),
        "group_size": group_size,
        "acc_pct": 100.0 * n_correct / len(rows),
        "missing_answer_pct": 100.0 * n_missing / len(rows),
        "identity": ident,
    }


def load_val_dump(path: str, n_rows: int = EVAL_ROWS) -> dict:
    rows = _read_jsonl(path)
    if len(rows) < n_rows:
        raise NotReady(f"{len(rows)} rows, expected {n_rows}")
    if len(rows) > n_rows:
        raise Invalid(f"{len(rows)} rows, expected {n_rows}")
    by_q = {}
    for r in rows:
        miss = [k for k in REQUIRED_VAL if k not in r]
        if miss:
            raise Invalid(f"row without {miss}")
        by_q[str(r["qid"])] = (_acc(r, "eval"), _h(r["input"]), _h(r["gts"]))
    if len(by_q) != n_rows:
        raise Invalid(f"{n_rows - len(by_q)} duplicate qids")
    return {"rows": n_rows, "acc_pct": 100.0 * sum(v[0] for v in by_q.values()) / n_rows, "by_qid": by_q}


def parse_metric_line(text: str) -> dict:
    """Parse the part after 'step:<k> - '. Non-numeric values are skipped; nan / inf are kept."""
    kv = {}
    for part in text.split(" - "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        v = v.strip()
        m = WRAPPED_RE.match(v)
        if m:
            v = m.group(1)
        try:
            kv[k.strip()] = float(v)
        except ValueError:
            pass
    return kv


def watched_metric(key: str) -> bool:
    return key.startswith("actor/") or key.startswith("rollout_corr/") or "loss" in key


class LogTail:
    """Incremental reader of a growing log; yields complete lines only."""

    def __init__(self, path: str):
        self.path, self.offset, self.rest = path, 0, b""

    def read_new(self) -> list:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.offset:  # truncated / replaced
            self.offset, self.rest = 0, b""
        if size == self.offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read(size - self.offset)
        self.offset += len(chunk)
        parts = (self.rest + chunk).split(b"\n")
        self.rest = parts.pop()
        return [p.decode("utf-8", "replace") for p in parts]


# --------------------------------------------------------------------------------------------------- statistics


def paired_bootstrap_pp(diffs: list, B: int = BOOTSTRAP, seed: int = 0) -> tuple:
    """Mean of per-question differences (in pp) and its percentile bootstrap 95 % interval (linear interpolation,
    numpy's default). The point estimate is 100 * sum / n, exact for 0/1 accuracies (e.g. -100 / 5000 -> -2.0)."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(100.0 * sum(rng.choices(diffs, k=n)) / n for _ in range(B))

    def pct(q):
        pos = q * (B - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, B - 1)
        return means[lo] + (means[hi] - means[lo]) * (pos - lo)

    return 100.0 * sum(diffs) / n, pct(0.025), pct(0.975)


def first_consecutive(series: dict, need: int = CONSECUTIVE):
    """series: step -> True / False / None (unevaluable). Return the step completing the first run of `need` True
    values on consecutive step numbers; None breaks a run, and so does a gap in step numbers."""
    run, prev = 0, None
    for k in sorted(series):
        v = series[k]
        run = run + 1 if (v is True and prev is not None and k == prev + 1 and run > 0) else (1 if v is True else 0)
        prev = k
        if run >= need:
            return k
    return None


def current_run(series: dict) -> int:
    run, prev = 0, None
    for k in sorted(series):
        v = series[k]
        run = run + 1 if (v is True and prev is not None and k == prev + 1 and run > 0) else (1 if v is True else 0)
        prev = k
    return run


# ------------------------------------------------------------------------------------------------- rule engine


def evaluate(state: dict, acc_rule: str, max_step: int) -> dict:
    """Pure evaluation of every rule over the evidence collected so far.

    state: train {k: summary|None}, ref {k: summary|None}, metrics {k: dict}, val {s: summary},
           guard_failures [str], log_guard_failed bool.
    Returns {"triggers": [...sorted by step...], "series": {...}, "unevaluable": {...}, "a5_baseline": b|None}.
    """
    train, ref, metrics, val = state["train"], state["ref"], state["metrics"], state["val"]
    last = max([k for k in train if train[k]] + [k for k in metrics if k >= 1] + [0])
    last = min(last, max_step)
    steps = range(1, last + 1)
    trig, unev = [], {"k3_kl": [], "missing_answer": [], "train_acc": [], "metrics": []}

    # non-finite metrics: immediate
    nonfinite = {}
    for k in sorted(metrics):
        if 1 <= k <= max_step:
            bad = sorted(key for key, v in metrics[k].items() if watched_metric(key) and not math.isfinite(v))
            if bad:
                nonfinite[k] = bad
    if nonfinite:
        k = min(nonfinite)
        trig.append(
            {
                "rule": "nonfinite",
                "step": k,
                "values": {key: str(metrics[k][key]) for key in nonfinite[k]},
                "threshold": "any non-finite actor/*, rollout_corr/* or *loss* metric",
            }
        )

    # guard
    if state.get("guard_failures") or state.get("log_guard_failed"):
        trig.append(
            {
                "rule": "guard",
                "step": None,
                "values": {
                    "first_forward_failures": state.get("guard_failures", []),
                    "driver_log_guard_failed": bool(state.get("log_guard_failed")),
                },
                "threshold": "any first-forward record not PASS or restored, or guard FAILED in the driver log",
            }
        )

    # k3_kl
    k3 = {}
    for k in steps:
        v = metrics.get(k, {}).get("rollout_corr/k3_kl")
        if v is None or not math.isfinite(v):
            k3[k] = None
            unev["k3_kl"].append(k)
        else:
            k3[k] = v > K3_MAX
    s = first_consecutive(k3)
    if s:
        trig.append(
            {
                "rule": "k3_kl",
                "step": s,
                "threshold": f"> {K3_MAX} on {CONSECUTIVE} consecutive steps",
                "values": {str(j): metrics[j]["rollout_corr/k3_kl"] for j in range(s - CONSECUTIVE + 1, s + 1)},
            }
        )

    # missing </answer>
    miss = {}
    for k in steps:
        t = train.get(k)
        if t:
            miss[k] = t["missing_answer_pct"] > MISSING_ANSWER_MAX_PCT
        else:
            miss[k] = None
            unev["missing_answer"].append(k)
    s = first_consecutive(miss)
    if s:
        trig.append(
            {
                "rule": "missing_answer",
                "step": s,
                "threshold": f"> {MISSING_ANSWER_MAX_PCT}% of responses on {CONSECUTIVE} consecutive steps",
                "values": {str(j): train[j]["missing_answer_pct"] for j in range(s - CONSECUTIVE + 1, s + 1)},
            }
        )

    # training accuracy vs the reference on identical prompt instances
    d, d_reason = {}, {}
    for k in steps:
        t, r = train.get(k), ref.get(k)
        if not t or not r:
            d[k], d_reason[k] = None, "arm dump not complete" if not t else "reference dump not complete"
        elif t["identity"] != r["identity"]:
            d[k], d_reason[k] = None, "prompt identity differs from the reference"
        else:
            d[k] = t["acc_pct"] - r["acc_pct"]
    unev["train_acc"] = [k for k in steps if d[k] is None]
    b = None
    if acc_rule == "matched":
        acc_series = {k: (None if d[k] is None else d[k] < -ACC_DROP_PP) for k in steps}
        thr = f"arm - reference < -{ACC_DROP_PP} pp on {CONSECUTIVE} consecutive steps"
    else:
        base = [d.get(k) for k in range(1, A5_BASELINE_STEPS + 1)]
        if all(x is not None for x in base):
            b = statistics.median(base)
        acc_series = {}
        for k in steps:
            if k <= A5_BASELINE_STEPS:
                acc_series[k] = False  # rule starts at step 6
            elif b is None or d[k] is None:
                acc_series[k] = None
            else:
                acc_series[k] = d[k] < -ACC_DROP_PP and d[k] - b < -ACC_DROP_PP
        thr = (
            f"from step {A5_BASELINE_STEPS + 1}: d < -{ACC_DROP_PP} pp and d - b < -{ACC_DROP_PP} pp on "
            f"{CONSECUTIVE} consecutive steps, b = median(d_1..d_{A5_BASELINE_STEPS})"
        )
    s = first_consecutive(acc_series)
    if s:
        vals = {
            str(j): {"arm": train[j]["acc_pct"], "ref": ref[j]["acc_pct"], "d": d[j]}
            for j in range(s - CONSECUTIVE + 1, s + 1)
        }
        if b is not None:
            vals["b"] = b
        trig.append({"rule": "train_acc", "step": s, "threshold": thr, "values": vals})

    # held-out vs own step 0 (complete evals never change, so each comparison is computed once and cached)
    heldout = {}
    cache = state.setdefault("heldout_cache", {})
    v0 = val.get(0)
    for sv in sorted(val):
        if sv == 0 or sv > max_step or not v0 or not val[sv]:
            continue
        if sv not in cache:
            a, c = v0["by_qid"], val[sv]["by_qid"]
            if set(a) != set(c) or any(a[q][1:] != c[q][1:] for q in a):
                cache[sv] = {"error": "eval qid -> input / ground truth differs from step 0"}
            else:
                qs = sorted(a)
                delta, lo, hi = paired_bootstrap_pp([c[q][0] - a[q][0] for q in qs], seed=sv)
                cache[sv] = {
                    "acc0": v0["acc_pct"],
                    "acc": val[sv]["acc_pct"],
                    "delta_pp": delta,
                    "ci95_pp": [lo, hi],
                    "n": len(qs),
                }
        heldout[sv] = cache[sv]
        h = cache[sv]
        if (
            "error" not in h
            and h["delta_pp"] <= -HELDOUT_DROP_PP + 1e-9
            and h["ci95_pp"][1] < 0
            and not any(t["rule"] == "heldout" for t in trig)
        ):
            trig.append(
                {
                    "rule": "heldout",
                    "step": sv,
                    "values": h,
                    "threshold": f"<= -{HELDOUT_DROP_PP} pp vs own step 0 and paired 95% upper bound < 0",
                }
            )

    trig.sort(key=lambda t: (t["step"] is not None, t["step"] or 0))
    return {
        "triggers": trig,
        "last_step": last,
        "a5_baseline": b,
        "heldout": heldout,
        "series": {"k3_kl": k3, "missing_answer": miss, "train_acc": acc_series, "d": d},
        "streaks": {
            "k3_kl": current_run(k3),
            "missing_answer": current_run(miss),
            "train_acc": current_run(acc_series),
        },
        "unevaluable": unev,
        "d_reason": {k: v for k, v in d_reason.items()},
    }


# ---------------------------------------------------------------------------------------------- process control


def find_driver_pids(exp: str, proc: str = "/proc") -> list:
    """PIDs whose argv contains both 'verl.trainer.main_ppo' and 'trainer.experiment_name=<exp>' (exact elements)."""
    want = f"trainer.experiment_name={exp}"
    out = []
    for d in os.listdir(proc):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(os.path.join(proc, d, "cmdline"), "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if "verl.trainer.main_ppo" in argv and want in argv:
            out.append(int(d))
    return sorted(out)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def stop_run(exp: str, wait_s: int) -> dict:
    pids = find_driver_pids(exp)
    rec = {"pids": pids, "sigterm_utc": None, "exited": None, "sigkill_utc": None}
    if not pids:
        rec["note"] = "no live driver process matched; the run had already exited"
        return rec
    rec["sigterm_utc"] = utc()
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    t_end = time.time() + wait_s
    while time.time() < t_end and any(_alive(p) for p in pids):
        time.sleep(5)
    if any(_alive(p) for p in pids):
        rec["sigkill_utc"] = utc()
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(10)
    rec["exited"] = not any(_alive(p) for p in pids)
    rec["exited_utc"] = utc()
    return rec


def ray_gpu_lines() -> list:
    try:
        p = subprocess.run(["ray", "status"], capture_output=True, text=True, timeout=90)
        return [ln.strip() for ln in p.stdout.splitlines() if "GPU" in ln][:6]
    except Exception as e:  # noqa: BLE001
        return [f"ray status failed: {e!r}"]


# --------------------------------------------------------------------------------------------------- the monitor


class Monitor:
    def __init__(self, a):
        self.a = a
        self.group_size = a.group_size or ARMS[a.arm][0]
        self.acc_rule = ARMS[a.arm][1]
        self.state = {"train": {}, "ref": {}, "metrics": {}, "val": {}, "guard_failures": [], "log_guard_failed": False}
        self.seen = {"dump": {}, "metric": {}, "val": {}}  # first time complete evidence became visible
        self.reasons = {"train": {}, "ref": {}, "val": {}}
        self.ref_attempts = {}
        self.alerts, self.alert_keys = [], set()
        self.tail = LogTail(a.log)
        self.ff_ok = set()
        self.launcher_rc = None
        self.lr_expect = None
        if a.expect_lr:
            try:
                self.lr_expect = float(a.expect_lr)
            except ValueError:
                with open(a.expect_lr) as f:
                    self.lr_expect = {int(k): float(v) for k, v in json.load(f)["logged_actor_lr"].items()}
        self.started = utc()
        self.last_status, self.last_status_t = None, 0.0
        if not a.replay:
            os.makedirs(a.out_dir, exist_ok=True)

    # events / alerts
    def event(self, kind: str, **kw):
        rec = {"utc": utc(), "kind": kind, **kw}
        line = json.dumps(rec, sort_keys=True, default=str)
        print(f"[stop-monitor] {line}", flush=True)
        if not self.a.replay:
            with open(os.path.join(self.a.out_dir, "monitor.jsonl"), "a") as f:
                f.write(line + "\n")

    def alert(self, key: str, detail: str):
        if key in self.alert_keys:
            return
        self.alert_keys.add(key)
        self.alerts.append({"utc": utc(), "key": key, "detail": detail})
        self.event("alert", key=key, detail=detail)

    # scanners
    def _steps_in(self, sub: str) -> list:
        p = os.path.join(self.a.dump_dir, sub)
        try:
            names = os.listdir(p)
        except OSError:
            return []
        return sorted(int(n[:-6]) for n in names if n.endswith(".jsonl") and n[:-6].isdigit())

    def scan_train(self):
        for k in self._steps_in("train"):
            if k < 1 or k > self.a.max_step or self.state["train"].get(k):
                continue
            try:
                self.state["train"][k] = load_train_dump(
                    os.path.join(self.a.dump_dir, "train", f"{k}.jsonl"), k, self.group_size, self.a.prompts
                )
                self.seen["dump"][k] = time.time()
                self.reasons["train"].pop(k, None)
            except (NotReady, Invalid) as e:
                self.state["train"][k] = None
                self.reasons["train"][k] = f"{type(e).__name__}: {e}"
                if isinstance(e, Invalid):
                    self.alert(f"train_invalid_{k}", f"training dump {k} violates the batch contract: {e}")
                continue
            self.load_ref(k)
            t, r = self.state["train"][k], self.state["ref"].get(k)
            if t and r and t["identity"] != r["identity"]:
                self.alert(
                    f"identity_{k}", f"step {k}: prompt instances differ from the reference (qid -> input / gts)"
                )
            self.on_step_dump(k)
        for k, t in self.state["train"].items():  # retry a reference that was unreadable when its arm step completed
            if t and not self.state["ref"].get(k) and self.ref_attempts.get(k, 0) < 5:
                self.load_ref(k)

    def load_ref(self, k: int):
        self.ref_attempts[k] = self.ref_attempts.get(k, 0) + 1
        try:
            self.state["ref"][k] = load_train_dump(
                os.path.join(self.a.ref_dump_dir, "train", f"{k}.jsonl"), k, self.a.ref_group_size, self.a.prompts
            )
            self.reasons["ref"].pop(k, None)
        except (NotReady, Invalid) as e:
            self.state["ref"][k] = None
            self.reasons["ref"][k] = f"{type(e).__name__}: {e}"
            self.alert(f"ref_{k}", f"reference dump {k} unusable: {e}")

    def scan_val(self):
        for s in self._steps_in("val"):
            if s > self.a.max_step or self.state["val"].get(s):
                continue
            try:
                self.state["val"][s] = load_val_dump(
                    os.path.join(self.a.dump_dir, "val", f"{s}.jsonl"), self.a.eval_rows
                )
                self.seen["val"][s] = time.time()
                self.event("eval", step=s, acc_pct=round(self.state["val"][s]["acc_pct"], 3))
            except (NotReady, Invalid) as e:
                self.state["val"][s] = None
                self.reasons["val"][s] = f"{type(e).__name__}: {e}"
                if isinstance(e, Invalid):
                    self.alert(f"val_invalid_{s}", f"eval dump {s} violates the eval contract: {e}")

    def scan_log(self):
        for line in self.tail.read_new():
            if GUARD_FAILED in line:
                self.state["log_guard_failed"] = True
            m = re.search(r"LAUNCHER_RC=(\d+)", line)
            if m:
                self.launcher_rc = int(m.group(1))
            m = STEP_RE.search(line)
            if not m:
                continue
            k = int(m.group(1))
            kv = parse_metric_line(m.group(2))
            if not kv:
                continue
            if k >= 1 and LAST_KEY not in m.group(2):
                self.alert(
                    f"metric_incomplete_{k}",
                    f"step {k}: metric line without its final key {LAST_KEY} "
                    "(split or truncated line; missing keys count as unevaluable)",
                )
            self.state["metrics"].setdefault(k, {}).update(kv)
            self.seen["metric"].setdefault(k, time.time())
            if k >= 1:
                self.check_lr(k, kv)

    def check_lr(self, k: int, kv: dict):
        if self.lr_expect is None or "actor/lr" not in kv:
            return
        want = self.lr_expect if isinstance(self.lr_expect, float) else self.lr_expect.get(k)
        if want is not None and abs(kv["actor/lr"] - want) > 1e-6 * abs(want) + 1e-15:
            self.alert(f"lr_{k}", f"step {k}: logged actor/lr {kv['actor/lr']:.10g} != expected {want:.10g}")

    def scan_guard(self):
        for p in glob.glob(os.path.join(self.a.guard_dir, "first_forward_actor_rank*.json")):
            if p in self.ff_ok:
                continue
            try:
                with open(p) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue  # not visible / complete yet
            if r.get("status") == "PASS" and not r.get("restored"):
                self.ff_ok.add(p)
            else:
                msg = f"rank {r.get('rank')}: status={r.get('status')} restored={r.get('restored')}"
                if msg not in self.state["guard_failures"]:
                    self.state["guard_failures"].append(msg)

    def on_step_dump(self, k: int):
        t, r = self.state["train"][k], self.state["ref"].get(k)
        self.event(
            "train_step",
            step=k,
            acc_pct=round(t["acc_pct"], 3),
            missing_answer_pct=round(t["missing_answer_pct"], 3),
            ref_acc_pct=None if not r else round(r["acc_pct"], 3),
        )

    # observability
    def check_freshness(self, now: float):
        dumps, mets = self.seen["dump"], self.seen["metric"]
        for k, t0 in dumps.items():
            if k not in mets and now - t0 > self.a.metric_lag_alert_s:
                self.alert(f"metric_lag_{k}", f"step {k}: dump complete for {now - t0:.0f}s but no metric line yet")
        # a step without a complete dump although a later dump, or its own metric line, has been visible for a while
        hi = max([k for k in dumps] + [k for k in mets if 1 <= k <= self.a.max_step] + [0])
        for k in range(1, hi + 1):
            if k in dumps:
                continue
            later = [dumps[j] for j in dumps if j > k] + ([mets[k]] if k in mets else [])
            if later and now - min(later) > self.a.gap_alert_s:
                self.alert(
                    f"dump_gap_{k}",
                    f"step {k}: no complete training dump ({self.reasons['train'].get(k, 'file not visible')})",
                )
        if dumps and max(dumps) >= 3 and len(self.ff_ok) < self.a.world and not self.state["guard_failures"]:
            self.alert(
                "guard_records",
                f"only {len(self.ff_ok)}/{self.a.world} first-forward PASS records visible at step >= 3",
            )

    def latency(self) -> dict:
        lags = {
            k: round(self.seen["metric"][k] - self.seen["dump"][k], 1)
            for k in self.seen["dump"]
            if k in self.seen["metric"]
        }
        out = {"metric_minus_dump_s": lags}
        if lags:
            v = sorted(lags.values())
            out.update({"median_s": statistics.median(v), "max_s": v[-1], "n": len(v)})
        return out

    def last_completed(self) -> dict:
        tr = [k for k, v in self.state["train"].items() if v]
        return {
            "last_metric_step": max([k for k in self.state["metrics"] if k >= 1], default=None),
            "last_complete_train_dump": max(tr, default=None),
            "last_complete_eval": max([s for s, v in self.state["val"].items() if v], default=None),
        }

    def status(self, ev: dict, extra: dict | None = None) -> dict:
        per = {}
        for k in range(1, ev["last_step"] + 1):
            t, r, m = self.state["train"].get(k), self.state["ref"].get(k), self.state["metrics"].get(k, {})
            per[k] = {
                "acc": None if not t else round(t["acc_pct"], 3),
                "ref_acc": None if not r else round(r["acc_pct"], 3),
                "d": None if ev["series"]["d"].get(k) is None else round(ev["series"]["d"][k], 3),
                "missing_answer_pct": None if not t else round(t["missing_answer_pct"], 3),
                "k3_kl": m.get("rollout_corr/k3_kl"),
                "entropy": m.get("actor/entropy"),
                "lr": m.get("actor/lr"),
            }
        st = {
            "exp": self.a.exp,
            "arm": self.a.arm,
            "acc_rule": self.acc_rule,
            "group_size": self.group_size,
            "acting": bool(self.a.act),
            "started_utc": self.started,
            "updated_utc": utc(),
            "evidence": {
                "dump_dir": self.a.dump_dir,
                "log": self.a.log,
                "guard_dir": self.a.guard_dir,
                "ref_dump_dir": self.a.ref_dump_dir,
            },
            **self.last_completed(),
            "streaks": ev["streaks"],
            "a5_baseline_pp": ev["a5_baseline"],
            "heldout": ev["heldout"],
            "triggers": ev["triggers"],
            "alerts": self.alerts,
            "first_forward_pass_records": len(self.ff_ok),
            "latency": self.latency(),
            "launcher_rc": self.launcher_rc,
            "not_complete": {k: v for k, v in self.reasons.items() if v},
            "per_step": per,
        }
        if extra:
            st.update(extra)
        return st

    def write_status(self, st: dict, force: bool = False):
        if self.a.replay:
            return
        key = json.dumps(
            {k: v for k, v in st.items() if k not in ("updated_utc", "latency")}, sort_keys=True, default=str
        )
        if force or key != self.last_status or time.time() - self.last_status_t > 300:
            with open(os.path.join(self.a.out_dir, "status.json"), "w") as f:
                json.dump(st, f, indent=1, sort_keys=True, default=str)
            self.last_status, self.last_status_t = key, time.time()

    def scan_all(self):
        self.scan_log()
        self.scan_train()
        self.scan_val()
        self.scan_guard()

    # main loops
    def replay(self) -> int:
        self.scan_all()
        ev = evaluate(self.state, self.acc_rule, self.a.max_step)
        st = self.status(ev)
        first = ev["triggers"][0] if ev["triggers"] else None
        print(
            json.dumps(
                {
                    "exp": self.a.exp,
                    "arm": self.a.arm,
                    "last_step": ev["last_step"],
                    "first_trigger": first,
                    "triggers": [(t["rule"], t["step"]) for t in ev["triggers"]],
                    "unevaluable": {k: v[:20] for k, v in ev["unevaluable"].items()},
                    "a5_baseline_pp": ev["a5_baseline"],
                    "alerts": self.alerts,
                },
                indent=1,
                default=str,
            )
        )
        if self.a.out_json:
            with open(self.a.out_json, "w") as f:
                json.dump(st, f, indent=1, sort_keys=True, default=str)
        return 3 if first else 0

    def live(self) -> int:
        self.event(
            "start",
            exp=self.a.exp,
            arm=self.a.arm,
            acc_rule=self.acc_rule,
            group_size=self.group_size,
            act=bool(self.a.act),
            dump_dir=self.a.dump_dir,
            log=self.a.log,
            ref=self.a.ref_dump_dir,
        )
        t_end = time.time() + 3600 * self.a.max_runtime_h
        while time.time() < t_end:
            self.scan_all()
            now = time.time()
            self.check_freshness(now)
            ev = evaluate(self.state, self.acc_rule, self.a.max_step)
            if self.launcher_rc is not None:  # the run has ended: final pass, never act
                time.sleep(self.a.final_wait_s)  # let the last dumps become visible through the FUSE listing
                self.scan_all()
                ev = evaluate(self.state, self.acc_rule, self.a.max_step)
                status = "finished" if not ev["triggers"] else "finished; a rule is satisfied in the final evidence"
                self.write_status(self.status(ev, {"final": True, "arm_status": status}), force=True)
                self.event(
                    "end",
                    launcher_rc=self.launcher_rc,
                    triggers=[(t["rule"], t["step"]) for t in ev["triggers"]],
                    **self.last_completed(),
                )
                return 3 if ev["triggers"] else 0
            if ev["triggers"]:
                return self.trigger(ev)
            self.write_status(self.status(ev))
            time.sleep(self.a.poll_s)
        self.event("timeout", hours=self.a.max_runtime_h)
        return 2

    def trigger(self, ev: dict) -> int:
        t = ev["triggers"][0]
        before = self.last_completed()
        rec = {
            "exp": self.a.exp,
            "arm": self.a.arm,
            "rule": t,
            "all_triggers": ev["triggers"],
            "detected_utc": utc(),
            "acting": bool(self.a.act),
            "last_completed_before_stop": before,
            "arm_status": "early-stopped/incomplete" if self.a.act else "rule triggered (monitor not acting)",
        }
        path = os.path.join(self.a.out_dir, "stop_trigger.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True, default=str)
        self.event("trigger", rule=t["rule"], step=t["step"], acting=bool(self.a.act))
        if self.a.act:
            rec["stop"] = stop_run(self.a.exp, self.a.kill_wait_s)
            rec["ray_status_gpu"] = ray_gpu_lines()
            time.sleep(self.a.final_wait_s)
            self.scan_all()
            rec["last_completed_after_stop"] = self.last_completed()
            with open(path, "w") as f:
                json.dump(rec, f, indent=1, sort_keys=True, default=str)
            self.event("stopped", **rec["stop"])
        self.write_status(
            self.status(
                evaluate(self.state, self.acc_rule, self.a.max_step), {"final": True, "arm_status": rec["arm_status"]}
            ),
            force=True,
        )
        return 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--guard-dir", default="")
    ap.add_argument("--ref-dump-dir", default=f"{LOGS}/gemma4_eval/{A3A}")
    ap.add_argument("--ref-group-size", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--group-size", type=int, default=0, help="override the arm's rollout.n")
    ap.add_argument("--prompts", type=int, default=PROMPTS_PER_STEP)
    ap.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    ap.add_argument("--world", type=int, default=64, help="trainer ranks expected to write first-forward records")
    ap.add_argument("--max-step", type=int, default=100)
    ap.add_argument("--expect-lr", default="", help="constant LR, or a gemma4_lr_schedule_export.py JSON (alerts only)")
    ap.add_argument("--poll-s", type=float, default=30)
    ap.add_argument("--act", action="store_true", help="terminate the run's driver when a rule triggers")
    ap.add_argument("--replay", action="store_true", help="evaluate existing evidence once; never acts")
    ap.add_argument("--out-json", default="", help="replay: write the full status here")
    ap.add_argument("--kill-wait-s", type=int, default=180)
    ap.add_argument("--final-wait-s", type=int, default=90)
    ap.add_argument("--metric-lag-alert-s", type=float, default=420)
    ap.add_argument("--gap-alert-s", type=float, default=300)
    ap.add_argument("--max-runtime-h", type=float, default=14)
    a = ap.parse_args(argv)
    a.dump_dir = a.dump_dir or f"{LOGS}/gemma4_eval/{a.exp}"
    a.log = a.log or f"{LOGS}/{a.exp}.log"
    a.guard_dir = a.guard_dir or f"{LOGS}/gemma4_guard/{a.exp}"
    a.out_dir = a.out_dir or f"{LOGS}/gemma4_stop_monitor/{a.exp}"
    if a.replay and a.act:
        ap.error("--replay never acts")
    m = Monitor(a)
    return m.replay() if a.replay else m.live()


if __name__ == "__main__":
    sys.exit(main())
