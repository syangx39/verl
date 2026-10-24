#!/usr/bin/env python3
"""Self-test for gemma4_config_diff.py (PR #12 launch-time configuration check). Standard library only.

Synthetic driver logs reproduce what Ray writes: an ANSI-coloured "(TaskRunner pid=..., ip=...)" prefix on every line,
a pprinted config dict, a warning (with indented code lines) interleaved inside the dict and another prefixed line
glued onto the end of a dict line. Cases:
  - identical configs except identity keys -> PASS;
  - a declared change (--expect) -> PASS; the same change undeclared -> FAIL; a wrong declared value -> FAIL;
  - --assert restating the reference value -> PASS; --assert on a misspelt key -> FAIL;
  - an extra undeclared key in the run -> FAIL;
  - if the committed PR #11 logs exist: A3b vs A3a declaring only LR 4e-6 -> PASS (LR + 6 identity keys), undeclared
    -> FAIL.

Usage:  python3 test_gemma4_config_diff.py
"""

from __future__ import annotations

import os
import pprint
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
P = "\x1b[36m(TaskRunner pid=4242, ip=10.0.0.9)\x1b[0m "


def cfg(exp, lr=2e-6, n=8, extra=None):
    c = {
        "actor_rollout_ref": {
            "actor": {"optim": {"lr": lr, "lr_scheduler_type": "constant", "min_lr_ratio": 0.0}, "rollout_n": n},
            "rollout": {"n": n, "temperature": 0.8, "trace": {"experiment_name": exp}},
        },
        "trainer": {
            "experiment_name": exp,
            "rollout_data_dir": f"/d/{exp}/train",
            "default_local_dir": f"ck/{exp}",
            "total_training_steps": 100,
            "logger": ["console", "tensorboard"],
        },
        "data": {"train_files": ["/data/train.parquet"], "chat_template": "{% if x %}{{ y }}{% endif %}"},
    }
    if extra:
        c["data"].update(extra)
    return c


def write_log(path, c):
    lines = pprint.pformat(c, width=60).split("\n")
    out = ["+ python3 -m verl.trainer.main_ppo ...", P + "TaskRunner hostname: worker-0, PID: 4242"]
    for i, ln in enumerate(lines):
        if i == 3:  # a warning with indented code lines from the same TaskRunner, inside the dict
            out += [
                P + "/workspace/verl-pin/verl/trainer/main_ppo.py:182: UserWarning: Disabled critic",
                P + "  use_critic=need_critic(config),",
            ]
        if i == 5:  # another prefixed line glued onto the end of a dict line (buffered driver stdout)
            out.append(P + ln + P + "(WARNING) something unrelated")
            continue
        out.append(P + ln)
    out.append(P + "step:0 - val-core/acc:0.36")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def run(ref, log, *extra):
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "gemma4_config_diff.py"), "--ref-log", ref, "--log", log, *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    tmp = tempfile.mkdtemp()
    f = lambda n: os.path.join(tmp, n)  # noqa: E731
    write_log(f("ref.log"), cfg("ref_exp"))
    write_log(f("same.log"), cfg("other_exp"))
    write_log(f("n16.log"), cfg("a6_exp", n=16))
    write_log(f("extra.log"), cfg("x_exp", extra={"shuffle": False}))
    lr = "actor_rollout_ref.actor.optim.lr"
    cases = [
        ("identity keys only", ("ref.log", "same.log"), (), True, "PASS"),
        (
            "declared n=16 (+ derived actor.rollout_n)",
            ("ref.log", "n16.log"),
            ("--expect", "actor_rollout_ref.rollout.n=16", "--expect", "actor_rollout_ref.actor.rollout_n=16"),
            True,
            "PASS",
        ),
        ("undeclared n=16", ("ref.log", "n16.log"), (), False, "other"),
        (
            "derived key undeclared",
            ("ref.log", "n16.log"),
            ("--expect", "actor_rollout_ref.rollout.n=16"),
            False,
            "actor_rollout_ref.actor.rollout_n",
        ),
        (
            "wrong declared value",
            ("ref.log", "n16.log"),
            ("--expect", "actor_rollout_ref.rollout.n=12", "--expect", "actor_rollout_ref.actor.rollout_n=16"),
            False,
            "expected-MISMATCH",
        ),
        (
            "assert restating the reference",
            ("ref.log", "same.log"),
            ("--assert", f"{lr}=2e-6", "--assert", "actor_rollout_ref.actor.optim.min_lr_ratio=0"),
            True,
            "asserted",
        ),
        (
            "assert on a misspelt key",
            ("ref.log", "same.log"),
            ("--assert", "actor_rollout_ref.actor.optim.lrr=2e-6"),
            False,
            "asserted-MISMATCH",
        ),
        ("extra undeclared key", ("ref.log", "extra.log"), (), False, "data.shuffle"),
    ]
    fails = 0
    for name, (a, b), extra, want_ok, want in cases:
        rc, out = run(f(a), f(b), *extra)
        ok = (rc == 0) == want_ok and want in out
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}: rc={rc}" + ("" if ok else f" :: {out[-600:]}"))
    logs = os.path.join(HERE, "logs")
    a3a = os.path.join(logs, "gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753.log")
    a3b = os.path.join(logs, "gemma4_e2b_rl05fp32_a3lr4e6_16n64g_s100_seed1_1023_1116.log")
    if os.path.exists(a3a) and os.path.exists(a3b):
        rc, out = run(a3a, a3b, "--expect", f"{lr}=4e-6")
        ok = rc == 0 and "differences=7 failures=0" in out  # LR + 6 identity keys
        fails += not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} real logs: A3b vs A3a, LR 4e-6 declared: rc={rc}" + ("" if ok else out[-600:])
        )
        rc, out = run(a3a, a3b)
        ok = rc != 0 and f"other              {lr}" in out
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} real logs: A3b vs A3a, undeclared: rc={rc}" + ("" if ok else out[-600:]))
    else:
        print("SKIP real logs (not present)")
    print(f"{'PASS' if not fails else 'FAIL'}: {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
