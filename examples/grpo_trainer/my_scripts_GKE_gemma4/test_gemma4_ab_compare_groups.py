#!/usr/bin/env python3
"""Self-test for gemma4_ab_compare.py's per-run batch shapes (PR #12 A6: n=16 vs the n=8 control).

Builds tiny synthetic runs (2 prompt groups per step, 2 steps, 4 eval rows): A with 8 completions per prompt, B with 16
on the same prompt instances, and runs the comparison script in a subprocess:
  positive : per-run shapes (A 16 = 2 x 8, B 32 = 2 x 16) -> exit 0, JSON records both shapes, B acc on all 16 rows;
  negative : the same runs with one shared shape (16 rows = 2 x 8) -> fails on B's row count;
  negative : B has groups of 15 and 17 rows (same total) -> fails on the group size;
  negative : B misses one row (31) -> fails on the row count;
  negative : B's step 2 trains a different question -> fails on the qid sets;
  negative : B declared with a different number of groups per step (48 = 3 x 16) -> rejected before reading;
  positive : the original single shared shape on two n=8 runs still passes.

Usage (needs numpy):  python3 test_gemma4_ab_compare_groups.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def prompt(q):
    return f"user\n<start_of_turn>user\nSolve: what is {q} + 1?<end_of_turn>\n<start_of_turn>model\nmodel\n"


def write_run(root, n, ragged_step=None, drop_one_step=None, other_qid_step=None):
    os.makedirs(os.path.join(root, "train"))
    os.makedirs(os.path.join(root, "val"))
    for step in (1, 2):
        rows = []
        for qid in (10 * step, 10 * step + 1):
            q = qid + 500 if other_qid_step == step and qid == 10 * step + 1 else qid
            for k in range(n):
                rows.append(
                    {
                        "qid": q,
                        "input": prompt(q),
                        "gts": json.dumps([str(q + 1)] * 2),
                        "acc": float(k % 4 == 0),
                        "fmt": 1.0,
                        "score": 0.1 + float(k % 4 == 0),
                        "output": f"o{k}",
                    }
                )
        if ragged_step == step:  # move one row from the first group to the second: groups of n-1 and n+1
            first = [r for r in rows if r["qid"] == 10 * step]
            rows.remove(first[0])
            rows.append(dict(rows[-1]))
        if drop_one_step == step:
            rows = rows[:-1]
        with open(os.path.join(root, "train", f"{step}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(root, "val", "0.jsonl"), "w") as f:
        for qid in range(4):
            f.write(
                json.dumps(
                    {
                        "qid": qid,
                        "input": prompt(1000 + qid),
                        "gts": json.dumps(["1", "1"]),
                        "acc": float(qid % 2),
                        "fmt": 1.0,
                        "output": "x",
                    }
                )
                + "\n"
            )


def run(a, b, extra, out_json=None):
    cmd = [sys.executable, os.path.join(HERE, "gemma4_ab_compare.py"), "--a", f"a={a}", "--b", f"b={b}", *extra]
    if out_json:
        cmd += ["--out-json", out_json]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr)[-500:]


def main() -> int:
    tmp = tempfile.mkdtemp()
    d = lambda name: os.path.join(tmp, name)  # noqa: E731
    write_run(d("a8"), 8)
    write_run(d("a8_copy"), 8)
    write_run(d("b16"), 16)
    write_run(d("b16_ragged"), 16, ragged_step=2)
    write_run(d("b16_short"), 16, drop_one_step=1)
    write_run(d("b16_otherq"), 16, other_qid_step=2)
    per_run = ["--a-rows-per-step", "16", "--a-group-size", "8", "--b-rows-per-step", "32", "--b-group-size", "16"]
    shared = ["--rows-per-step", "16", "--group-size", "8"]
    cases = [
        ("positive: per-run shapes 2x8 vs 2x16", d("a8"), d("b16"), per_run, True, ""),
        ("negative: shared shape on an n=16 run", d("a8"), d("b16"), shared, False, "32 rows, expected 16"),
        ("negative: ragged 15/17-row groups", d("a8"), d("b16_ragged"), per_run, False, "rows, expected 16"),
        ("negative: one row missing", d("a8"), d("b16_short"), per_run, False, "31 rows, expected 32"),
        ("negative: different question at step 2", d("a8"), d("b16_otherq"), per_run, False, "prompt qid sets differ"),
        (
            "negative: different groups per step",
            d("a8"),
            d("b16"),
            ["--a-rows-per-step", "16", "--a-group-size", "8", "--b-rows-per-step", "48", "--b-group-size", "16"],
            False,
            "prompt groups per step",
        ),
        ("positive: shared shape, two n=8 runs", d("a8"), d("a8_copy"), shared, True, ""),
    ]
    fails = 0
    for name, a, b, extra, want_ok, want_msg in cases:
        out = os.path.join(tmp, name.split(":")[0] + f"_{abs(hash(name))}.json")
        rc, tail = run(a, b, extra, out_json=out)
        ok = (rc == 0) if want_ok else (rc != 0 and want_msg in tail)
        if ok and name.startswith("positive: per-run"):
            res = json.load(open(out))
            bc = res["train"]["batch_checks"]
            per = res["train"]["per_step"]["1"]
            ok = (
                bc["a"] == {"rows_per_step": 16, "group_size": 8}
                and bc["b"] == {"rows_per_step": 32, "group_size": 16}
                and bc["groups_per_step"] == 2
                and bc["rows_per_step"] is None
                and abs(per["acc_a"] - 25.0) < 1e-9
                and abs(per["acc_b"] - 25.0) < 1e-9
            )
            tail = json.dumps(bc) + " " + json.dumps(per)
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}: rc={rc}" + ("" if ok else f" :: {tail}"))
    print(f"{'PASS' if not fails else 'FAIL'}: {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
