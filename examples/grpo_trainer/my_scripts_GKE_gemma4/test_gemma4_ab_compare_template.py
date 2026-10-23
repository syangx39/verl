#!/usr/bin/env python3
"""Self-test for gemma4_ab_compare.py's training-prompt identity checks (PR #11 A2).

Builds tiny synthetic run directories (2 prompt groups x 8 completions per step, 2 steps; 4 eval rows; one training
question ends with a newline, which the Gemma-4 chat template trims once the inner markers are gone) and runs the
comparison script in a subprocess:
  positive : B = A with the inner Gemma-3 template stripped, --train-template-change strip_inner_gemma3 -> exit 0;
  negative : the same B in strict mode (no template change declared) -> fails;
  negative : template mode, but one B question differs -> fails;
  negative : template mode, but one B ground truth differs -> fails;
  negative : template mode, but A is not in the double-wrapped form -> fails;
  negative : template mode, but the eval prompts differ -> fails (eval stays strict);
  positive : strict mode on identical prompts -> exit 0.

Usage (needs numpy):  python3 test_gemma4_ab_compare_template.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = "You are given a problem. Place it between <reasoning> and </reasoning> ..."


def double(q):
    return f"user\n<start_of_turn>user\n{SYS}\n\n{q}<end_of_turn>\n<start_of_turn>model\nmodel\n"


def single(q):  # Gemma-4 chat template: message content | trim
    return f"user\n{(SYS + chr(10) * 2 + q).strip()}\nmodel\n"


def write_run(root, render_train, render_val, mutate_q=None, mutate_gt=None):
    os.makedirs(os.path.join(root, "train"))
    os.makedirs(os.path.join(root, "val"))
    for step in (1, 2):
        with open(os.path.join(root, "train", f"{step}.jsonl"), "w") as f:
            for qid in (10 * step, 10 * step + 1):
                q = f"What is {qid} + 1?" + ("\n" if qid == 21 else "")  # one question ends with a newline
                if mutate_q == (step, qid):
                    q = q + " (changed)"
                gt = json.dumps([str(qid + 1)] * 2)
                if mutate_gt == (step, qid):
                    gt = json.dumps(["999", "999"])
                for k in range(8):
                    f.write(json.dumps({"qid": qid, "input": render_train(q), "gts": gt, "acc": float(k % 2),
                                        "fmt": 1.0, "output": f"o{k}"}) + "\n")
    with open(os.path.join(root, "val", "0.jsonl"), "w") as f:
        for qid in range(4):
            f.write(json.dumps({"qid": qid, "input": render_val(f"Eval {qid}?"), "gts": json.dumps(["1", "1"]),
                                "acc": float(qid % 2), "fmt": 1.0, "output": "x"}) + "\n")


def run(a, b, mode):
    cmd = [sys.executable, os.path.join(HERE, "gemma4_ab_compare.py"), "--a", f"a={a}", "--b", f"b={b}",
           "--rows-per-step", "16", "--group-size", "8"]
    if mode:
        cmd += ["--train-template-change", mode]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr)[-400:]


def main() -> int:
    tmp = tempfile.mkdtemp()
    d = lambda name: os.path.join(tmp, name)  # noqa: E731
    write_run(d("a"), double, double)
    write_run(d("b_single"), single, double)
    write_run(d("b_single_q"), single, double, mutate_q=(2, 21))
    write_run(d("b_single_gt"), single, double, mutate_gt=(1, 11))
    write_run(d("a_single"), single, double)
    write_run(d("b_single_evalchanged"), single, single)
    write_run(d("a_copy"), double, double)
    cases = [
        ("positive: declared template change", d("a"), d("b_single"), "strip_inner_gemma3", True, ""),
        ("negative: same B in strict mode", d("a"), d("b_single"), None, False, "different input/ground truth"),
        ("negative: changed question", d("a"), d("b_single_q"), "strip_inner_gemma3", False, "different input"),
        ("negative: changed ground truth", d("a"), d("b_single_gt"), "strip_inner_gemma3", False, "different input"),
        ("negative: A not double-wrapped", d("a_single"), d("b_single"), "strip_inner_gemma3", False, "not in the form"),
        ("negative: eval prompts changed", d("a"), d("b_single_evalchanged"), "strip_inner_gemma3", False, "input/ground truth"),
        ("positive: strict mode, identical prompts", d("a"), d("a_copy"), None, True, ""),
    ]
    fails = 0
    for name, a, b, mode, want_ok, want_msg in cases:
        rc, tail = run(a, b, mode)
        ok = (rc == 0) if want_ok else (rc != 0 and want_msg in tail)
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}: rc={rc}" + ("" if ok else f" :: {tail}"))
    print(f"{'PASS' if not fails else 'FAIL'}: {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
