#!/usr/bin/env python3
"""Self-test for gemma4_signal_density.py's n=8-equivalent mixed share and --shape validation (PR #12 A6).

  - mixed_prob_at: n=8 is the observed indicator; for n=16, c=0/16 -> 0, c=1 -> 0.5 (the one correct answer is in a
    random half), c=8 -> 1 - 2/C(16,8);
  - a 4x16 run with --shape 4x16 passes and reports mixed_n8eq below the observed mixed share;
  - --shape 4x8 on that run fails; a 15/17-row ragged group fails under --shape 4x16;
  - an n=8 run reports mixed_n8eq == mixed.

Usage:  python3 test_gemma4_signal_density_shapes.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma4_signal_density import mixed_prob_at  # noqa: E402


def write(root, n, correct_per_group, ragged=False):
    os.makedirs(root)
    for step in range(1, 3):
        rows = []
        for j, c in enumerate(correct_per_group):
            for k in range(n):
                acc = float(k < c)
                rows.append({"qid": 100 * step + j, "input": f"q{j}", "gts": "[1]", "acc": acc, "score": acc + 0.1})
        if ragged:
            rows.remove(rows[0])
            rows.append(dict(rows[-1]))
        with open(os.path.join(root, f"{step}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


def run(args):
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "gemma4_signal_density.py"), *args, "--windows", "1-2"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {name}" + ("" if ok else f" :: {detail}"))

    check("n=8 indicator", mixed_prob_at(3, 8) == 1.0 and mixed_prob_at(0, 8) == 0.0 and mixed_prob_at(8, 8) == 0.0)
    check("n=16 c=0/16 -> 0", mixed_prob_at(0, 16) == 0.0 and mixed_prob_at(16, 16) == 0.0)
    check("n=16 c=1 -> 0.5", abs(mixed_prob_at(1, 16) - 0.5) < 1e-12, str(mixed_prob_at(1, 16)))
    check("n=16 c=8", abs(mixed_prob_at(8, 16) - (1 - 2 / math.comb(16, 8))) < 1e-12)

    tmp = tempfile.mkdtemp()
    write(os.path.join(tmp, "n16"), 16, [0, 1, 8, 16])
    write(os.path.join(tmp, "n16r"), 16, [0, 1, 8, 16], ragged=True)
    write(os.path.join(tmp, "n8"), 8, [0, 1, 4, 8])
    out = os.path.join(tmp, "o.json")
    rc, txt = run(["--run", f"a6={tmp}/n16", "--shape", "a6=4x16", "--out-json", out])
    w = json.load(open(out))["a6"]["1-2"] if rc == 0 else {}
    want_n8 = 100 * (0 + 0.5 + (1 - 2 / math.comb(16, 8)) + 0) / 4
    check(
        "4x16 run with --shape passes; mixed 50%, n8-equivalent lower",
        rc == 0 and abs(w["mixed"] - 50.0) < 1e-9 and abs(w["mixed_n8eq"] - want_n8) < 1e-9 and w["shape_validated"],
        f"rc={rc} {w} {txt[-300:]}",
    )
    rc, txt = run(["--run", f"a6={tmp}/n16", "--shape", "a6=4x8"])
    check("--shape 4x8 on an n=16 run fails", rc != 0 and "expected 8" in txt, txt[-300:])
    rc, txt = run(["--run", f"a6={tmp}/n16r", "--shape", "a6=4x16"])
    check("ragged 15/17 groups fail under --shape", rc != 0 and "expected 16" in txt, txt[-300:])
    rc, txt = run(["--run", f"c={tmp}/n8", "--shape", "c=4x8", "--out-json", out])
    w = json.load(open(out))["c"]["1-2"] if rc == 0 else {}
    check(
        "n=8 run: mixed_n8eq == mixed",
        rc == 0 and abs(w["mixed"] - w["mixed_n8eq"]) < 1e-9 and w["mixed"] == 50.0,
        f"{w}",
    )
    print(f"{'PASS' if not fails else 'FAIL'}: {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
