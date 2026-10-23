#!/usr/bin/env python3
"""Self-test for the training-only format coefficient in maxtext_math_reward.py (PR #11 A1).

Each scenario imports the scorer in a fresh subprocess, because the knobs are read at import time:
  1. default knobs: score = acc + 0.1 * fmt for every row, no extra keys (unchanged behaviour);
  2. REWARD_FMT_WEIGHT_TRAIN=0 + eval manifest: training rows score = acc (fmt_coef 0, eval_scope 0), eval rows keep
     acc + 0.1 * fmt (fmt_coef 0.1, eval_scope 1); acc and fmt flags identical to scenario 1;
  3. the knob without REWARD_EVAL_QUESTIONS, a row without extra_info["question"], and a manifest whose count
     disagrees with its hashes all fail loudly.
Cases cover correct / incorrect x formatted / unformatted completions.

Usage (needs math_verify, i.e. the training image):  python3 test_maxtext_reward_fmt_scope.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GT = json.dumps(["72", "72"])
CASES = [  # (name, completion, acc, fmt)
    ("correct_formatted", "<reasoning>2*36</reasoning><answer>72</answer>", 1.0, 1.0),
    ("correct_unformatted", "blah <answer> 72 </answer>", 1.0, 0.0),
    ("wrong_formatted", "<reasoning>hmm</reasoning><answer>71</answer>", 0.0, 1.0),
    ("wrong_unformatted", "no tags at all 72", 0.0, 0.0),
]
TRAIN_Q = "A training question that is not in the eval set."
EVAL_Q = "An eval question that is in the manifest."

CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("mmr", sys.argv[1])
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as e:
    print(json.dumps({"import_error": type(e).__name__, "msg": str(e)})); sys.exit(0)
out = {}
for name, completion, question in json.loads(sys.argv[2]):
    extra = {"index": 7} if question is None else {"index": 7, "question": question}
    try:
        out[name] = m.compute_score("openmathinstruct2_maxtext", completion, sys.argv[3], extra_info=extra)
    except Exception as e:
        out[name] = {"error": type(e).__name__, "msg": str(e)}
print(json.dumps(out))
"""


def _run(env_extra, rows):
    env = {k: v for k, v in os.environ.items() if not k.startswith("REWARD_")}
    env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", CHILD, os.path.join(HERE, "maxtext_math_reward.py"), json.dumps(rows), GT],
                       env=env, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise SystemExit(f"child failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    fails = []

    def check(cond, msg):
        print(("OK   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    tmp = tempfile.mkdtemp()
    manifest = os.path.join(tmp, "eval_q.json")
    with open(manifest, "w") as f:
        json.dump({"n": 1, "question_sha256": [hashlib.sha256(EVAL_Q.encode()).hexdigest()]}, f)
    bad_manifest = os.path.join(tmp, "eval_q_bad.json")
    with open(bad_manifest, "w") as f:
        json.dump({"n": 2, "question_sha256": [hashlib.sha256(EVAL_Q.encode()).hexdigest()]}, f)

    rows = [(f"{c}_{split}", comp, q) for c, comp, _, _ in CASES for split, q in (("train", TRAIN_Q), ("eval", EVAL_Q))]
    base = _run({}, rows)
    scoped = _run({"REWARD_FMT_WEIGHT_TRAIN": "0", "REWARD_EVAL_QUESTIONS": manifest}, rows)
    for c, _, acc, fmt in CASES:
        for split in ("train", "eval"):
            k = f"{c}_{split}"
            b, s = base[k], scoped[k]
            check(b["acc"] == acc and b["fmt"] == fmt and abs(b["score"] - (acc + 0.1 * fmt)) < 1e-9
                  and "fmt_coef" not in b and "eval_scope" not in b, f"default {k}: score {b['score']} = acc + 0.1*fmt, no extra keys")
            check(s["acc"] == b["acc"] and s["fmt"] == b["fmt"], f"scoped {k}: acc / fmt flags unchanged")
            if split == "train":
                check(abs(s["score"] - acc) < 1e-9 and s["fmt_coef"] == 0.0 and s["eval_scope"] == 0.0,
                      f"scoped {k}: training row score {s['score']} = acc, fmt_coef 0")
            else:
                check(abs(s["score"] - (acc + 0.1 * fmt)) < 1e-9 and s["fmt_coef"] == 0.1 and s["eval_scope"] == 1.0,
                      f"scoped {k}: eval row score {s['score']} = acc + 0.1*fmt, fmt_coef 0.1")
            check(s["length_penalty"] == 0.0 and b["length_penalty"] == 0.0, f"{k}: length penalty off")

    r = _run({"REWARD_FMT_WEIGHT_TRAIN": "0"}, rows[:1])
    check(r.get("import_error") == "ValueError", "knob without REWARD_EVAL_QUESTIONS fails at import")
    r = _run({"REWARD_FMT_WEIGHT_TRAIN": "0", "REWARD_EVAL_QUESTIONS": bad_manifest}, rows[:1])
    check(r.get("import_error") == "ValueError", "manifest count mismatch fails at import")
    r = _run({"REWARD_FMT_WEIGHT_TRAIN": "0", "REWARD_EVAL_QUESTIONS": manifest}, [("noq", CASES[0][1], None)])
    check(r["noq"].get("error") == "RuntimeError", "row without extra_info['question'] fails")
    r = _run({}, [("noq", CASES[0][1], None)])
    check("error" not in r["noq"] and abs(r["noq"]["score"] - 1.1) < 1e-9, "default knobs do not need extra_info['question']")

    print(f"{'PASS' if not fails else 'FAIL'}: {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
