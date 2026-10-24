#!/usr/bin/env python3
"""Compare the runtime configuration of a run with a reference run (PR #12: "only the declared factor changes").

verl's TaskRunner prints its resolved configuration (OmegaConf.to_container(resolve=True), pprinted) at start-up, so
every driver log carries the configuration the run actually used, before any training step. This tool extracts that
dict from two driver logs, flattens it to dotted keys and classifies every difference:

  identity  keys that name the run (experiment name, dump / guard / checkpoint directories); always allowed;
  expected  keys declared with --expect KEY=VALUE: must differ from the reference AND equal VALUE in the run;
  asserted  keys declared with --assert KEY=VALUE: must equal VALUE in the run, whether or not the reference has it
            too (an explicit override that restates the reference value);
  allowed   keys matching --allow REGEX (e.g. a logging-only env var); reported, not a failure;
  other     anything else: a failure (an undeclared recipe change).

A declared key that is absent or has a different value also fails. Exit 0 only if every check holds. Values the
trainer injects after printing (e.g. actor.optim.total_training_steps, printed as -1) must be checked elsewhere.

Usage:
    python3 gemma4_config_diff.py --ref-log A3a.log --log A4.log \\
        --expect actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \\
        --expect actor_rollout_ref.actor.optim.num_cycles=0.25 \\
        [--allow REGEX] [--out-json diff.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys

ANSI = r"(?:\x1b\[[0-9;]*m)*"
PREFIX_ANY = re.compile(rf"{ANSI}\(TaskRunner pid=(\d+), ip=[^)]*\){ANSI} ?")  # Ray colours the prefix
START = "{'actor_rollout_ref':"
# pprint continuation: indentation, then a key / string / container / number / literal (not code lines of a warning)
CONT_RE = re.compile(r"^ +(?:['\"\[{(\-0-9]|True\b|False\b|None\b|inf\b|nan\b)")
IDENTITY = (
    r"^trainer\.experiment_name$",
    r"^trainer\.rollout_data_dir$",
    r"^trainer\.validation_data_dir$",
    r"^trainer\.default_local_dir$",
    r"^actor_rollout_ref\.rollout\.trace\.experiment_name$",
    r"^ray_kwargs\.ray_init\.runtime_env\.env_vars\.VERL_GEMMA4_ROPE_REPORT_DIR$",
)


def logical_lines(text: str):
    """(pid, content) for every TaskRunner-prefixed segment. A buffered driver can glue another prefixed line onto the
    end of a physical line, so physical lines are split at every embedded prefix."""
    for phys in text.split("\n"):
        ms = list(PREFIX_ANY.finditer(phys))
        for i, m in enumerate(ms):
            yield m.group(1), phys[m.end() : ms[i + 1].start() if i + 1 < len(ms) else len(phys)]


def _literal(text: str):
    return ast.literal_eval(re.sub(r"(?<![\w'\"])(-?inf|nan)(?![\w'\"])", r"'\1'", text))  # non-literal floats


def extract_config(path: str) -> dict:
    """The first TaskRunner-printed config dict in a driver log. pprint continuation lines are indented and start with a
    key, string, container, number or literal, so only such segments of the same TaskRunner are joined after the
    opening line (warnings, their code lines and other prints are not).
    The dict is complete at the first joined line ending in '}' that parses (braces inside strings are harmless)."""
    with open(path, errors="replace") as f:
        segs = list(logical_lines(f.read()))
    for i, (pid, content) in enumerate(segs):
        if not content.startswith(START):
            continue
        buf = [content]
        for pid2, c in segs[i + 1 :]:
            if pid2 != pid or not CONT_RE.match(c):
                continue
            buf.append(c)
            if c.rstrip().endswith("}"):
                try:
                    return _literal("\n".join(buf))
                except (SyntaxError, ValueError):
                    pass
        raise SystemExit(f"{path}: could not parse the TaskRunner config dict starting at segment {i}")
    raise SystemExit(f"{path}: no TaskRunner config dict found")


def flatten(d, prefix="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
        if not d and prefix:
            out[prefix[:-1]] = {}
    else:
        out[prefix[:-1]] = d
    return out


def parse_value(s: str):
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return s


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-log", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--expect", action="append", default=[], help="KEY=VALUE declared change (dotted key)")
    ap.add_argument(
        "--assert",
        dest="asserts",
        action="append",
        default=[],
        help="KEY=VALUE that must hold in the run whether or not it differs from the reference",
    )
    ap.add_argument("--allow", action="append", default=[], help="regex of keys allowed to differ (reported)")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args(argv)
    ref, run = flatten(extract_config(a.ref_log)), flatten(extract_config(a.log))
    expect, asserts = {}, {}
    for items, dst in ((a.expect, expect), (a.asserts, asserts)):
        for item in items:
            k, v = item.split("=", 1)
            dst[k] = parse_value(v)
    rows, fails = [], []
    for k in sorted(set(ref) | set(run) | set(asserts)):
        rv, nv = ref.get(k, "<absent>"), run.get(k, "<absent>")
        if same(rv, nv) and k not in expect and k not in asserts:
            continue
        if k in expect:
            ok = (not same(rv, nv)) and same(nv, expect[k])
            cls = "expected" if ok else "expected-MISMATCH"
        elif k in asserts and same(rv, nv):
            ok = same(nv, asserts[k])
            cls = "asserted" if ok else "asserted-MISMATCH"
        elif k in asserts:  # asserted and different from the reference: still a declared value
            ok = same(nv, asserts[k])
            cls = "asserted-changed" if ok else "asserted-MISMATCH"
        elif any(re.search(p, k) for p in IDENTITY):
            cls, ok = "identity", True
        elif any(re.search(p, k) for p in a.allow):
            cls, ok = "allowed", True
        else:
            cls, ok = "other", False
        rows.append({"key": k, "ref": rv, "run": nv, "class": cls, "ok": ok})
        if not ok:
            fails.append(k)
    for k in expect:
        if k not in ref and k not in run:
            rows.append({"key": k, "ref": "<absent>", "run": "<absent>", "class": "expected-MISSING", "ok": False})
            fails.append(k)
    for r in rows:
        print(f"[{'OK  ' if r['ok'] else 'FAIL'}] {r['class']:18s} {r['key']}: {r['ref']!r} -> {r['run']!r}")
    verdict = "PASS" if not fails else "FAIL"
    print(
        f"GEMMA4-CONFIG-DIFF: {verdict} keys_compared={len(set(ref) | set(run))} differences={len(rows)} "
        f"failures={len(fails)}"
    )
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "ref_log": a.ref_log,
                    "log": a.log,
                    "expect": a.expect,
                    "allow": a.allow,
                    "differences": rows,
                    "n_keys": len(set(ref) | set(run)),
                },
                f,
                indent=1,
                default=str,
            )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
