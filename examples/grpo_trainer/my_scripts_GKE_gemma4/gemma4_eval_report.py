#!/usr/bin/env python3
"""Summarize Gemma-4 200-step accuracy baseline runs from validation/training JSONL dumps and driver logs.

Usage:
    python3 gemma4_eval_report.py \
        --run seed1=/workspace/meta-RL/logs/gemma4_eval/<EXP_SEED1> \
        [--log seed1=/workspace/meta-RL/logs/<EXP_SEED1>.log] \
        [--expected-n 5000] [--out-json summary.json] [--out-md summary.md]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys


def _parse_kv(items: list[str]) -> dict[str, pathlib.Path]:
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"ERROR: expected label=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = pathlib.Path(v)
    return out


def _read_val_step(path: pathlib.Path, expected_n: int) -> dict:
    rows = [json.loads(line) for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if expected_n > 0 and len(rows) != expected_n:
        raise ValueError(f"{path}: expected {expected_n} rows, got {len(rows)}")
    n = len(rows)
    accs = [float(r["acc"]) for r in rows]
    fmts = [float(r["fmt"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    resp_chars = [len(r.get("output", "")) for r in rows]
    p_acc = sum(accs) / n
    p_fmt = sum(fmts) / n
    m_score = sum(scores) / n
    se_acc = math.sqrt(max(p_acc * (1.0 - p_acc), 0.0) / n)
    return {
        "step": int(path.stem),
        "n": n,
        "acc": p_acc,
        "acc_pct": 100.0 * p_acc,
        "se_pct": 100.0 * se_acc,
        "fmt": p_fmt,
        "fmt_pct": 100.0 * p_fmt,
        "score": m_score,
        "mean_output_chars": sum(resp_chars) / n,
        "_acc_vec": accs,
    }


def _paired_delta(curr: list[float], base: list[float]) -> tuple[float, float, int, int]:
    assert len(curr) == len(base)
    n = len(curr)
    diffs = [c - b for c, b in zip(curr, base)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    se_d = math.sqrt(var_d / n)
    n01 = sum(1 for c, b in zip(curr, base) if b < 0.5 <= c)
    n10 = sum(1 for c, b in zip(curr, base) if c < 0.5 <= b)
    return 100.0 * mean_d, 100.0 * se_d, n01, n10


STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")


def _parse_log_steps(log_path: pathlib.Path) -> dict[int, dict[str, float]]:
    if not log_path or not log_path.exists():
        return {}
    steps: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        metrics: dict[str, float] = {}
        for part in m.group(2).split(" - "):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            v = re.sub(r"^np\.\w+\((.*)\)$", r"\1", v.strip())
            try:
                metrics[k.strip()] = float(v)
            except ValueError:
                pass
        steps[step] = metrics
    return steps


def _summarize_run(label: str, run_dir: pathlib.Path, log_path: pathlib.Path | None, expected_n: int) -> dict:
    val_dir = run_dir / "val" if (run_dir / "val").is_dir() else run_dir
    val_files = sorted(val_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
    val_steps = []
    base_vec = None
    for vf in val_files:
        if not vf.stem.isdigit():
            continue
        rec = _read_val_step(vf, expected_n)
        if base_vec is None and rec["step"] == 0:
            base_vec = rec["_acc_vec"]
        if base_vec is not None:
            d_pct, se_d_pct, n01, n10 = _paired_delta(rec["_acc_vec"], base_vec)
            rec["delta_vs_step0_pct"] = d_pct
            rec["delta_se_pct"] = se_d_pct
            rec["gained_vs_step0"] = n01
            rec["lost_vs_step0"] = n10
        del rec["_acc_vec"]
        val_steps.append(rec)

    train_dir = run_dir / "train"
    train_steps = {}
    if train_dir.is_dir():
        for tf in sorted(train_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9):
            if not tf.stem.isdigit():
                continue
            rows = [json.loads(l) for l in tf.read_text(errors="replace").splitlines() if l.strip()]
            if not rows:
                continue
            s = int(tf.stem)
            train_steps[s] = {
                "n": len(rows),
                "acc_pct": 100.0 * sum(float(r["acc"]) for r in rows) / len(rows),
                "fmt_pct": 100.0 * sum(float(r["fmt"]) for r in rows) / len(rows),
                "score": sum(float(r["score"]) for r in rows) / len(rows),
            }

    log_steps = _parse_log_steps(log_path) if log_path else {}
    return {
        "label": label,
        "run_dir": str(run_dir),
        "log_path": str(log_path) if log_path else None,
        "val_steps": val_steps,
        "n_train_steps_dumped": len(train_steps),
        "n_train_steps_logged": len(log_steps),
        "train_steps": train_steps,
        "log_steps": log_steps,
    }


def _render_md(runs: dict[str, dict]) -> str:
    lines = ["# Gemma-4 E2B (`rl-05`) Held-Out Accuracy Report (`eval_5000`)", ""]
    for label, r in runs.items():
        lines.append(f"## Run `{label}`")
        lines.append("")
        lines.append("| Step | N | Acc (%) | Binomial SE (pp) | $\\Delta$ vs step 0 (pp) | Paired SE (pp) | +Gained / -Lost | Format (%) | Mean reward |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v in r["val_steps"]:
            d_str = f"{v['delta_vs_step0_pct']:+.2f}" if "delta_vs_step0_pct" in v else "—"
            dse_str = f"{v['delta_se_pct']:.2f}" if "delta_se_pct" in v else "—"
            gl_str = f"+{v['gained_vs_step0']} / -{v['lost_vs_step0']}" if "gained_vs_step0" in v else "—"
            lines.append(
                f"| {v['step']} | {v['n']} | **{v['acc_pct']:.2f}** | {v['se_pct']:.2f} | "
                f"{d_str} | {dse_str} | {gl_str} | {v['fmt_pct']:.2f} | {v['score']:.4f} |"
            )
        lines.append("")

    if len(runs) > 1:
        all_steps = sorted({v["step"] for r in runs.values() for v in r["val_steps"]})
        lines.append("## Multi-seed summary")
        lines.append("")
        lines.append("| Step | Seeds | Mean Acc (%) | SD across seeds (pp) | Min / Max Acc (%) | Mean Format (%) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for s in all_steps:
            vals = [v for r in runs.values() for v in r["val_steps"] if v["step"] == s]
            if not vals:
                continue
            accs = [v["acc_pct"] for v in vals]
            fmts = [v["fmt_pct"] for v in vals]
            m = sum(accs) / len(accs)
            sd = math.sqrt(sum((a - m) ** 2 for a in accs) / max(len(accs) - 1, 1)) if len(accs) > 1 else 0.0
            lines.append(
                f"| {s} | {len(vals)} | **{m:.2f}** | {sd:.2f} | {min(accs):.2f} / {max(accs):.2f} | {sum(fmts)/len(fmts):.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=/path/to/eval_dump_dir")
    ap.add_argument("--log", action="append", default=[], help="label=/path/to/driver.log")
    ap.add_argument("--expected-n", type=int, default=5000)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    run_map = _parse_kv(args.run)
    log_map = _parse_kv(args.log)
    runs = {
        label: _summarize_run(label, rdir, log_map.get(label), args.expected_n)
        for label, rdir in run_map.items()
    }
    md = _render_md(runs)
    print(md)
    if args.out_md:
        p = pathlib.Path(args.out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md)
    if args.out_json:
        p = pathlib.Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(runs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
