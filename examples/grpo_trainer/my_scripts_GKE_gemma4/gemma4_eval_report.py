#!/usr/bin/env python3
"""Summarize Gemma-4 accuracy runs from validation/training JSONL dumps and driver logs.

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


def _rolling(arr: list[float], window: int = 15) -> list[float]:
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out.append(sum(arr[lo : i + 1]) / (i - lo + 1))
    return out


def _plot_run(label: str, data: dict, plot_dir: pathlib.Path) -> list[pathlib.Path]:
    import statistics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[pathlib.Path] = []
    val_steps = data.get("val_steps", [])
    if not val_steps:
        return saved

    steps_v = [v["step"] for v in val_steps]
    acc_v = [v["acc_pct"] for v in val_steps]
    se_v = [v["se_pct"] for v in val_steps]
    delta_v = [v.get("delta_vs_step0_pct", 0.0) for v in val_steps]
    delta_se_v = [v.get("delta_se_pct", 0.0) for v in val_steps]
    fmt_v = [v["fmt_pct"] for v in val_steps]
    score_v = [v["score"] for v in val_steps]
    gained_v = [v.get("gained_vs_step0", 0) for v in val_steps]
    lost_v = [v.get("lost_vs_step0", 0) for v in val_steps]
    net_q_v = [g - l for g, l in zip(gained_v, lost_v)]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.35,
    })

    # Figure 1: Held-out eval_5000 curve (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    fig.suptitle(
        f"Gemma-4 E2B (rl-05) GPU Reference ({label}, steps {steps_v[0]}–{steps_v[-1]}) — Held-Out eval_5000 (N = 5,000, Greedy T = 0)",
        fontsize=12.5, fontweight="bold", y=0.98,
    )

    ax = axes[0, 0]
    ax.errorbar(
        steps_v, acc_v, yerr=se_v, fmt="-o", color="#1f77b4", ecolor="#7fb3d5",
        elinewidth=1.5, capsize=4, markersize=6, linewidth=2.0,
        label=f"Greedy Accuracy ± 1 Binomial SE ({se_v[0]:.2f} pp)",
    )
    ax.axhline(acc_v[0], color="#555555", linestyle="--", linewidth=1.2, label=f"Step 0 Base Model: {acc_v[0]:.2f}%")
    if len(acc_v) >= 4:
        plateau_mean = statistics.mean(acc_v[3:])
        ax.hlines(
            plateau_mean, steps_v[3], steps_v[-1], color="#2ca02c", linestyle="-.", linewidth=1.6,
            label=f"Steps {steps_v[3]}–{steps_v[-1]} Mean: {plateau_mean:.2f}% ({plateau_mean - acc_v[0]:+.2f} pp)",
        )
    for s, a in zip(steps_v, acc_v):
        offset = 8 if s not in (20, 100, 140, 200) else -14
        ax.annotate(
            f"{a:.2f}%", (s, a), textcoords="offset points", xytext=(0, offset),
            ha="center", fontsize=8.5, fontweight="bold" if s in (0, 60, 120, 200) else "normal",
        )
    ax.set_title("A. Held-Out Greedy Accuracy (%) on eval_5000")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(steps_v)
    ax.set_ylim(min(acc_v) - 1.3, max(acc_v) + 1.2)
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[0, 1]
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2, label="Step 0 Reference (0.00 pp)")
    if len(steps_v) > 1:
        ax.errorbar(
            steps_v[1:], delta_v[1:], yerr=[1.96 * x for x in delta_se_v[1:]],
            fmt="none", ecolor="#aec7e8", elinewidth=1.5, capsize=3, label="±1.96 Paired SE (95% CI)",
        )
        ax.errorbar(
            steps_v[1:], delta_v[1:], yerr=delta_se_v[1:], fmt="-s", color="#d62728",
            ecolor="#d62728", elinewidth=2.0, capsize=4, markersize=6, linewidth=2.0,
            label="Paired ΔAcc vs Step 0 ± 1 Paired SE",
        )
        for s, d, g, l in zip(steps_v[1:], delta_v[1:], gained_v[1:], lost_v[1:]):
            ax.annotate(
                f"{d:+.2f} pp\n(+{g}/-{l})", (s, d), textcoords="offset points",
                xytext=(0, 10 if d >= 0.6 else -22), ha="center", fontsize=7.8,
            )
    ax.set_title("B. Paired Accuracy Gain ΔAcc(t) = Acc(t) − Acc(0) on Identical 5,000 Questions")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Paired Δ Accuracy vs Step 0 (pp)")
    ax.set_xticks(steps_v)
    ax.set_ylim(min(delta_v) - 1.1, max(delta_v) + 1.3)
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(steps_v, fmt_v, "-^", color="#2ca02c", linewidth=2.0, markersize=6, label="Format Compliance (%) [left]")
    l2 = ax2.plot(steps_v, score_v, "-D", color="#9467bd", linewidth=2.0, markersize=5, label="Mean Total Reward (acc + 0.1·fmt) [right]")
    for s, f in zip(steps_v, fmt_v):
        if s in (0, 20, 40, 100, 140, 180, 200):
            ax.annotate(f"{f:.2f}%", (s, f), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#1b661b")
    ax.set_title("C. Held-Out Format Compliance (%) & Mean Reward")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Format Compliance (%)", color="#2ca02c")
    ax2.set_ylabel("Mean Reward (score = acc + 0.1·fmt)", color="#9467bd")
    ax.set_xticks(steps_v)
    ax.set_ylim(min(fmt_v) - 0.5, max(fmt_v) + 0.7)
    ax2.set_ylim(min(score_v) - 0.005, max(score_v) + 0.005)
    ax2.grid(False)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8.5)

    ax = axes[1, 1]
    if len(steps_v) > 1:
        x = np.array(steps_v[1:])
        w = 5.5
        ax.bar(x - w / 2, gained_v[1:], width=w, color="#2ca02c", alpha=0.8, label="+Gained (0 → 1 vs Step 0)")
        ax.bar(x + w / 2, lost_v[1:], width=w, color="#d62728", alpha=0.8, label="-Lost (1 → 0 vs Step 0)")
        ax_net = ax.twinx()
        ax_net.plot(x, net_q_v[1:], "-o", color="#111111", linewidth=2.0, markersize=5, label="Net Questions Gained (+Gained − Lost) [right]")
        for sx, nq in zip(x, net_q_v[1:]):
            ax_net.annotate(f"{nq:+d}", (sx, nq), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(steps_v[1:])
        ax.set_ylim(min(lost_v[1:] + gained_v[1:]) - 50, max(lost_v[1:] + gained_v[1:]) + 30)
        ax_net.set_ylim(min(net_q_v[1:]) - 15, max(net_q_v[1:]) + 20)
        ax_net.set_ylabel("Net Questions Gained vs Step 0")
        ax_net.grid(False)
        h1, lab1 = ax.get_legend_handles_labels()
        h2, lab2 = ax_net.get_legend_handles_labels()
        ax.legend(h1 + h2, lab1 + lab2, loc="lower right", fontsize=8.0)
    ax.set_title("D. Per-Question Transitions vs Step 0 (+Gained / -Lost out of 5,000)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Number of Questions (out of 5,000)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = plot_dir / f"{label}_eval_curve.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved.append(p1)

    # Figure 2: Training & guard diagnostics
    train_steps = {int(k): v for k, v in data.get("train_steps", {}).items()}
    log_steps = {int(k): v for k, v in data.get("log_steps", {}).items() if int(k) >= 1}
    ts = sorted(set(train_steps.keys()) & set(log_steps.keys()))
    if not ts:
        return saved

    tr_acc = [train_steps[s]["acc_pct"] for s in ts]
    tr_fmt = [train_steps[s]["fmt_pct"] for s in ts]
    k3_kl = [log_steps[s]["rollout_corr/k3_kl"] for s in ts]
    pearson = [log_steps[s]["training/rollout_actor_probs_pearson_corr"] for s in ts]
    grad_norm = [log_steps[s]["actor/grad_norm"] for s in ts]
    entropy = [log_steps[s]["actor/entropy"] for s in ts]
    resp_len = [log_steps[s]["response_length/mean"] for s in ts]
    step_s = [log_steps[s]["timing_s/step"] for s in ts]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    fig.suptitle(
        f"Gemma-4 E2B (rl-05) GPU Reference ({label}, steps {ts[0]}–{ts[-1]}) — Training & Numerical Guard Diagnostics (16n64g)",
        fontsize=12.5, fontweight="bold", y=0.98,
    )

    ax = axes[0, 0]
    n_rows = sorted({int(train_steps[s].get("n", 0)) for s in ts})
    rows_lbl = f"{n_rows[0]:,}" if len(n_rows) == 1 else "/".join(f"{n:,}" for n in n_rows)
    ax.plot(ts, tr_acc, color="#1f77b4", alpha=0.30, linewidth=1.0,
            label=f"Per-step Train Accuracy (T={data.get('train_temp', 0.8):g}, {rows_lbl} completions)")
    ax.plot(ts, _rolling(tr_acc, 20), color="#1f77b4", linewidth=2.2, label=f"20-step Rolling Mean Acc ({statistics.mean(tr_acc[:20]):.2f}% → {statistics.mean(tr_acc[-20:]):.2f}%)")
    ax2 = ax.twinx()
    ax2.plot(ts, _rolling(tr_fmt, 20), color="#2ca02c", linestyle="--", linewidth=1.8, label=f"20-step Rolling Format ({statistics.mean(tr_fmt[:20]):.2f}% → {statistics.mean(tr_fmt[-20:]):.2f}%) [right]")
    ax.set_title(f"A. Online Training Rollout Accuracy & Format Rate (Steps {ts[0]}–{ts[-1]})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sampled Train Accuracy (%)", color="#1f77b4")
    ax2.set_ylabel("Sampled Format Compliance (%)", color="#2ca02c")
    ax.set_ylim(50, 74)
    ax2.set_ylim(95, 100.0)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.2)

    ax = axes[0, 1]
    ax.plot(ts, [x * 1e3 for x in k3_kl], color="#d62728", linewidth=1.3, alpha=0.85, label=f"rollout_corr/k3_kl (×10⁻³): med={statistics.median(k3_kl)*1e3:.3f}, range=[{min(k3_kl)*1e3:.3f}, {max(k3_kl)*1e3:.3f}]")
    ax.axhline(statistics.median(k3_kl) * 1e3, color="#8c1515", linestyle="--", linewidth=1.2)
    ax2 = ax.twinx()
    ax2.plot(ts, pearson, color="#1f77b4", linewidth=1.2, alpha=0.75, label=f"Pearson(p_train, p_rollout): med={statistics.median(pearson):.5f} [right]")
    ax.set_title(f"B. Trainer–Sampler Log-Prob Parity (Steps {ts[0]}–{ts[-1]}; Pre-Fix k3_kl ~980–1350 ×10⁻³)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("rollout_corr/k3_kl (× 10⁻³)", color="#d62728")
    ax2.set_ylabel("Pearson Correlation", color="#1f77b4")
    ax.set_ylim(0.55, 0.85)
    ax2.set_ylim(0.9968, 0.9976)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.0)

    ax = axes[1, 0]
    ax.plot(ts, grad_norm, color="#ff7f0e", alpha=0.55, linewidth=1.0, label=f"actor/grad_norm (med={statistics.median(grad_norm):.3f}, max={max(grad_norm):.2f})")
    ax.plot(ts, _rolling(grad_norm, 15), color="#d95f02", linewidth=2.0, label="15-step Rolling Mean grad_norm")
    ax2 = ax.twinx()
    ax2.plot(ts, entropy, color="#9467bd", linewidth=1.6, label=f"actor/entropy ({entropy[0]:.3f} → {entropy[-1]:.3f}) [right]")
    ax.set_title(f"C. Actor Gradient Norm & Policy Entropy (Steps {ts[0]}–{ts[-1]})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("actor/grad_norm (post-fix; pre-fix was 10–350)", color="#d95f02")
    ax2.set_ylabel("actor/entropy", color="#9467bd")
    ax.set_ylim(0, max(3.6, max(grad_norm) * 1.1))
    ax2.set_ylim(0.15, 0.25)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.2)

    ax = axes[1, 1]
    ax.plot(ts, resp_len, color="#2ca02c", alpha=0.45, linewidth=1.0)
    ax.plot(ts, _rolling(resp_len, 15), color="#1b7837", linewidth=2.0, label=f"Mean Response Length (med={statistics.median(resp_len):.0f} tok)")
    ax2 = ax.twinx()
    if len(ts) > 1:
        ax2.plot(ts[1:], step_s[1:], color="#555555", alpha=0.55, linewidth=1.2, label=f"timing_s/step (steps {ts[1]}–{ts[-1]} med={statistics.median(step_s[1:]):.1f}s, p90={sorted(step_s[1:])[int(0.9*len(step_s[1:]))]:.1f}s) [right]")
    ax.set_title("D. Mean Response Length (Tokens) & Trainer Step Time (s)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Response Length (tokens)", color="#1b7837")
    ax2.set_ylabel("timing_s/step (s)", color="#555555")
    ax.margins(y=0.08)  # Keep every response-length point visible as the recipe changes.
    ax2.set_ylim(50, 115)
    # Annotate step times above the clipped axis instead of silently cutting them off.
    gen_s = [log_steps[s].get("timing_s/gen", float("nan")) for s in ts]
    for s, v, g in zip(ts[1:], step_s[1:], gen_s[1:]):
        if v > 115:
            ax2.plot([s], [114], marker="^", color="#d62728", markersize=7, clip_on=False)
            ax2.annotate(f"step {s}: {v:.0f}s (rollout {g:.0f}s), off scale", (s, 114), textcoords="offset points",
                         xytext=(-215, -30), fontsize=7.5, color="#d62728")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = plot_dir / f"{label}_training_diagnostics.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved.append(p2)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="label=/path/to/eval_dump_dir")
    ap.add_argument("--log", action="append", default=[], help="label=/path/to/driver.log")
    ap.add_argument("--from-json", default="", help="Load pre-computed summary JSON instead of re-reading dumps")
    ap.add_argument("--max-step", type=int, default=None, help="Report and plot only steps at or below this value")
    ap.add_argument("--expected-n", type=int, default=5000)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--plot-dir", default="", help="Directory to write <label>_eval_curve.png and <label>_training_diagnostics.png")
    ap.add_argument("--train-temp", action="append", default=[],
                    help="label=T: training rollout temperature for plot labels (default 0.8)")
    args = ap.parse_args()

    if args.from_json:
        runs = json.loads(pathlib.Path(args.from_json).read_text())
    else:
        if not args.run:
            sys.exit("ERROR: either --run or --from-json is required")
        run_map = _parse_kv(args.run)
        log_map = _parse_kv(args.log)
        runs = {
            label: _summarize_run(label, rdir, log_map.get(label), args.expected_n)
            for label, rdir in run_map.items()
        }
    if args.max_step is not None:
        for data in runs.values():
            data["val_steps"] = [v for v in data["val_steps"] if v["step"] <= args.max_step]
            for key in ("train_steps", "log_steps"):
                data[key] = {s: v for s, v in data.get(key, {}).items() if int(s) <= args.max_step}
            data["n_train_steps_dumped"] = len(data["train_steps"])
            data["n_train_steps_logged"] = len(data["log_steps"])
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
    if args.plot_dir:
        pdir = pathlib.Path(args.plot_dir)
        temps = {k: float(v) for k, v in (t.split("=", 1) for t in args.train_temp)}
        for label, rdata in runs.items():
            rdata["train_temp"] = temps.get(label, rdata.get("train_temp", 0.8))
            _plot_run(label, rdata, pdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
