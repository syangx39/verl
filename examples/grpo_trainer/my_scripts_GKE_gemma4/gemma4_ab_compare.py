#!/usr/bin/env python3
"""Paired comparison of two Gemma-4 runs that share seed / data order (PR #10 item 3: BF16 vs FP32 master).

Both runs must be on the same eval set and the same training data order (same data.seed), so that step k trains on
the same 256 prompts in both. The script checks that, then reports:

  1. Held-out eval at every common checkpoint, paired by validated `qid` (unique ids, identical input and
     ground-truth hashes): accuracy of each run, B - A difference, paired SE, question-paired bootstrap 95% interval,
     gained / lost, and each run's own change vs its step 0. Step 0 scores identical weights in both runs, so the
     step-0 difference is the eval noise floor.
  2. Training accuracy (sampled rollouts, T = 0.8 unless --a-temp / --b-temp say otherwise) per step on identical
     prompts: B - A per step, and its mean, standard deviation and number of steps with B ahead over step windows. These
     are descriptive: consecutive steps of the same two runs are not independent replicates (each run's state carries
     over), so no standard error is reported.
     Before comparing, every training dump is validated: complete step coverage, exactly rows_per_step rows,
     rows_per_step / group_size prompt groups of exactly group_size rows, one input and one ground truth per group,
     and the same qid -> (input, ground truth) mapping in both runs.
  3. Per-step diagnostics from the driver logs: rollout_corr/k3_kl, actor/grad_norm, actor/entropy,
     response_length/mean, critic/score/mean, timing_s/step (medians over windows).

With --train-template-change strip_inner_gemma3 (PR #11 A2), B's training prompts are allowed to differ from A's by
exactly that declared template change: A's input is mapped to its expected B form before the same exact equality check,
so a different question or instruction still fails. Eval prompts are always compared strictly.

Runs with different completions per prompt (PR #12 A6: n=16 vs the n=8 control) are compared with per-run batch
shapes, e.g. --a-rows-per-step 2048 --a-group-size 8 --b-rows-per-step 4096 --b-group-size 16. Both runs must have the
same number of prompt groups per step and the same qid -> (input, ground truth) per step; each side is validated
against its own declared shape (no identity check is relaxed). Training accuracy is then each run's mean over all of
its completions on the same prompt instances.

Usage:
    python3 gemma4_ab_compare.py --a bf16=<eval_dump_dir_A> --a-log <driver_log_A> \\
        --b fp32=<eval_dump_dir_B> --b-log <driver_log_B> [--max-step 60] [--out-json x.json] [--out-md x.md] \\
        [--train-template-change strip_inner_gemma3] [--b-rows-per-step 4096 --b-group-size 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys


def _rows(path):
    with open(path, errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _h(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8", "replace")).hexdigest()[:16]


def _steps(d, sub):
    p = os.path.join(d, sub)
    return sorted(int(f[:-6]) for f in os.listdir(p) if f.endswith(".jsonl") and f[:-6].isdigit()) if os.path.isdir(p) else []


def _load_val(path):
    rows = list(_rows(path))
    out = {str(r["qid"]): (float(r["acc"]), float(r["fmt"]), _h(r.get("input")), _h(r.get("gts")), r.get("output", ""))
           for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{path}: duplicate qids")
    return out


def _paired(a, b, B=2000, seed=0):
    import numpy as np

    qids = sorted(a)
    if set(qids) != set(b):
        raise ValueError("qid sets differ")
    bad = [q for q in qids if a[q][2:4] != b[q][2:4]]
    if bad:
        raise ValueError(f"{len(bad)} qids with different input/ground truth")
    d = np.array([b[q][0] - a[q][0] for q in qids])
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"acc_a": 100 * float(np.mean([a[q][0] for q in qids])), "acc_b": 100 * float(np.mean([b[q][0] for q in qids])),
            "fmt_a": 100 * float(np.mean([a[q][1] for q in qids])), "fmt_b": 100 * float(np.mean([b[q][1] for q in qids])),
            "delta_pp": 100 * float(d.mean()), "se_pp": 100 * float(d.std(ddof=1) / math.sqrt(len(d))),
            "boot95_pp": [100 * float(lo), 100 * float(hi)], "gained": int((d > 0).sum()), "lost": int((d < 0).sum()),
            "identical_outputs": int(sum(a[q][4] == b[q][4] for q in qids)), "n": len(qids)}


STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")

# --train-template-change strip_inner_gemma3 (PR #11 A2). verl's rollout dump stores the prompt decoded with
# skip_special_tokens=True, so the reference (double-wrapped) and A2 (single-wrapped) training prompts read
#   A: "user\n<start_of_turn>user\n" + core + "<end_of_turn>\n<start_of_turn>model\nmodel\n"
#   B: "user\n" + core.strip() + "\nmodel\n"
# with the same core (instructions + question). .strip() is the Gemma-4 chat template's own `message['content'] | trim`:
# with the inner markers gone, whitespace at the end of a question (~0.15% of rows) is no longer shielded by them.
# The mode maps every A input to its expected B form and then applies the same exact-equality check as the strict
# mode; any other difference (question, instructions) still fails.
_TEMPLATE_CHANGES = {
    "strip_inner_gemma3": ("user\n<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\nmodel\n", "user\n", "\nmodel\n"),
}


def _template_transform(name):
    if not name or name == "none":
        return None
    pre_a, suf_a, pre_b, suf_b = _TEMPLATE_CHANGES[name]

    def f(s):
        s = "" if s is None else str(s)
        if not (s.startswith(pre_a) and s.endswith(suf_a) and len(s) >= len(pre_a) + len(suf_a)):
            raise SystemExit(f"[train] A input is not in the form expected by --train-template-change {name}: {s[:80]!r}")
        return pre_b + s[len(pre_a):len(s) - len(suf_a)].strip() + suf_b
    return f


def _check_batch(rows, where, rows_per_step, group_size, transform=None):
    """Validate one training dump; return {qid: (input_hash, ground_truth_hash)}. Exits on any violation.
    With `transform`, the input hash is taken of transform(input) (A side of a declared template change)."""
    if len(rows) != rows_per_step:
        raise SystemExit(f"[train] {where}: {len(rows)} rows, expected {rows_per_step}")
    groups = {}
    for r in rows:
        groups.setdefault(str(r["qid"]), []).append(r)
    n_groups = rows_per_step // group_size
    if len(groups) != n_groups:
        raise SystemExit(f"[train] {where}: {len(groups)} prompt groups, expected {n_groups}")
    out = {}
    for q, g in groups.items():
        if len(g) != group_size:
            raise SystemExit(f"[train] {where}: qid {q} has {len(g)} rows, expected {group_size}")
        ins = {_h(transform(r.get("input")) if transform else r.get("input")) for r in g}
        gts = {_h(r.get("gts")) for r in g}
        if len(ins) != 1 or len(gts) != 1:
            raise SystemExit(f"[train] {where}: qid {q} mixes inputs or ground truths inside its group")
        out[q] = (ins.pop(), gts.pop())
    return out
KEYS = ("rollout_corr/k3_kl", "actor/grad_norm", "actor/entropy", "response_length/mean", "critic/score/mean",
        "timing_s/step", "response_length/clip_ratio")


def _log_steps(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = STEP_RE.search(line)
        if not m:
            continue
        kv = {}
        for part in m.group(2).split(" - "):
            if ":" in part:
                k, v = part.split(":", 1)
                v = re.sub(r"^np\.\w+\((.*)\)$", r"\1", v.strip())
                try:
                    kv[k.strip()] = float(v)
                except ValueError:
                    pass
        out[int(m.group(1))] = kv
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="label=eval_dump_dir (reference)")
    ap.add_argument("--b", required=True, help="label=eval_dump_dir (variant)")
    ap.add_argument("--a-log", default="")
    ap.add_argument("--b-log", default="")
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--expect-steps", type=int, default=0,
                    help="require training dumps for steps 1..N in both runs (default: 1..last step present)")
    ap.add_argument("--rows-per-step", type=int, default=2048, help="train_batch_size x rollout.n (both runs)")
    ap.add_argument("--group-size", type=int, default=8, help="rollout.n (both runs)")
    for side in ("a", "b"):
        ap.add_argument(f"--{side}-rows-per-step", type=int, default=0,
                        help=f"run {side.upper()}'s rows per step if it differs (default: --rows-per-step)")
        ap.add_argument(f"--{side}-group-size", type=int, default=0,
                        help=f"run {side.upper()}'s rollout.n if it differs (default: --group-size)")
        ap.add_argument(f"--{side}-temp", type=float, default=0.8,
                        help=f"run {side.upper()}'s training temperature (labels only)")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--plot", default="", help="write a 2x2 PNG summary here (needs matplotlib)")
    ap.add_argument("--replicate", action="append", default=[],
                    help="label=eval_dump_dir of another run on the same eval set; its step-0 accuracy joins the "
                         "identical-weights band in the plot")
    ap.add_argument("--train-template-change", default="none", choices=["none", *_TEMPLATE_CHANGES],
                    help="declared training-prompt template change between A and B (eval prompts stay strict): "
                         "strip_inner_gemma3 = B's training prompts are A's with the inner Gemma-3 turn template removed")
    args = ap.parse_args()
    la, da = args.a.split("=", 1)
    lb, db = args.b.split("=", 1)
    tmpl = _template_transform(args.train_template_change)
    same_prompts = tmpl is None
    a_rows, a_g = args.a_rows_per_step or args.rows_per_step, args.a_group_size or args.group_size
    b_rows, b_g = args.b_rows_per_step or args.rows_per_step, args.b_group_size or args.group_size
    for side, rows, g in (("A", a_rows, a_g), ("B", b_rows, b_g)):
        if g < 1 or rows % g:
            raise SystemExit(f"[train] {side}: rows per step {rows} is not a multiple of group size {g}")
    if a_rows // a_g != b_rows // b_g:
        raise SystemExit(f"[train] A has {a_rows // a_g} prompt groups per step but B has {b_rows // b_g}: "
                         "the runs must train on the same prompt instances")
    n_groups = a_rows // a_g
    same_shape = (a_rows, a_g) == (b_rows, b_g)

    res = {"a": {"label": la, "dir": da}, "b": {"label": lb, "dir": db}, "eval": {}, "train": {}, "diag": {},
           "train_template_change": args.train_template_change, "train_temps": (args.a_temp, args.b_temp)}
    va = {s: _load_val(os.path.join(da, "val", f"{s}.jsonl")) for s in _steps(da, "val") if s <= args.max_step}
    vb = {s: _load_val(os.path.join(db, "val", f"{s}.jsonl")) for s in _steps(db, "val") if s <= args.max_step}
    common = sorted(set(va) & set(vb))
    for s in common:
        res["eval"][s] = {"b_minus_a": _paired(va[s], vb[s], seed=s)}
        if s and 0 in va and 0 in vb:
            res["eval"][s]["a_vs_step0"] = _paired(va[0], va[s], seed=100 + s)
            res["eval"][s]["b_vs_step0"] = _paired(vb[0], vb[s], seed=200 + s)

    sa = [s for s in _steps(da, "train") if s <= args.max_step]
    sb = [s for s in _steps(db, "train") if s <= args.max_step]
    tsteps = sorted(set(sa) & set(sb))
    expected = list(range(1, (min(args.max_step, sa[-1], sb[-1]) if sa and sb else 0) + 1))
    if args.expect_steps:
        expected = list(range(1, args.expect_steps + 1))
    missing = {"a": sorted(set(expected) - set(sa)), "b": sorted(set(expected) - set(sb))}
    if missing["a"] or missing["b"]:
        raise SystemExit(f"[train] incomplete step coverage: missing in A {missing['a'][:10]}, in B {missing['b'][:10]}")
    per = {}
    for s in tsteps:
        ra = list(_rows(os.path.join(da, "train", f"{s}.jsonl")))
        rb = list(_rows(os.path.join(db, "train", f"{s}.jsonl")))
        ga_ = _check_batch(ra, f"A step {s}", a_rows, a_g, transform=tmpl)
        gb_ = _check_batch(rb, f"B step {s}", b_rows, b_g)
        if set(ga_) != set(gb_):
            raise SystemExit(f"[train] step {s}: prompt qid sets differ between A and B")
        bad = [q for q in ga_ if ga_[q] != gb_[q]]
        if bad:
            what = "input/ground truth" if same_prompts else f"input (after {args.train_template_change}) / ground truth"
            raise SystemExit(f"[train] step {s}: {len(bad)} qids with different {what} between A and B")
        acc = lambda rr: 100 * sum(float(r["acc"]) for r in rr) / len(rr)  # noqa: E731
        fmt = lambda rr: 100 * sum(float(r["fmt"]) for r in rr) / len(rr)  # noqa: E731
        per[s] = {"same_prompts": same_prompts, "acc_a": acc(ra), "acc_b": acc(rb), "fmt_a": fmt(ra), "fmt_b": fmt(rb)}
    res["train"]["per_step"] = per
    res["train"]["batch_checks"] = {
        "steps": [tsteps[0], tsteps[-1]] if tsteps else None, "n_steps": len(tsteps),
        "rows_per_step": a_rows if same_shape else None, "groups_per_step": n_groups,
        "group_size": a_g if same_shape else None,
        "a": {"rows_per_step": a_rows, "group_size": a_g}, "b": {"rows_per_step": b_rows, "group_size": b_g},
        "checks": "complete step coverage; exact rows per step; groups of exactly group_size per qid (per run); one "
                  "input and one ground truth per group; identical qid -> (input, ground truth) between A and B"
                  + ("" if same_prompts else f", where A's input is first mapped by the declared template change "
                                              f"{args.train_template_change}"),
    }

    def window(lo, hi):
        sel = [per[s] for s in tsteps if lo <= s <= hi]
        if len(sel) < 2:
            return None
        d = [x["acc_b"] - x["acc_a"] for x in sel]
        # Descriptive spread only: consecutive steps of the same two runs are not independent replicates.
        return {"steps": [lo, hi], "n": len(sel), "acc_a": statistics.mean(x["acc_a"] for x in sel),
                "acc_b": statistics.mean(x["acc_b"] for x in sel), "diff_mean": statistics.mean(d),
                "diff_sd": statistics.stdev(d), "n_positive": sum(1 for x in d if x > 0),
                "fmt_a": statistics.mean(x["fmt_a"] for x in sel), "fmt_b": statistics.mean(x["fmt_b"] for x in sel)}

    def twenty_step_windows(last):
        # (1, 20), (21, 40), (41, 60), ... with the last window clipped to `last` (60-, 100- and 200-step runs alike).
        return [(lo, min(lo + 19, last)) for lo in range(1, last + 1, 20)]

    if tsteps:
        last = tsteps[-1]
        wins = [(1, min(10, last)), (11, min(20, last))] + twenty_step_windows(last)[1:] + [(1, last)]
        wins = list(dict.fromkeys((lo, hi) for lo, hi in wins if hi >= lo))
        res["train"]["windows"] = [w for w in (window(lo, hi) for lo, hi in wins) if w]

    ga, gb = _log_steps(args.a_log), _log_steps(args.b_log)
    log_last = min(max(ga, default=0), max(gb, default=0), args.max_step)
    for lo, hi in twenty_step_windows(log_last):
        row = {}
        for k in KEYS:
            xa = [ga[s][k] for s in ga if lo <= s <= hi and s <= args.max_step and k in ga[s]]
            xb = [gb[s][k] for s in gb if lo <= s <= hi and s <= args.max_step and k in gb[s]]
            if xa and xb:
                row[k] = {"a_median": statistics.median(xa), "b_median": statistics.median(xb), "n_a": len(xa), "n_b": len(xb)}
        if row:
            res["diag"][f"{lo}-{hi}"] = row

    prompt_label = "identical prompts" if same_prompts else "same raw questions, different training templates"
    if not same_shape:
        prompt_label += f", n={a_g} (A) vs n={b_g} (B)"
    if args.a_temp != args.b_temp:
        prompt_label += f", T={args.a_temp:g} (A) vs T={args.b_temp:g} (B)"
    shape = (f"{a_rows} rows = {n_groups} x {a_g}" if same_shape else
             f"A {a_rows} rows = {n_groups} x {a_g}, B {b_rows} rows = {n_groups} x {b_g}")
    L = [f"### Paired A/B: `{lb}` (B) vs `{la}` (A)", ""]
    L.append(f"- A: `{da}`  B: `{db}`; eval checkpoints in common: {common}; training steps in common: "
             f"{tsteps[0] if tsteps else '-'}–{tsteps[-1] if tsteps else '-'}; training dumps validated at every step "
             f"({shape}, one input / ground "
             f"truth per group, identical qid -> input / ground truth in A and B"
             + ("" if same_prompts else f" after the declared training-template change `{args.train_template_change}`; "
                                        "eval prompts compared strictly") + ").")
    L.append("")
    L.append("| Eval step | A acc (%) | B acc (%) | B − A (pp) | paired SE | bootstrap 95% | +gained / −lost (B vs A) | "
             "A vs own step 0 (pp) | B vs own step 0 (pp, 95%) | identical outputs |")
    L.append("|---:|---:|---:|---:|---:|---|---|---:|---|---:|")
    for s in common:
        e = res["eval"][s]
        x = e["b_minus_a"]
        a0 = f"{e['a_vs_step0']['delta_pp']:+.2f}" if "a_vs_step0" in e else "—"
        b0 = (f"{e['b_vs_step0']['delta_pp']:+.2f} [{e['b_vs_step0']['boot95_pp'][0]:+.2f}, {e['b_vs_step0']['boot95_pp'][1]:+.2f}]"
              if "b_vs_step0" in e else "—")
        L.append(f"| {s} | {x['acc_a']:.2f} | {x['acc_b']:.2f} | **{x['delta_pp']:+.2f}** | {x['se_pp']:.2f} | "
                 f"[{x['boot95_pp'][0]:+.2f}, {x['boot95_pp'][1]:+.2f}] | +{x['gained']} / −{x['lost']} | {a0} | {b0} | "
                 f"{x['identical_outputs']} / {x['n']} |")
    L.append("")
    if res["train"].get("windows"):
        L.append(f"| Training steps ({prompt_label}) | n | A train acc (%) | B train acc (%) | B − A mean (pp) | SD of per-step B − A (pp) | steps with B ahead | "
                 "A format (%) | B format (%) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for w in res["train"]["windows"]:
            L.append(f"| {w['steps'][0]}–{w['steps'][1]} | {w['n']} | {w['acc_a']:.2f} | {w['acc_b']:.2f} | "
                     f"**{w['diff_mean']:+.2f}** | {w['diff_sd']:.2f} | {w['n_positive']} / {w['n']} | {w['fmt_a']:.2f} | {w['fmt_b']:.2f} |")
        L.append("")
    if res["diag"]:
        L.append("| Steps | metric | A median | B median |")
        L.append("|---|---|---:|---:|")
        for win, row in res["diag"].items():
            for k, v in row.items():
                L.append(f"| {win} | `{k}` | {v['a_median']:.4g} | {v['b_median']:.4g} |")
    md = "\n".join(L) + "\n"
    print(md)
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=1, sort_keys=True)
    if args.out_md:
        open(args.out_md, "w").write(md)
    if args.plot:
        rep = {}
        for item in args.replicate:
            lab, d = item.split("=", 1)
            p = os.path.join(d, "val", "0.jsonl")
            if os.path.exists(p):
                rows = list(_rows(p))
                rep[lab] = 100 * sum(float(r["acc"]) for r in rows) / len(rows)
        _plot(res, la, lb, ga, gb, rep, args.plot, prompt_label)
    return 0


def _rolling(x, w):
    return [sum(x[max(0, i - w + 1): i + 1]) / (i - max(0, i - w + 1) + 1) for i in range(len(x))]


def _sampling_label(res):
    """'T=0.8, 2,048 completions/step' when both runs share it, else per run (n and T are also in the prompt label)."""
    bc = res["train"].get("batch_checks", {})
    ra, rb = bc.get("a", {}).get("rows_per_step"), bc.get("b", {}).get("rows_per_step")
    ta, tb = res.get("train_temps", (0.8, 0.8))
    t = f"T={ta:g}" if ta == tb else f"T={ta:g} / {tb:g}"
    c = f"{ra:,} completions/step" if ra == rb else f"{ra:,} / {rb:,} completions/step"
    return f"{t}, {c}"


def _plot(res, la, lb, ga, gb, rep, path, prompt_label="identical prompts"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.35, "axes.facecolor": "#fafafa"})
    ca, cb = "#7f7f7f", "#1f77b4"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=150)
    fig.suptitle(f"Gemma-4 E2B rl-05, seed 1, 16n64g: {lb} vs {la} ({prompt_label} per step)", fontsize=13,
                 fontweight="bold")

    ax = axes[0, 0]
    es = sorted(res["eval"])
    ya = {s: res["eval"][s]["b_minus_a"]["acc_a"] for s in es}
    yb = {s: res["eval"][s]["b_minus_a"]["acc_b"] for s in es}
    for lab, ys_d, other, c in ((la, ya, yb, ca), (lb, yb, ya, cb)):
        ys = [ys_d[s] for s in es]
        se = [100 * math.sqrt((y / 100) * (1 - y / 100) / 5000) for y in ys]
        ax.errorbar(es, ys, yerr=se, fmt="-o", color=c, capsize=4, lw=2, label=f"{lab} (± binomial SE)")
        for s, y in zip(es, ys):
            ax.annotate(f"{y:.2f}", (s, y), textcoords="offset points", xytext=(14, 6 if y >= other[s] else -14),
                        ha="left", fontsize=8.5, color=c)
    s0 = [res["eval"][0]["b_minus_a"]["acc_a"], res["eval"][0]["b_minus_a"]["acc_b"], *rep.values()] if 0 in res["eval"] else []
    if s0:
        ax.axhspan(min(s0), max(s0), color="#ffbb78", alpha=0.35,
                   label=f"step-0 evals of identical weights ({len(s0)} runs): {min(s0):.2f}–{max(s0):.2f}")
    ax.set_title("A. Held-out greedy accuracy on eval_5000 (paired by qid)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(es)
    ax.legend(loc="upper left", fontsize=8.5)

    per = res["train"]["per_step"]
    ts = sorted(per)
    aa = [per[s]["acc_a"] for s in ts]
    bb = [per[s]["acc_b"] for s in ts]
    ax = axes[0, 1]
    ax.plot(ts, aa, color=ca, alpha=0.25, lw=1)
    ax.plot(ts, bb, color=cb, alpha=0.25, lw=1)
    ax.plot(ts, _rolling(aa, 10), color=ca, lw=2.2, label=f"{la}: 10-step rolling mean")
    ax.plot(ts, _rolling(bb, 10), color=cb, lw=2.2, label=f"{lb}: 10-step rolling mean")
    ax.set_title(f"B. Training accuracy ({_sampling_label(res)}, {prompt_label})")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Sampled accuracy (%)")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1, 0]
    d = [b - a for a, b in zip(aa, bb)]
    ax.bar(ts, d, color=[cb if x >= 0 else "#d62728" for x in d], alpha=0.5, width=0.8, label="per-step difference")
    ax.plot(ts, _rolling(d, 10), color="#111111", lw=2, label="10-step rolling mean")
    ax.axhline(0, color="#555555", lw=1)
    for w in res["train"].get("windows", []):
        lo, hi = w["steps"]
        if (lo, hi) == (1, ts[-1]):
            continue
        ax.hlines(w["diff_mean"], lo, hi, color="#2ca02c", lw=2.5)
        ax.annotate(f"{w['diff_mean']:+.2f} ({w['n_positive']}/{w['n']} > 0)", ((lo + hi) / 2, w["diff_mean"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color="#2ca02c")
    ax.set_title(f"C. Train-acc difference {lb} − {la}, {prompt_label} (window means; descriptive)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("pp")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1, 1]
    for g, lab, c in ((ga, la, ca), (gb, lb, cb)):
        xs = [s for s in ts if s in g and "response_length/mean" in g[s]]
        ax.plot(xs, _rolling([g[s]["response_length/mean"] for s in xs], 10), color=c, lw=2,
                label=f"{lab}: mean response length (10-step rolling)")
    ax2 = ax.twinx()
    for g, lab, c in ((ga, la, ca), (gb, lb, cb)):
        xs = [s for s in ts if s in g and "actor/entropy" in g[s]]
        ax2.plot(xs, _rolling([g[s]["actor/entropy"] for s in xs], 10), color=c, lw=1.5, ls="--",
                 label=f"{lab}: entropy (right)")
    ax2.grid(False)
    ax.set_title("D. Response length (tokens) and policy entropy")
    ax.set_xlabel("Training step")
    ax.set_ylabel("tokens")
    ax2.set_ylabel("entropy")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
