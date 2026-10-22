#!/usr/bin/env python3
"""Offline learning-signal, eval-pairing and grading audit of a Gemma-4 GRPO run from its rollout / validation dumps.

Inputs are the JSONL dumps written by trainer.rollout_data_dir (train/<step>.jsonl, 2,048 rows = 256 prompts x 8)
and trainer.validation_data_dir (val/<step>.jsonl, one row per eval question). Nothing is re-generated.

1. Training groups (rows sharing `qid` within a step, expected 256 groups of 8 per step). Validates group size and
   that all rows of a group share the same input and ground truth. Classifies each group:
   - same_all_correct / same_all_wrong : all 8 scores identical -> every advantage is exactly 0 (no gradient);
   - fmt_only_correct / fmt_only_wrong : correctness identical across the 8, scores differ only by the 0.1 format bonus;
   - acc_mixed                         : correctness differs inside the group (the only correctness signal).
   Advantages are recomputed as verl's GRPO (norm_adv_by_std_in_grpo=True): A = (s - mean) / (std + 1e-6) with the
   unbiased std. Because of the std normalization a format-only group gets advantages as large as a correctness
   group, so the report gives each class's share of sum|A| and of the output-length-weighted sum|A| (proxy for the
   token-mean loss). Also reports within-group std of score and of accuracy.
2. Eval pairing by question identity. Every val file must have unique `qid`s and, per `qid`, the same input and
   ground truth (hashes) as the reference step-0 file; only then are checkpoints paired by `qid` (not by row order):
   gained / lost questions, paired difference, paired SE and a paired bootstrap 95% interval. With --replicate-run
   (a second run evaluated on the same eval set), the step-0 evals of the two runs score identical weights, so their
   difference measures the run-to-run nondeterminism of greedy evaluation (noise floor).
3. Grading audit of wrong answers (acc = 0) and of questions correct at step 0 but wrong at the last checkpoint:
   answer extraction / normalization with the reward function's own `extract_answer` / `preprocess_math_string`,
   flags for likely false negatives, extraction fallbacks, math-verify length rejects; re-scoring a sample with
   `compute_score` (in-process math-verify) to confirm the dumped `acc` reproduces.

Usage:
    python3 gemma4_learning_signal.py --run-dir <eval_dump_dir> [--replicate-run <other_eval_dump_dir>]
        [--reward-fn maxtext_math_reward.py] [--audit-steps 1,100,200] [--audit-val-steps 0,200] [--audit-n 400]
        [--rescore-n 100] [--examples 10] [--bootstrap 2000] [--out-json s.json] [--out-md s.md]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

CLASSES = ("same_all_correct", "same_all_wrong", "fmt_only_correct", "fmt_only_wrong", "acc_mixed")


def _rows(path):
    with open(path, errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _f(x) -> float:
    return float(x)


def _h(s) -> str:
    return hashlib.sha1(str(s).encode("utf-8", "replace")).hexdigest()[:16]


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _grpo_adv(scores):
    n = len(scores)
    if n < 2:
        return [0.0] * n
    m = sum(scores) / n
    sd = math.sqrt(sum((s - m) ** 2 for s in scores) / (n - 1))
    return [(s - m) / (sd + 1e-6) for s in scores]


def classify(accs, scores) -> str:
    if len(set(scores)) == 1:
        return "same_all_correct" if accs[0] >= 0.5 else "same_all_wrong"
    if len(set(accs)) == 1:
        return "fmt_only_correct" if accs[0] >= 0.5 else "fmt_only_wrong"
    return "acc_mixed"


def analyze_step(path):
    groups = defaultdict(list)
    for r in _rows(path):
        groups[str(r.get("qid", r.get("input", "")[:200]))].append(r)
    sizes = Counter(len(g) for g in groups.values())
    cnt, mass, mass_len = Counter(), Counter(), Counter()
    n_rows, acc_sum, inconsistent = 0, 0.0, 0
    sd_score, sd_acc = [], []
    for g in groups.values():
        if len({r.get("input") for r in g}) != 1 or len({str(r.get("gts")) for r in g}) != 1:
            inconsistent += 1
        accs = [_f(r["acc"]) for r in g]
        scores = [_f(r["score"]) for r in g]
        lens = [len(r.get("output", "")) for r in g]
        c = classify(accs, scores)
        adv = _grpo_adv(scores)
        cnt[c] += 1
        mass[c] += sum(abs(a) for a in adv)
        mass_len[c] += sum(abs(a) * L for a, L in zip(adv, lens))
        n_rows += len(g)
        acc_sum += sum(accs)
        sd_score.append(_std(scores))
        sd_acc.append(_std(accs))
    return {"n_groups": len(groups), "n_rows": n_rows, "group_sizes": dict(sizes), "inconsistent_groups": inconsistent,
            "counts": dict(cnt), "adv_mass": dict(mass), "adv_mass_len": dict(mass_len), "acc": acc_sum / max(n_rows, 1),
            "mean_group_std_score": statistics.mean(sd_score), "mean_group_std_acc": statistics.mean(sd_acc)}


def _window(per_step, lo, hi):
    sel = [v for s, v in per_step.items() if lo <= s <= hi]
    if not sel:
        return None
    tot_g = sum(v["n_groups"] for v in sel)
    cnt, mass, mass_len = Counter(), Counter(), Counter()
    for v in sel:
        cnt.update(v["counts"])
        mass.update(v["adv_mass"])
        mass_len.update(v["adv_mass_len"])
    tm, tml = sum(mass.values()), sum(mass_len.values())
    return {
        "steps": [lo, hi], "n_steps": len(sel), "n_groups": tot_g,
        "train_acc_mean": statistics.mean(v["acc"] for v in sel),
        "mean_group_std_score": statistics.mean(v["mean_group_std_score"] for v in sel),
        "mean_group_std_acc": statistics.mean(v["mean_group_std_acc"] for v in sel),
        "group_frac": {c: cnt[c] / tot_g for c in CLASSES},
        "adv_mass_share": {c: (mass[c] / tm if tm else 0.0) for c in CLASSES},
        "adv_mass_len_share": {c: (mass_len[c] / tml if tml else 0.0) for c in CLASSES},
    }


# ----------------------------------------------------------------------------- eval pairing
def _load_val(path):
    rows = list(_rows(path))
    qids = [str(r["qid"]) for r in rows]
    if len(set(qids)) != len(qids):
        raise ValueError(f"{path}: duplicate qids ({len(qids) - len(set(qids))})")
    return {str(r["qid"]): {"acc": _f(r["acc"]), "fmt": _f(r["fmt"]), "in": _h(r.get("input")), "gt": _h(r.get("gts")),
                            "out": r.get("output", ""), "gts": r.get("gts")} for r in rows}


def _check_identity(ref, other, label):
    if set(ref) != set(other):
        raise ValueError(f"{label}: qid sets differ ({len(set(ref) ^ set(other))} not shared)")
    bad = [q for q in ref if (ref[q]["in"], ref[q]["gt"]) != (other[q]["in"], other[q]["gt"])]
    if bad:
        raise ValueError(f"{label}: {len(bad)} qids with different input/ground truth, e.g. {bad[:3]}")


def _paired(a, b, qids, B, seed):
    """Paired stats of b - a over qids (accuracy in %)."""
    import numpy as np

    d = np.array([b[q]["acc"] - a[q]["acc"] for q in qids], dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(0, B, 200):
        idx = rng.integers(0, n, size=(min(200, B - i), n))
        means[i:i + idx.shape[0]] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": n, "acc_a": 100 * float(np.mean([a[q]["acc"] for q in qids])),
            "acc_b": 100 * float(np.mean([b[q]["acc"] for q in qids])),
            "delta_pp": 100 * float(d.mean()), "paired_se_pp": 100 * float(d.std(ddof=1) / math.sqrt(n)),
            "boot95_pp": [100 * float(lo), 100 * float(hi)], "gained": int((d > 0).sum()), "lost": int((d < 0).sum()),
            "identical_outputs": int(sum(a[q]["out"] == b[q]["out"] for q in qids))}


def eval_pairing(run_dir, replicate_dir, B, seed):
    val_dir = os.path.join(run_dir, "val")
    steps = sorted(int(f[:-6]) for f in os.listdir(val_dir) if f.endswith(".jsonl") and f[:-6].isdigit())
    vals = {s: _load_val(os.path.join(val_dir, f"{s}.jsonl")) for s in steps}
    ref = vals[steps[0]]
    for s in steps[1:]:
        _check_identity(ref, vals[s], f"step {s} vs step {steps[0]}")
    qids = sorted(ref)
    out = {"steps": steps, "n_questions": len(qids), "identity_checked": True,
           "vs_step0": {s: _paired(ref, vals[s], qids, B, seed + s) for s in steps[1:]}}
    per_q = [sum(vals[s][q]["acc"] for s in steps) for q in qids]
    k = len(steps)
    out["stability"] = {"never": sum(1 for x in per_q if x == 0) / len(qids),
                        "always": sum(1 for x in per_q if x == k) / len(qids),
                        "sometimes": sum(1 for x in per_q if 0 < x < k) / len(qids)}
    if replicate_dir:
        rdir = os.path.join(replicate_dir, "val")
        rsteps = sorted(int(f[:-6]) for f in os.listdir(rdir) if f.endswith(".jsonl") and f[:-6].isdigit())
        rep = {s: _load_val(os.path.join(rdir, f"{s}.jsonl")) for s in rsteps if s in vals}
        reps = {}
        for s, v in rep.items():
            _check_identity(ref, v, f"replicate step {s}")
            reps[s] = _paired(vals[s], v, qids, B, seed + 1000 + s)
        out["replicate"] = {"dir": replicate_dir, "same_step": reps}
    return out, vals


# ----------------------------------------------------------------------------- grading audit
_NUM_CLEAN = re.compile(r"\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\[,;!]|\\%|%|\$|\\\$|\s|,")


def _as_number(s: str):
    t = _NUM_CLEAN.sub("", s)
    t = t.replace("\\left", "").replace("\\right", "").strip("{}()")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)) / float(m.group(2)) if float(m.group(2)) else None
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}", t)
    if m:
        return float(m.group(1)) / float(m.group(2)) if float(m.group(2)) else None
    try:
        return float(t)
    except ValueError:
        return None


def _flag(out, gts_raw, R):
    gts = json.loads(gts_raw) if isinstance(gts_raw, str) else gts_raw
    gts = list(dict.fromkeys(gts if isinstance(gts, list) else [str(gts)]))
    guess = R.extract_answer(out)
    if guess == R.FALLBACK_ANSWER:
        return "no_answer_extracted", guess, gts[0], ""
    ng = R.preprocess_math_string(guess)
    ngts = [R.preprocess_math_string(g) for g in gts]
    a = _as_number(ng)
    b = [_as_number(x) for x in ngts]
    if len(guess) > R._MV_MAX_CHARS:
        return "mv_length_reject", guess, gts[0], ng
    if a is not None and any(x is not None and abs(a - x) <= 1e-6 * max(1.0, abs(x)) for x in b):
        return "numeric_equal_after_norm", guess, gts[0], ng
    if any(x and x in ng for x in ngts) and ng not in ngts:
        return "gold_contained_in_guess", guess, gts[0], ng
    if a is not None and all(x is not None for x in b):
        return "both_numeric_different", guess, gts[0], ng
    return "other_symbolic_or_text", guess, gts[0], ng


def audit(rows, R, n, rng, n_rescore):
    """rows: list of (output, gts_raw) for wrong (acc=0) answers."""
    sample = rows if len(rows) <= n else rng.sample(rows, n)
    flags, examples = Counter(), defaultdict(list)
    rescored = now_correct = 0
    for i, (out, gts_raw) in enumerate(sample):
        cat, guess, gt, ng = _flag(out, gts_raw, R)
        flags[cat] += 1
        examples[cat].append((guess if cat != "no_answer_extracted" else out[-300:], gt, ng))
        if rescored < n_rescore:
            rescored += 1
            res = R.compute_score("audit", out, gts_raw if isinstance(gts_raw, str) else json.dumps(gts_raw),
                                  extra_info={"index": i})
            now_correct += float(res["acc"]) >= 0.5
    return {"n_wrong_total": len(rows), "n_audited": len(sample), "flags": dict(flags), "rescored": rescored,
            "rescore_now_correct": int(now_correct), "examples": {k: v[:50] for k, v in examples.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="eval dump dir with train/ and val/")
    ap.add_argument("--replicate-run", default="", help="second run on the same eval set (identical step-0 weights)")
    ap.add_argument("--reward-fn", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "maxtext_math_reward.py"))
    ap.add_argument("--audit-steps", default="1,100,200")
    ap.add_argument("--audit-val-steps", default="0,200")
    ap.add_argument("--audit-n", type=int, default=400)
    ap.add_argument("--rescore-n", type=int, default=100)
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    os.environ.setdefault("REWARD_MV_POOL", "0")  # in-process math-verify: no forked pool for offline use
    spec = importlib.util.spec_from_file_location("reward_fn", args.reward_fn)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    rng = random.Random(args.seed)

    train_dir = os.path.join(args.run_dir, "train")
    steps = sorted(int(f[:-6]) for f in os.listdir(train_dir) if f.endswith(".jsonl") and f[:-6].isdigit())
    per_step = {s: analyze_step(os.path.join(train_dir, f"{s}.jsonl")) for s in steps}
    last = steps[-1]
    windows = {name: _window(per_step, lo, hi) for name, lo, hi in
               (("early", 1, 20), ("mid", max(1, last // 2 - 9), last // 2 + 10), ("late", last - 19, last),
                ("all", 1, last))}

    pairing, vals = eval_pairing(args.run_dir, args.replicate_run, args.bootstrap, args.seed)

    audits = {}
    for s in [int(x) for x in args.audit_steps.split(",") if x]:
        p = os.path.join(train_dir, f"{s}.jsonl")
        if os.path.exists(p):
            wrong = [(r.get("output", ""), r["gts"]) for r in _rows(p) if _f(r["acc"]) < 0.5]
            audits[f"train_step{s}_wrong"] = audit(wrong, R, args.audit_n, rng, args.rescore_n)
    for s in [int(x) for x in args.audit_val_steps.split(",") if x]:
        if s in vals:
            wrong = [(v["out"], v["gts"]) for v in vals[s].values() if v["acc"] < 0.5]
            audits[f"val_step{s}_wrong"] = audit(wrong, R, args.audit_n, rng, args.rescore_n)
    s0, sT = pairing["steps"][0], pairing["steps"][-1]
    lost = [(vals[sT][q]["out"], vals[sT][q]["gts"]) for q in vals[s0] if vals[s0][q]["acc"] >= 0.5 > vals[sT][q]["acc"]]
    audits[f"val_correct_step{s0}_to_wrong_step{sT}"] = audit(lost, R, args.audit_n, rng, args.rescore_n)

    summary = {"run_dir": args.run_dir, "n_train_steps": len(steps), "per_step": per_step, "windows": windows,
               "eval_pairing": pairing, "audits": audits}

    L = [f"### Learning-signal / eval-pairing / grading audit: `{args.run_dir}`", ""]
    sizes = Counter()
    n_groups_per_step = Counter()
    inconsistent = 0
    for v in per_step.values():
        sizes.update({int(k): c for k, c in v["group_sizes"].items()})
        n_groups_per_step[v["n_groups"]] += 1
        inconsistent += v["inconsistent_groups"]
    L.append(f"- Training dumps: {len(steps)} steps; groups per step {dict(n_groups_per_step)}; group sizes "
             f"{dict(sorted(sizes.items()))}; groups with mixed input or ground truth: {inconsistent}.")
    L.append("")
    L.append("| Window | Steps | Groups | Train acc (T=0.8) | mean group std (score / acc) | all-same correct | all-same wrong | "
             "format-only (correct) | format-only (wrong) | correctness-mixed | sum|A| share: format-only | "
             "sum|A| share: correctness-mixed | length-weighted share: format-only |")
    L.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, w in windows.items():
        if not w:
            continue
        gf, ms, ml = w["group_frac"], w["adv_mass_share"], w["adv_mass_len_share"]
        L.append(f"| {name} | {w['steps'][0]}–{w['steps'][1]} | {w['n_groups']} | {100 * w['train_acc_mean']:.2f}% | "
                 f"{w['mean_group_std_score']:.3f} / {w['mean_group_std_acc']:.3f} | "
                 f"{100 * gf['same_all_correct']:.1f}% | {100 * gf['same_all_wrong']:.1f}% | "
                 f"{100 * gf['fmt_only_correct']:.1f}% | {100 * gf['fmt_only_wrong']:.1f}% | "
                 f"**{100 * gf['acc_mixed']:.1f}%** | {100 * (ms['fmt_only_correct'] + ms['fmt_only_wrong']):.1f}% | "
                 f"{100 * ms['acc_mixed']:.1f}% | {100 * (ml['fmt_only_correct'] + ml['fmt_only_wrong']):.1f}% |")
    L.append("")
    st = pairing["stability"]
    L.append(f"- Eval identity: {pairing['n_questions']} unique qids per file; qid set, input and ground-truth hashes "
             f"identical across checkpoints {pairing['steps']}" + (" and the replicate run" if args.replicate_run else "")
             + "; all pairing below is by qid.")
    L.append(f"- Across the {len(pairing['steps'])} checkpoints: never solved {100 * st['never']:.1f}%, always solved "
             f"{100 * st['always']:.1f}%, sometimes {100 * st['sometimes']:.1f}%.")
    L.append("")
    L.append("| Comparison | Acc A → B (%) | Δ (pp) | paired SE (pp) | bootstrap 95% (pp) | +gained / −lost | identical outputs |")
    L.append("|---|---|---:|---:|---|---|---:|")
    for name, p in [(f"step {s0} → {s} (same run)", v) for s, v in pairing["vs_step0"].items()] + \
                   [(f"run vs replicate, step {s} (identical weights at step 0)", v)
                    for s, v in pairing.get("replicate", {}).get("same_step", {}).items()]:
        L.append(f"| {name} | {p['acc_a']:.2f} → {p['acc_b']:.2f} | {p['delta_pp']:+.2f} | {p['paired_se_pp']:.2f} | "
                 f"[{p['boot95_pp'][0]:+.2f}, {p['boot95_pp'][1]:+.2f}] | +{p['gained']} / −{p['lost']} | "
                 f"{p['identical_outputs']} / {p['n']} |")
    L.append("")
    L.append("| Audit source | wrong rows | audited | no answer extracted | numerically equal after norm | gold contained in "
             "guess | both numeric, different | other (symbolic/text) | math-verify length reject | re-scored now correct |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, a in audits.items():
        fl = a["flags"]
        L.append(f"| {name} | {a['n_wrong_total']} | {a['n_audited']} | {fl.get('no_answer_extracted', 0)} | "
                 f"{fl.get('numeric_equal_after_norm', 0)} | {fl.get('gold_contained_in_guess', 0)} | "
                 f"{fl.get('both_numeric_different', 0)} | {fl.get('other_symbolic_or_text', 0)} | "
                 f"{fl.get('mv_length_reject', 0)} | {a['rescore_now_correct']}/{a['rescored']} |")
    L.append("")
    for name, a in audits.items():
        for cat in ("numeric_equal_after_norm", "gold_contained_in_guess", "other_symbolic_or_text",
                    "no_answer_extracted", "mv_length_reject"):
            ex = a["examples"].get(cat, [])[: args.examples]
            if not ex:
                continue
            L.append(f"<details><summary>{name}: {cat} (first {len(ex)})</summary>")
            L.append("")
            L.append("| guess (extracted) | gold | normalized guess |")
            L.append("|---|---|---|")
            for g, gt, ng in ex:
                cell = lambda x: str(x).replace("|", "\\|").replace("\n", " ")[:160]  # noqa: E731
                L.append(f"| `{cell(g)}` | `{cell(gt)}` | `{cell(ng)}` |")
            L.append("")
            L.append("</details>")
            L.append("")
    md = "\n".join(L) + "\n"
    print(md)
    if args.out_json:
        json.dump(summary, open(args.out_json, "w"), indent=1, sort_keys=True, default=str)
    if args.out_md:
        open(args.out_md, "w").write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
