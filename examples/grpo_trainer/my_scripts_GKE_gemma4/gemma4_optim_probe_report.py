#!/usr/bin/env python3
"""Summarize a [GEMMA4-OPTIM-PROBE] run (see gemma4_optim_probe/__init__.py).

Reads <probe_dir>/init_rank*.json and <probe_dir>/rank*_call*.json, checks that every rank 0..W-1 reported for every
probed call, and prints / writes per-call totals over all ranks:

  * dtypes held by the optimizer (param, grad, exp_avg, exp_avg_sq, step) and the AdamW hyper-parameters;
  * changed / numel, changed / nonzero-grad, expected-to-survive / numel (round-to-nearest prediction) and how well
    the prediction matches the measured set;
  * the 2x2 optimizer-only counterfactual on the same gradients (probe v2): ideal (fp32 moments, fp32 write-back),
    fb (fp32 moments, param-dtype write-back), bf (optimizer moments, fp32 write-back), applied (measured), as norm
    ratio and cosine against ideal;
  * rollout visibility: elements whose bf16 cast changed in the step and since the first probed call (probe v2);
  * the gradient norm seen by the optimizer (post-clip) and the relative update ||applied|| / ||param||;
  * a per-tensor breakdown (FSDP flat-parameter index) and, for probe v2, a per-category breakdown over original
    parameters (norm, embedding, per-layer embedding, attention, MLP, other), summed over ranks.

Usage:
    python3 gemma4_optim_probe_report.py --probe-dir <dir> --world 64 [--label bf16] [--out-json s.json] [--out-md s.md]
Exit code 0 if coverage is complete and no probe errors were recorded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

SEG_FIELDS = ("numel", "changed", "expect", "small", "cast_changed", "cast_cum", "ideal_sq", "applied_sq", "dot",
              "param_sq")


def category(fqn: str) -> str:
    n = fqn.lower()
    if "norm" in n:
        return "norm"
    if "per_layer" in n:
        return "per-layer embedding / projection"
    if "embed" in n:
        return "embedding"
    if "attn" in n or "attention" in n:
        return "attention"
    if "mlp" in n:
        return "mlp"
    return "other"


def _load(probe_dir: str):
    inits, calls, errors = [], defaultdict(dict), []
    for f in sorted(glob.glob(os.path.join(probe_dir, "init_rank*.json"))):
        inits.append(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(probe_dir, "rank*_call*.json"))):
        m = re.search(r"rank(\d+)_call(\d+)\.json$", f)
        rec = json.load(open(f))
        calls[int(m.group(2))][int(m.group(1))] = rec
    for f in sorted(glob.glob(os.path.join(probe_dir, "error_*.json"))):
        errors.append({"file": os.path.basename(f), **json.load(open(f))})
    return inits, calls, errors


def _ratio(a, b):
    return a / b if b else float("nan")


def summarize(probe_dir: str, world: int) -> dict:
    inits, calls, errors = _load(probe_dir)
    out = {"probe_dir": probe_dir, "world": world, "errors": errors, "init": {}, "calls": {}}
    init_ranks = sorted({r["rank"] for r in inits})
    out["init"] = {
        "ranks": init_ranks,
        "complete": init_ranks == list(range(world)),
        "param_dtypes": sorted({d for r in inits for d in r["param_dtypes"]}),
        "optimizer_class": sorted({r["optimizer_class"] for r in inits}),
        "defaults": inits[0]["defaults"] if inits else None,
        "numel_total": int(sum(r["numel"] for r in inits)),
        "torch": sorted({r["torch"] for r in inits}),
        "probe_version": sorted({r.get("probe_version", "1") for r in inits}),
    }
    for k in sorted(calls):
        by_rank = calls[k]
        ranks = sorted(by_rank)
        tot = defaultdict(float)
        dtypes = defaultdict(set)
        per_tensor = defaultdict(lambda: defaultdict(float))
        per_fqn = defaultdict(lambda: defaultdict(float))
        names = {}
        step_t, lrs = set(), set()
        for r in ranks:
            rec = by_rank[r]
            for key, val in rec["totals"].items():
                tot[key] += val
            for key, vals in rec["dtypes"].items():
                dtypes[key].update(vals)
            step_t.update(rec["optimizer_step_t"])
            lrs.add(rec["hyper"]["lr"])
            for p in rec["params"]:
                key = (p["group"], p["index"])
                for f in ("numel", "grad_nonzero", "changed", "expect_survive", "ideal_sq", "applied_sq", "dot",
                          "param_sq"):
                    per_tensor[key][f] += p[f]
                if p.get("names") and key not in names:
                    names[key] = p["names"]
                for seg in p.get("segments", []):
                    d = per_fqn[seg[0]]
                    for f, val in zip(SEG_FIELDS, seg[1:]):
                        d[f] += val
        n_app, n_ide = math.sqrt(tot["applied_sq"]), math.sqrt(tot["ideal_sq"])
        tensors = []
        for key in sorted(per_tensor):
            t = per_tensor[key]
            a, i = math.sqrt(t["applied_sq"]), math.sqrt(t["ideal_sq"])
            tensors.append({
                "group": key[0], "index": key[1], "names": names.get(key), "numel": int(t["numel"]),
                "changed_frac": _ratio(t["changed"], t["numel"]),
                "expect_frac": _ratio(t["expect_survive"], t["numel"]),
                "applied_over_ideal": _ratio(a, i), "cos": _ratio(t["dot"], a * i),
            })
        cats = defaultdict(lambda: defaultdict(float))
        for fqn, d in per_fqn.items():
            c = cats[category(fqn)]
            c["n_params"] += 1
            for f in SEG_FIELDS:
                c[f] += d[f]
        categories = {}
        for name, c in sorted(cats.items()):
            a, i = math.sqrt(c["applied_sq"]), math.sqrt(c["ideal_sq"])
            categories[name] = {
                "n_params": int(c["n_params"]), "numel": int(c["numel"]),
                "changed_frac": _ratio(c["changed"], c["numel"]), "expect_frac": _ratio(c["expect"], c["numel"]),
                "small_frac": _ratio(c["small"], c["numel"]),
                "cast_changed_frac": _ratio(c["cast_changed"], c["numel"]),
                "cast_cum_frac": _ratio(c["cast_cum"], c["numel"]),
                "applied_over_ideal": _ratio(a, i), "cos": _ratio(c["dot"], a * i),
                "rel_update_applied": _ratio(a, math.sqrt(c["param_sq"])),
                "rel_update_ideal": _ratio(i, math.sqrt(c["param_sq"])),
            }
        v2 = "fb_sq" in tot
        n_fb, n_bf = math.sqrt(tot.get("fb_sq", 0.0)), math.sqrt(tot.get("bf_sq", 0.0))
        out["calls"][k] = {
            "ranks": ranks,
            "complete": ranks == list(range(world)),
            "optimizer_step_t": sorted(step_t),
            "lr": sorted(lrs),
            "hyper": by_rank[ranks[0]]["hyper"] if ranks else None,
            "dtypes": {key: sorted(val) for key, val in dtypes.items()},
            "numel": int(tot["numel"]),
            "grad_nonzero_frac": _ratio(tot["grad_nonzero"], tot["numel"]),
            "changed_frac": _ratio(tot["changed"], tot["numel"]),
            "changed_over_grad_nonzero": _ratio(tot["changed"], tot["grad_nonzero"]),
            "expect_survive_frac": _ratio(tot["expect_survive"], tot["numel"]),
            "changed_and_expected_over_changed": _ratio(tot["changed_and_expected"], tot["changed"]),
            "changed_and_expected_over_expected": _ratio(tot["changed_and_expected"], tot["expect_survive"]),
            "grad_norm_post_clip": math.sqrt(tot["grad_sq"]),
            "ideal_norm": n_ide,
            "applied_norm": n_app,
            "applied_over_ideal_norm": _ratio(n_app, n_ide),
            "cos_applied_ideal": _ratio(tot["dot"], n_app * n_ide),
            "mean_abs_ideal": _ratio(tot["ideal_abs_sum"], tot["numel"]),
            "mean_abs_applied": _ratio(tot["applied_abs_sum"], tot["numel"]),
            "relative_update": _ratio(n_app, math.sqrt(tot["param_sq"])),
            "relative_update_ideal": _ratio(n_ide, math.sqrt(tot["param_sq"])),
            "param_abs_lt_2pow-11_frac": _ratio(tot.get("param_abs_lt_2pow-11", float("nan")), tot["numel"]),
            "counterfactual_2x2": None if not v2 else {
                "fb_over_ideal_norm": _ratio(n_fb, n_ide),
                "fb_cos_ideal": _ratio(tot["fb_dot_ideal"], n_fb * n_ide),
                "fb_changed_frac": _ratio(tot["fb_changed"], tot["numel"]),
                "bf_over_ideal_norm": _ratio(n_bf, n_ide) if tot.get("bf_available") else float("nan"),
                "bf_cos_ideal": _ratio(tot["bf_dot_ideal"], n_bf * n_ide) if tot.get("bf_available") else float("nan"),
                "applied_over_fb_norm": _ratio(n_app, n_fb),
                "applied_cos_fb": _ratio(tot["applied_dot_fb"], n_app * n_fb),
            },
            "rollout_visible": None if not v2 else {
                "cast_changed_frac": _ratio(tot["cast_changed"], tot["numel"]),
                "cast_changed_cum_frac": _ratio(tot["cast_changed_cum"], tot["numel"]),
            },
            "categories": categories,
            "tensors": tensors,
        }
    return out


def render_md(s: dict, label: str) -> str:
    L = [f"### Optimizer write-back probe: `{label}`", ""]
    i = s["init"]
    L.append(f"- Probe dir: `{s['probe_dir']}` (probe v{','.join(i['probe_version'])}); init records from "
             f"{len(i['ranks'])}/{s['world']} ranks ({'complete' if i['complete'] else 'INCOMPLETE'}); optimizer "
             f"{i['optimizer_class']}, torch {i['torch']}.")
    L.append(f"- Optimizer-held parameter dtypes at construction: {i['param_dtypes']}; "
             f"total elements held over all ranks: {i['numel_total']:,}.")
    if i["defaults"]:
        d = i["defaults"]
        L.append(f"- AdamW defaults: lr={d.get('lr')}, betas={d.get('betas')}, eps={d.get('eps')}, "
                 f"weight_decay={d.get('weight_decay')}, foreach={d.get('foreach')}, fused={d.get('fused')}.")
    if s["errors"]:
        L.append(f"- **Probe errors: {len(s['errors'])}** (first: `{s['errors'][0]['file']}`).")
    L.append("")
    L.append("| Call | Ranks | AdamW step t | lr | param / grad / exp_avg / exp_avg_sq dtype | elements | nonzero grad | "
             "changed | changed / nonzero grad | predicted to survive | ‖applied‖ / ‖ideal‖ | cos(applied, ideal) | "
             "‖Δw‖ / ‖w‖ (applied) | ‖Δw‖ / ‖w‖ (ideal fp32) | post-clip grad norm |")
    L.append("|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, c in sorted(s["calls"].items(), key=lambda kv: int(kv[0])):
        dt = c["dtypes"]
        L.append(
            f"| {k} | {len(c['ranks'])}/{s['world']} | {c['optimizer_step_t']} | {c['lr']} | "
            f"{'/'.join(','.join(dt.get(x, [])) or '-' for x in ('param', 'grad', 'exp_avg', 'exp_avg_sq'))} | "
            f"{c['numel']:,} | {100 * c['grad_nonzero_frac']:.2f}% | **{100 * c['changed_frac']:.3f}%** | "
            f"{100 * c['changed_over_grad_nonzero']:.3f}% | {100 * c['expect_survive_frac']:.3f}% | "
            f"**{c['applied_over_ideal_norm']:.4f}** | {c['cos_applied_ideal']:.4f} | {c['relative_update']:.3e} | "
            f"{c['relative_update_ideal']:.3e} | {c['grad_norm_post_clip']:.4f} |"
        )
    L.append("")
    for k, c in sorted(s["calls"].items(), key=lambda kv: int(kv[0])):
        L.append(f"- Call {k}: measured changed set vs round-to-nearest prediction: "
                 f"{100 * c['changed_and_expected_over_changed']:.2f}% of changed elements were predicted; "
                 f"{100 * c['changed_and_expected_over_expected']:.2f}% of predicted elements changed. "
                 f"Mean |ideal| = {c['mean_abs_ideal']:.3e}, mean |applied| = {c['mean_abs_applied']:.3e}; "
                 f"|w| < 2^-11 for {100 * c['param_abs_lt_2pow-11_frac']:.3f}% of elements.")
    if any(c["counterfactual_2x2"] for c in s["calls"].values()):
        L.append("")
        L.append("Optimizer-only counterfactual on the same gradients (moments x write-back):")
        L.append("")
        L.append("| Call | ideal: fp32 m,v + fp32 write-back | fb: fp32 m,v + param-dtype write-back (‖·‖/‖ideal‖, cos, changed) "
                 "| bf: optimizer m,v + fp32 write-back (‖·‖/‖ideal‖, cos) | applied: optimizer m,v + param-dtype "
                 "write-back (‖·‖/‖ideal‖, cos, changed) | cos(applied, fb) | bf16-visible change: this step / cumulative |")
        L.append("|---:|---|---|---|---|---:|---|")
        for k, c in sorted(s["calls"].items(), key=lambda kv: int(kv[0])):
            x, rv = c["counterfactual_2x2"], c["rollout_visible"]
            if not x:
                continue
            L.append(f"| {k} | 1, 1, — | {x['fb_over_ideal_norm']:.4f}, {x['fb_cos_ideal']:.4f}, "
                     f"{100 * x['fb_changed_frac']:.3f}% | {x['bf_over_ideal_norm']:.4f}, {x['bf_cos_ideal']:.4f} | "
                     f"{c['applied_over_ideal_norm']:.4f}, {c['cos_applied_ideal']:.4f}, {100 * c['changed_frac']:.3f}% | "
                     f"{x['applied_cos_fb']:.4f} | {100 * rv['cast_changed_frac']:.3f}% / "
                     f"{100 * rv['cast_changed_cum_frac']:.3f}% |")
    if any(c["categories"] for c in s["calls"].values()):
        L.append("")
        L.append("Per original-parameter category (summed over ranks):")
        L.append("")
        L.append("| Call | Category | params | elements | changed | predicted | |w| < 2^-11 | ‖applied‖/‖ideal‖ | cos | "
                 "‖Δw‖/‖w‖ applied | ‖Δw‖/‖w‖ ideal | bf16-visible (step / cum) |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for k, c in sorted(s["calls"].items(), key=lambda kv: int(kv[0])):
            for name, g in c["categories"].items():
                L.append(f"| {k} | {name} | {g['n_params']} | {g['numel']:,} | {100 * g['changed_frac']:.3f}% | "
                         f"{100 * g['expect_frac']:.3f}% | {100 * g['small_frac']:.3f}% | {g['applied_over_ideal']:.4f} | "
                         f"{g['cos']:.4f} | {g['rel_update_applied']:.3e} | {g['rel_update_ideal']:.3e} | "
                         f"{100 * g['cast_changed_frac']:.3f}% / {100 * g['cast_cum_frac']:.3f}% |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()
    s = summarize(args.probe_dir, args.world)
    md = render_md(s, args.label)
    print(md)
    if args.out_json:
        json.dump(s, open(args.out_json, "w"), indent=1, sort_keys=True)
    if args.out_md:
        open(args.out_md, "w").write(md)
    ok = s["init"]["complete"] and s["calls"] and all(c["complete"] for c in s["calls"].values()) and not s["errors"]
    print(f"GEMMA4-OPTIM-PROBE-REPORT: {'COMPLETE' if ok else 'INCOMPLETE'} label={args.label} "
          f"calls={sorted(s['calls'])} errors={len(s['errors'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
