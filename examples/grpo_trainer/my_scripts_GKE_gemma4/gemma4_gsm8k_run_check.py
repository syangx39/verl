#!/usr/bin/env python3
"""Run check for the Gemma-4 E2B GSM8K 8K baseline (PR #14): smoke or seed run, from its log, dumps and checkpoints.

    python3 gemma4_gsm8k_run_check.py EXP [--seed 1] [--scorer <boxed_math_reward_v2.py>] [--steps 1,2,3]

Reads /workspace/meta-RL/logs/<EXP>.log, /workspace/meta-RL/logs/gemma4_gsm8k/<EXP>/{val,train}/<step>.jsonl and
/workspace/meta-RL/ckpt/<EXP>/ (override the roots with --logs / --ckpt-root). Prints:
  provenance  the launcher's [gsm8k-provenance] line, the smoke stop note, the driver exit lines
  ckpt        the [GEMMA4-CKPT-STEPS] decisions and the global_step_* directories that exist (files, GB),
              latest_checkpointed_iteration.txt
  eval        per eval dump: n, unique qids, acc (the training scorer), acc_v1 and v2_only when the scorer is v2,
              boxed rate, scorer_version values
  train       per rollout dump: rows, prompt groups and sizes, acc / acc_v1 / v2_only, mixed groups (non-zero GRPO
              advantage), same question IDs as the handoff train_order_seed<seed>.parquet; with --scorer the stored
              score of every row must equal the v2 reward recomputed offline
  metrics     the per-step console metrics: LR, grad norm, losses, entropy (actor/entropy_loss: the token entropy
              of the update batch; no bonus is applied), every actor/rollout_corr/* tag (token TIS), rewards,
              lengths, timing and memory, and the coordinator step clock wallclock/interval_s when present
  jsonl       <dump>/metrics.jsonl from verl's file logger: one record per logged step without duplicates or gaps,
              the required keys in every training record (core metrics, every actor/rollout_corr/* tag,
              wallclock/coordinator_unix), and every value equal to the console line of the same step
"""
import argparse
import glob
import importlib.util
import json
import os
import re

STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
WRAPPED_RE = re.compile(r"^[A-Za-z_][\w.]*\(\s*([^,()]*?)\s*(?:,.*)?\)$")   # np.float64(x) / tensor(x, ...)
CORE_KEYS = ["actor/lr", "actor/grad_norm", "actor/pg_loss", "actor/ppo_kl", "actor/entropy_loss",
             "critic/score/mean", "critic/rewards/mean", "critic/advantages/max", "critic/advantages/min",
             "response_length/mean", "response_length/max", "response_length/clip_ratio",
             "timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/update_weights",
             "timing_s/save_checkpoint", "timing_s/testing", "actor/perf/max_memory_allocated_gb",
             "actor/perf/max_memory_reserved_gb", "perf/total_num_tokens", "wallclock/interval_s"]
NOT_EVERY_STEP = {"timing_s/save_checkpoint", "timing_s/testing", "wallclock/interval_s"}


def parse_metric_line(text):
    kv = {}
    for part in text.split(" - "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        v = v.strip()
        m = WRAPPED_RE.match(v)
        if m:
            v = m.group(1)
        try:
            kv[k.strip()] = float(v)
        except ValueError:
            pass
    return kv


def jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def by_step(d):
    out = []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        b = os.path.basename(f)[:-6]
        if b.isdigit():
            out.append((int(b), f))
    return sorted(out)


def fnum(rows, k):
    return [float(r[k]) for r in rows if k in r]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scorer", help="boxed_math_reward_v2.py: recompute every training row's reward and compare")
    ap.add_argument("--steps", help="comma list of metric steps to print (default: all)")
    ap.add_argument("--logs", default="/workspace/meta-RL/logs")
    ap.add_argument("--ckpt-root", default="/workspace/meta-RL/ckpt")
    ap.add_argument("--handoff-data", default="/workspace/meta-RL/handoff/gsm8k_8k/data")
    a = ap.parse_args()
    log_path = os.path.join(a.logs, f"{a.exp}.log")
    dump = os.path.join(a.logs, "gemma4_gsm8k", a.exp)
    with open(log_path, errors="replace") as f:
        log = f.read().split("\n")
    print("EXP", a.exp)
    print("provenance:", next((ln.split("[gsm8k-provenance]", 1)[1].strip() for ln in log if "[gsm8k-provenance]" in ln
                               and "echo" not in ln), None))
    stop = os.path.join(dump, "smoke_stop.txt")
    print("stop note:", open(stop).read().strip() if os.path.exists(stop) else None)
    print("driver exit lines:", [ln.strip() for ln in log if "LAUNCHER_RC" in ln or "main_ppo finished" in ln][-2:])

    # checkpoints
    ck = [ln[ln.index("[GEMMA4-CKPT-STEPS]"):].strip() for ln in log if "[GEMMA4-CKPT-STEPS]" in ln]
    print(f"ckpt hook lines ({len(ck)}):")
    for ln in ck:
        print("  ", re.sub(r" \d{4}-\d\d-\d\dT\S+ pid=\d+", "", ln))
    root = os.path.join(a.ckpt_root, a.exp)
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                files = [os.path.join(dp, fn) for dp, _, fns in os.walk(p) for fn in fns]
                gb = sum(os.path.getsize(x) for x in files) / 1e9
                print(f"  ckpt dir {name}: {len(files)} files, {gb:.1f} GB, actor/ entries "
                      f"{len(os.listdir(os.path.join(p, 'actor'))) if os.path.isdir(os.path.join(p, 'actor')) else 0},"
                      f" data.pt {'yes' if os.path.exists(os.path.join(p, 'data.pt')) else 'no'}")
            else:
                print(f"  ckpt file {name}: {open(p).read().strip()[:80]!r}")
    else:
        print("  no checkpoint directory", root)

    # eval dumps
    for s, f in by_step(os.path.join(dump, "val")):
        v = jsonl(f)
        n = len(v)
        acc, a1, only = fnum(v, "acc"), fnum(v, "acc_v1"), fnum(v, "v2_only")
        line = (f"eval step {s}: n={n} unique_qid={len({r['qid'] for r in v})} acc={100 * sum(acc) / n:.2f}% "
                f"boxed={100 * sum(fnum(v, 'fmt')) / n:.2f}%")
        if a1:
            line += (f" acc_v1={100 * sum(a1) / n:.2f}% v2_only={int(sum(only))}"
                     f" scorer_version={sorted({r.get('scorer_version') for r in v})}")
        print(line)

    # training dumps
    order = {}
    op = os.path.join(a.handoff_data, f"train_order_seed{a.seed}.parquet")
    if os.path.exists(op):
        import pyarrow.parquet as pq
        for r in pq.read_table(op).to_pylist():
            order.setdefault(r["step"], set()).add(float(r["qid"]))
    v2 = None
    if a.scorer:
        spec = importlib.util.spec_from_file_location("boxed_math_reward_v2", a.scorer)
        v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2)
    for s, f in by_step(os.path.join(dump, "train")):
        rows = jsonl(f)
        groups = {}
        for r in rows:
            groups.setdefault(float(r["qid"]), []).append(r)
        n = len(rows)
        mixed = sum(1 for g in groups.values() if len({float(r["acc"]) for r in g}) > 1)
        line = (f"train step {s}: rows={n} groups={len(groups)} sizes={sorted({len(g) for g in groups.values()})} "
                f"acc={100 * sum(fnum(rows, 'acc')) / n:.2f}% boxed={100 * sum(fnum(rows, 'fmt')) / n:.2f}% "
                f"mixed_groups={mixed}/{len(groups)} length_penalty_sum={sum(abs(x) for x in fnum(rows, 'length_penalty'))}")
        if "acc_v1" in rows[0]:
            mixed1 = sum(1 for g in groups.values() if len({float(r["acc_v1"]) for r in g}) > 1)
            line += (f" | acc_v1={100 * sum(fnum(rows, 'acc_v1')) / n:.2f}% mixed_groups_v1={mixed1} "
                     f"v2_only_rows={int(sum(fnum(rows, 'v2_only')))} "
                     f"scorer_version={sorted({r.get('scorer_version') for r in rows})}")
        if order:
            line += f" same_qids_as_handoff_seed{a.seed}={set(groups) == order.get(s)}"
        if v2 is not None:
            bad = sum(1 for r in rows if abs(v2.boxed_math_v2(r["output"], str(r["gts"]))[0] - float(r["score"])) > 1e-6)
            line += f" score==v2_recomputed: {n - bad}/{n}"
        print(line)

    # console metrics
    steps = {}
    for ln in log:
        m = STEP_RE.search(ln)
        if m:
            steps.setdefault(int(m.group(1)), {}).update(parse_metric_line(m.group(2)))
    show = sorted(steps) if not a.steps else [int(x) for x in a.steps.split(",") if int(x) in steps]
    rc_keys = sorted({k for kv in steps.values() for k in kv if k.startswith("actor/rollout_corr/")})
    val_keys = sorted({k for kv in steps.values() for k in kv if k.startswith(("val-core/", "val-aux/gsm8k"))
                       and k.endswith("/mean@1")})
    print("metric steps:", sorted(steps))
    for k in CORE_KEYS + rc_keys + val_keys:
        cells = "  ".join(f"s{s}={steps[s][k]:.5g}" if k in steps.get(s, {}) else f"s{s}=-" for s in show)
        print(f"  {k:52s} {cells}")
    print("actor/rollout_corr tags:", len(rc_keys), "| missing core keys:",
          [k for k in CORE_KEYS if not any(k in steps[s] for s in steps)])

    # metrics JSONL from verl's file logger: one record per logged step, complete, equal to the console values
    mj = os.path.join(dump, "metrics.jsonl")
    if not os.path.exists(mj):
        print("metrics.jsonl: MISSING", mj)
    else:
        recs = jsonl(mj)
        st = [int(r["step"]) for r in recs]
        seen, dup = set(), set()
        for s in st:
            (dup if s in seen else seen).add(s)
        gaps = [s for s in range(min(seen), max(seen) + 1) if s not in seen] if seen else []
        need = [k for k in CORE_KEYS if k not in NOT_EVERY_STEP] + rc_keys + ["wallclock/coordinator_unix"]
        data = {}
        for r in recs:
            data.setdefault(int(r["step"]), {}).update(r["data"])
        incomplete = {s: [k for k in need if k not in d] for s, d in data.items() if s > 0}
        incomplete = {s: v for s, v in incomplete.items() if v}
        no_console = sorted(s for s in data if s not in steps)
        worst, n_cmp = (0.0, None), 0
        for s, d in data.items():
            for k, v in d.items():
                c = steps.get(s, {}).get(k)
                if isinstance(v, (int, float)) and c is not None:
                    n_cmp += 1
                    rel = abs(v - c) / max(abs(c), 1e-12)
                    if rel > worst[0]:
                        worst = (rel, f"step {s} {k}: jsonl {v!r} console {c!r}")
        iv = [d["wallclock/interval_s"] for s, d in data.items() if s > 0 and "wallclock/interval_s" in d]
        print(f"metrics.jsonl: {len(recs)} records, steps {min(seen) if seen else '-'}..{max(seen) if seen else '-'}, "
              f"duplicates {sorted(dup)}, gaps {gaps[:10]}, records with missing required keys {len(incomplete)} "
              f"{dict(list(incomplete.items())[:3])}, steps absent from the console {no_console[:10]}, "
              f"values compared with the console {n_cmp}, worst relative difference {worst[0]:.2e} ({worst[1]}), "
              f"records with wallclock/interval_s {len(iv)}")


if __name__ == "__main__":
    main()
