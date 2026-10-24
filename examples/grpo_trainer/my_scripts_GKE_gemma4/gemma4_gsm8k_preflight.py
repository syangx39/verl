#!/usr/bin/env python3
"""Preflight for the Gemma-4 E2B GSM8K 8K baseline (PR #14, GSM8K_8K_PLAN.md). No GPU; aborts on any failure.

Checks, all written to --out (JSON):
  data      train 7,473 / test 1,319 rows; unique extra_info.index (train 0..7472, test >= 10,000,000); data_source
            names; prompt = [system, user]; non-empty gold answers; zero train/test question overlap; sha256 of the
            parquets against the handoff PACKAGE_MANIFEST.sha256 when present
  template  every prompt rendered ONCE by the Gemma chat template exactly as verl does (apply_chat_template(...,
            add_generation_prompt=True, tokenize=False), then tokenized with add_special_tokens=False): starts with the
            system turn, ends with the model turn, carries the boxed instruction once, and fits --max-prompt (no
            truncation or filtering is allowed in the launcher); a fixture (shortest, first, longest examples with
            token IDs) is saved next to --out
  scorer    boxed_math_reward.py imported in fresh subprocesses: (1) its own self-test under Meta's fixture settings
            (cap 2048, penalty on the train source) must PASS; (2) under the run settings (cap --cap, REWARD_PENALTY_SOURCES
            empty): correct box 1.0 (not 1.1), parseable wrong 0.1, unparseable 0, length penalty 0 even without
            extra_info["response_len"]
  model     generation_config eos ids and tokenizer special tokens recorded (Gemma-native stopping; no Qwen ids)
  schedule  update-used and logged LRs for linear warmup --warmup then cosine to 0 at --total-steps, from a copy of
            verl's get_cosine_schedule_with_warmup lambda (zero-indexed): update 1 uses 0, update 2 uses peak/warmup,
            update warmup+1 uses peak, the post-update-N value is 0

Usage:
    python3 gemma4_gsm8k_preflight.py --data-dir D --model M --reward R --out preflight.json \\
        [--max-prompt 512 --cap 8192 --total-steps 250 --warmup 10 --peak-lr 2e-6]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

BOXED_INSTRUCTION = "put your final answer within \\boxed{}"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_q(s: str) -> str:
    return " ".join(str(s).split()).lower()


def check_data(a, out, fails):
    import pyarrow.parquet as pq

    res = {}
    rows = {}
    for split, fname, n_want, base in (("train", "gsm8k_boxed_train.parquet", 7473, 0),
                                       ("test", "gsm8k_boxed_test.parquet", 1319, 10_000_000)):
        path = os.path.join(a.data_dir, fname)
        t = pq.read_table(path).to_pylist()
        rows[split] = t
        idx = [r["extra_info"]["index"] for r in t]
        srcs = sorted({r["data_source"] for r in t})
        roles = sorted({tuple(m["role"] for m in r["prompt"]) for r in t})
        empty_gt = sum(1 for r in t if not str(r["reward_model"]["ground_truth"]).strip())
        r = {"path": path, "sha256": sha256(path), "rows": len(t), "unique_index": len(set(idx)) == len(idx),
             "index_min": min(idx), "index_max": max(idx), "data_sources": srcs, "roles": [list(x) for x in roles],
             "empty_ground_truth": empty_gt}
        ok = (len(t) == n_want and r["unique_index"] and min(idx) == base and max(idx) == base + n_want - 1
              and srcs == [f"gsm8k_boxed_{split}"] and roles == [("system", "user")] and empty_gt == 0)
        r["ok"] = ok
        if not ok:
            fails.append(f"data {split}: {r}")
        res[split] = r
    overlap = {norm_q(r["extra_info"]["question"]) for r in rows["train"]} & \
              {norm_q(r["extra_info"]["question"]) for r in rows["test"]}
    res["train_test_question_overlap"] = len(overlap)
    if overlap:
        fails.append(f"data: {len(overlap)} questions in both train and test")
    man = os.path.join(os.path.dirname(a.data_dir.rstrip("/")), "PACKAGE_MANIFEST.sha256")
    if os.path.exists(man):
        want = {}
        for line in open(man):
            parts = line.split()
            if len(parts) == 2:
                want[os.path.basename(parts[1])] = parts[0]
        for split in ("train", "test"):
            b = os.path.basename(res[split]["path"])
            res[split]["manifest_sha256_match"] = want.get(b) == res[split]["sha256"]
            if want.get(b) and want.get(b) != res[split]["sha256"]:
                fails.append(f"data {split}: sha256 differs from the handoff manifest")
    out["data"] = res
    return rows


def check_template(a, rows, out, fails):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    res, fixture = {}, {}
    for split in ("train", "test"):
        lens, bad = [], []
        for r in rows[split]:
            text = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            lens.append((len(ids), r["extra_info"]["index"], text, ids, r["prompt"]))
            if not (text.startswith("<bos><|turn>system\n") and text.endswith("<|turn>model\n")
                    and text.count(BOXED_INSTRUCTION) == 1 and text.count("<bos>") == 1):
                bad.append(r["extra_info"]["index"])
        ls = sorted(x[0] for x in lens)
        res[split] = {"n": len(ls), "min": ls[0], "median": ls[len(ls) // 2], "p99": ls[int(0.99 * (len(ls) - 1))],
                      "max": ls[-1], "over_limit": sum(1 for x in ls if x > a.max_prompt), "bad_render": bad[:10],
                      "n_bad_render": len(bad)}
        if res[split]["over_limit"] or bad:
            fails.append(f"template {split}: {res[split]['over_limit']} prompts > {a.max_prompt}, {len(bad)} bad renders")
        pick = [min(lens, key=lambda x: x[0]), lens[0], max(lens, key=lambda x: x[0])]
        fixture[split] = [{"qid": x[1], "n_tokens": x[0], "messages": x[4], "rendered": x[2], "input_ids": x[3]}
                          for x in pick]
    res["eos_token"] = [tok.eos_token, tok.eos_token_id]
    res["pad_token"] = [tok.pad_token, tok.pad_token_id]
    gc = json.load(open(os.path.join(a.model, "generation_config.json")))
    res["generation_config"] = gc
    res["stop_tokens"] = {str(i): tok.convert_ids_to_tokens(i) for i in (gc.get("eos_token_id") or [])}
    out["template"] = res
    fx = os.path.join(os.path.dirname(os.path.abspath(a.out)), "prompt_fixture_gemma.json")
    with open(fx, "w") as f:
        json.dump(fixture, f, indent=1)
    out["template"]["fixture"] = fx


SCORER_CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("bmr", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = {}
for name, sol, gt, extra in json.loads(sys.argv[2]):
    try:
        out[name] = m.compute_score("gsm8k_boxed_train", sol, gt, extra_info=extra)
    except Exception as e:
        out[name] = {"error": type(e).__name__, "msg": str(e)}
print(json.dumps(out))
"""

SCORER_CASES = [  # name, solution, ground truth, extra_info, expected (score, acc, fmt)
    ("correct", "Step by step ... so the answer is \\boxed{72}.", "72", {"index": 0}, (1.0, 1.0, 1.0)),
    ("correct_8k_no_resp_len", "x " * 100 + "\\boxed{72}", "72", {"index": 1}, (1.0, 1.0, 1.0)),
    ("correct_with_resp_len_8192", "\\boxed{72}", "72", {"index": 2, "response_len": 8192}, (1.0, 1.0, 1.0)),
    ("wrong_boxed", "... \\boxed{71}", "72", {"index": 3}, (0.1, 0.0, 1.0)),
    ("unparseable", "The answer is 72.", "72", {"index": 4}, (0.0, 0.0, 0.0)),
    ("comma_number", "\\boxed{1,000}", "1000", {"index": 5}, None),
    ("multiple_boxes", "\\boxed{5} then \\boxed{72}", "72", {"index": 6}, None),
    ("nested_box", "\\boxed{\\frac{144}{2}}", "72", {"index": 7}, None),
]


def check_scorer(a, out, fails):
    res = {"sha256": sha256(a.reward)}
    env = {k: v for k, v in os.environ.items() if not k.startswith("REWARD_")}
    # (1) the scorer's own self-test under Meta's fixture settings (validates the rule itself)
    e1 = dict(env, REWARD_MAX_RESP_LEN="2048", REWARD_PENALTY_SOURCES="gsm8k_boxed_train")
    p = subprocess.run([sys.executable, a.reward], env=e1, capture_output=True, text=True, timeout=300)
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    res["meta_fixture_selftest"] = {"rc": p.returncode, "result": last[-1] if last else p.stdout[-300:]}
    if not last or "PASS" not in last[-1]:
        fails.append(f"scorer: Meta fixture self-test did not PASS: {res['meta_fixture_selftest']}")
    # (2) the run settings: no length penalty anywhere
    e2 = dict(env, REWARD_FORMAT_SCORE="0.1", REWARD_PENALTY_SOURCES="", REWARD_MAX_RESP_LEN=str(a.cap),
              REWARD_OVERLONG_BUFFER="512", REWARD_OVERLONG_PENALTY="1.0")
    cases = [[c[0], c[1], c[2], c[3]] for c in SCORER_CASES]
    p = subprocess.run([sys.executable, "-c", SCORER_CHILD, a.reward, json.dumps(cases)], env=e2,
                       capture_output=True, text=True, timeout=300)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        fails.append(f"scorer: run-settings child failed: {p.stderr[-500:]}")
        out["scorer"] = res
        return
    rows = {}
    for name, _, _, _, want in SCORER_CASES:
        g = got.get(name, {})
        ok = "error" not in g and g.get("length_penalty") == 0.0
        if want is not None:
            ok = ok and abs(g.get("score", -9) - want[0]) < 1e-9 and g.get("acc") == want[1] and g.get("fmt") == want[2]
        rows[name] = {"got": g, "want": want, "ok": ok}
        if not ok:
            fails.append(f"scorer case {name}: got {g}, want {want}")
    res["run_settings_cases"] = rows
    out["scorer"] = res


def lr_lambda(step: int, warmup: int, total: int, num_cycles: float = 0.5, min_lr_ratio: float = 0.0) -> float:
    """Copy of verl get_cosine_schedule_with_warmup.lr_lambda (zero_indexed_step=True, init_lr_ratio=0)."""
    coef, intercept = (1 - min_lr_ratio) * 0.5, (1 + min_lr_ratio) * 0.5
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return max(min_lr_ratio, math.cos(math.pi * float(num_cycles) * 2.0 * progress) * coef + intercept)


def check_schedule(a, out, fails):
    used = [a.peak_lr * lr_lambda(k - 1, a.warmup, a.total_steps) for k in range(1, a.total_steps + 1)]
    logged = [a.peak_lr * lr_lambda(k, a.warmup, a.total_steps) for k in range(1, a.total_steps + 1)]
    checks = {"update_1_uses_0": used[0] == 0.0,
              "update_2_uses_peak_over_warmup": abs(used[1] - a.peak_lr / a.warmup) < 1e-18,
              "update_warmup_plus_1_uses_peak": abs(used[a.warmup] - a.peak_lr) < 1e-18,
              "logged_after_last_update_is_0": abs(logged[-1]) < 1e-18,
              "monotone_after_warmup": all(x >= y for x, y in zip(used[a.warmup:], used[a.warmup + 1:], strict=False))}
    for k, v in checks.items():
        if not v:
            fails.append(f"schedule: {k} failed")
    out["schedule"] = {"peak": a.peak_lr, "warmup": a.warmup, "total_steps": a.total_steps, "checks": checks,
                       "mean_update_used": sum(used) / len(used),
                       "update_used_lr": {str(k + 1): v for k, v in enumerate(used)},
                       "logged_actor_lr": {str(k + 1): v for k, v in enumerate(logged)}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reward", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompt", type=int, default=512)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--total-steps", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--peak-lr", type=float, default=2e-6)
    a = ap.parse_args(argv)
    out, fails = {"args": vars(a)}, []
    rows = check_data(a, out, fails)
    check_template(a, rows, out, fails)
    check_scorer(a, out, fails)
    check_schedule(a, out, fails)
    out["verdict"] = "PASS" if not fails else "FAIL"
    out["failures"] = fails
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    t = out["template"]
    print(f"GEMMA4-GSM8K-PREFLIGHT: {out['verdict']} train={out['data']['train']['rows']} test={out['data']['test']['rows']} "
          f"prompt_tokens_max={t['train']['max']}/{t['test']['max']} (limit {a.max_prompt}) "
          f"scorer={out['scorer'].get('meta_fixture_selftest', {}).get('result', '?')} "
          f"lr(update 1/2/{a.warmup + 1})={out['schedule']['update_used_lr']['1']:.3g}/"
          f"{out['schedule']['update_used_lr']['2']:.3g}/{out['schedule']['update_used_lr'][str(a.warmup + 1)]:.3g} "
          f"out={a.out}")
    for f in fails:
        print("FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
