#!/usr/bin/env python3
"""Gemma-4 trainer-vs-sampler log-prob parity harness (offline, single GPU).

Inputs: a pre-update fixture written by patch_verl_pin_logprob_fixture.py during a 1-step run at
T=1, top_k=-1, top_p=1.0 (fixture_step1.npz). Token IDs come straight from the rollout; nothing is
decoded or re-tokenized. All log-probs here are T=1, full-vocabulary log_softmax of the logits at
position t-1 gathered at token t (next-token shift), restricted to response tokens
(response_mask), i.e. the same convention as the online rollout/trainer values in the fixture.

Stages (run in order; each writes <out>/<stage>.npz, `report` reads them all):
  select  pick --n sequences stratified by length (CPU)                -> selection.json
  hf      HF forwards on the selected token IDs (GPU, one process):
            fp32_eager_unpacked                        precision reference
            bf16_{eager,sdpa,flex_attention}_unpacked  attention backends (verl precision: bf16 params,
                                                       fp32 buffers, bf16 autocast)
            bf16_{...}_packedA / _packedB              verl remove-padding layout: sequences concatenated
                                                       into rows of <= --pack_tokens, position_ids
                                                       restart per sequence, attention_mask=None, verl's
                                                       apply_monkey_patch() applied; A/B = two orders
  vllm    offline vLLM prefill scoring (prompt_logprobs) of the same IDs, weights from the checkpoint
  report  per-token |dlogp| mean / p50 / p99 / max, signed mean, prob-MAE, k3 for each comparison,
          plus |dlogp| vs absolute token position (sliding window = 512 marked) -> summary.{json,md}, *.png

Comparison ladder (what each pair localizes):
  backend   bf16_{sdpa,flex}_unpacked vs bf16_eager_unpacked (and bf16 vs fp32 eager = noise floor)
  packing   bf16_X_packed{A,B} vs bf16_X_unpacked             (packing / position IDs / masks)
  prefill   vllm_prefill vs HF unpacked                        (vLLM vs HF prefill implementation)
  online    online_rollout vs vllm_prefill / HF; online_trainer vs bf16_flex_attention_packedA
            (decode path, KV cache, weight sync, record alignment; trainer reproduction)

Provenance checks (results from different fixtures or selections are never mixed):
  - The fixture must be a parity fixture: its sidecar fixture_step<N>.json says global_step == 1
    (dumped before the first update), temperature == 1, top_k in (-1, 0), top_p == 1 (no truncation);
    the npz's nt__{,min_,max_}global_steps (weight version that generated each sample) must be 0, i.e.
    the rollout came from the initial weights. Otherwise every stage exits, unless --allow_nonparity_fixture.
  - selection.json records the fixture's sha256; hf / vllm / report refuse a selection made from a
    different fixture, and `select` refuses to overwrite a different selection while results exist.
  - Every hf__*/vllm__* result carries a JSON sidecar with the fixture sha256 and the selected
    indices; `report` refuses (exit 1) any result whose sidecar is missing or does not match, or whose
    per-sequence arrays do not match the response lengths. Use one --out directory per fixture.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

SLIDING_WINDOW = 512


# ----------------------------------------------------------------------------- fixture
def load_fixture(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fixture_identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return {"path": os.path.abspath(path), "sha256": h.hexdigest(), "size": os.path.getsize(path)}


def check_parity_config(path, fx, allow):
    """The fixture must come from step 1 (pre-update) at T=1 with no top-k/top-p truncation."""
    side = os.path.splitext(path)[0] + ".json"
    meta = json.load(open(side)) if os.path.exists(side) else None
    problems, cfg = [], {}
    if meta is None:
        problems.append(f"missing sidecar {side}")
    else:
        ro = meta.get("rollout", {})
        cfg = {"global_step": meta.get("global_step"), "temperature": ro.get("temperature"),
               "top_k": ro.get("top_k"), "top_p": ro.get("top_p"), "logprobs_mode": ro.get("logprobs_mode")}
        if cfg["global_step"] != 1:
            problems.append(f"global_step={cfg['global_step']} (need 1: pre-update weights)")
        if cfg["temperature"] is None or float(cfg["temperature"]) != 1.0:
            problems.append(f"temperature={cfg['temperature']} (need 1.0)")
        if cfg["top_k"] not in (-1, 0):
            problems.append(f"top_k={cfg['top_k']} (need -1/0: no top-k truncation)")
        if cfg["top_p"] is None or float(cfg["top_p"]) != 1.0:
            problems.append(f"top_p={cfg['top_p']} (need 1.0: no top-p truncation)")
    for key in ("nt__global_steps", "nt__min_global_steps", "nt__max_global_steps"):
        if key in fx:  # weight version that generated each sample: 0 = initial weights
            try:
                if not np.all(fx[key].astype(np.int64) == 0):
                    problems.append(f"{key} != 0 (rollout not generated by the initial weights)")
            except ValueError:
                problems.append(f"{key} is not numeric")
    if problems:
        msg = "not a step-1, T=1, untruncated parity fixture: " + "; ".join(problems)
        if not allow:
            sys.exit(f"[fixture] {msg} (pass --allow_nonparity_fixture to override)")
        print(f"[fixture] WARNING: {msg}", flush=True)
    return cfg


def existing_results(out):
    return sorted(fn for fn in os.listdir(out) if fn.endswith(".npz") and fn.startswith(("hf__", "vllm__")))


def extract(fx, i):
    """Unpad sequence i exactly as the rollout produced it; assert the layout assumptions."""
    P = fx["prompts"].shape[1]
    am = fx["attention_mask"][i].astype(bool)
    rmask = fx["response_mask"][i].astype(bool)
    R = int(rmask.sum())
    assert rmask[:R].all() and not rmask[R:].any(), f"seq {i}: response_mask is not a contiguous prefix"
    assert (am[P:] == rmask).all(), f"seq {i}: attention_mask[P:] != response_mask"
    prompt_ids = fx["prompts"][i][am[:P]]
    resp_ids = fx["responses"][i][:R]
    ids = np.concatenate([prompt_ids, resp_ids]).astype(np.int64)
    pos = fx["position_ids"][i][am]
    assert (pos == np.arange(len(ids))).all(), f"seq {i}: position_ids are not 0..L-1"
    rec = {"idx": int(i), "ids": ids, "P": int(len(prompt_ids)), "R": R,
           "online_rollout": fx["rollout_log_probs"][i][:R].astype(np.float64),
           "online_trainer": fx["old_log_probs"][i][:R].astype(np.float64)}
    if "old_log_probs_repeat" in fx:
        rec["online_trainer_repeat"] = fx["old_log_probs_repeat"][i][:R].astype(np.float64)
    return rec


def select(fx, n, seed, max_len):
    am = fx["attention_mask"].astype(bool)
    L = am.sum(1)
    R = fx["response_mask"].astype(bool).sum(1)
    rng = np.random.default_rng(seed)
    ok = (L >= 128) & (L < max_len)  # flex decoding kernel is used below 128 query tokens; vLLM needs L+1 <= max_len
    bins = [(0, SLIDING_WINDOW), (SLIDING_WINDOW, 2048), (2048, 1 << 30)]
    quota = [n // 4, (3 * n) // 8]
    quota.append(n - sum(quota))
    chosen = []
    for (lo, hi), q in zip(bins, quota):
        cand = np.where(ok & (L > lo) & (L <= hi))[0]
        chosen += rng.choice(cand, size=min(q, len(cand)), replace=False).tolist()
    cap = np.where(ok & (R == fx["responses"].shape[1]))[0]  # cap-hit responses, if any fit
    if len(cap) and not set(cap.tolist()) & set(chosen):
        chosen[-1] = int(rng.choice(cap))
    return sorted(int(c) for c in chosen)


# ----------------------------------------------------------------------------- HF
def chunked_logprob(logits, target, chunk=2048):
    import torch
    out = torch.empty(target.shape[0], dtype=torch.float32, device=logits.device)
    for s in range(0, target.shape[0], chunk):
        lp = torch.log_softmax(logits[s:s + chunk].float(), dim=-1)
        out[s:s + chunk] = lp.gather(-1, target[s:s + chunk, None]).squeeze(-1)
    return out


def build_hf(model_path, attn, dtype_name, verl_patch):
    import torch
    from transformers import AutoConfig

    if attn == "flex_attention" or verl_patch:
        import verl.models.transformers.monkey_patch as vmp  # import applies the [GEMMA4] head_dim>256 flex tiles
    from verl.utils.model import get_hf_auto_model_class

    cfg = AutoConfig.from_pretrained(model_path)
    cls = get_hf_auto_model_class(cfg)
    dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
    model, info = cls.from_pretrained(model_path, torch_dtype=dtype, attn_implementation=attn,
                                      output_loading_info=True)
    if dtype_name == "bf16":  # verl FSDP MixedPrecision: param bf16, buffer fp32, autocast bf16
        for _, b in model.named_buffers():
            if b.is_floating_point():
                b.data = b.data.float()
    if verl_patch:
        vmp.apply_monkey_patch(model=model, use_remove_padding=True, ulysses_sp_size=1,
                               use_fused_kernels=False, fused_kernels_backend=None)
    model.eval().cuda()
    meta = {"class": type(model).__name__, "auto_class": cls.__name__, "attn": attn, "dtype": dtype_name,
            "verl_patch": verl_patch, "missing_keys": list(info.get("missing_keys", []))[:20],
            "n_missing": len(info.get("missing_keys", [])), "n_unexpected": len(info.get("unexpected_keys", [])),
            "buffer_dtypes": sorted({str(b.dtype) for _, b in model.named_buffers()})}
    return model, meta


def hf_forward(model, rows, autocast):
    """rows: list of (ids_row, pos_row, spans); span = (rec_key, offset, P, R, resp_ids). Returns {key: lp}."""
    import contextlib

    import torch
    res = {}
    for ids_row, pos_row, spans in rows:
        inp = torch.from_numpy(ids_row)[None].cuda()
        pos = torch.from_numpy(pos_row)[None].cuda()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            logits = model(input_ids=inp, attention_mask=None, position_ids=pos, use_cache=False).logits[0]
        for key, o, P, R, resp in spans:
            tgt = torch.from_numpy(resp).cuda()
            res[key] = chunked_logprob(logits[o + P - 1:o + P + R - 1], tgt).double().cpu().numpy()
        del logits
        torch.cuda.empty_cache()
    return res


def unpacked_rows(recs):
    return [(r["ids"], np.arange(len(r["ids"]), dtype=np.int64), [(r["idx"], 0, r["P"], r["R"], r["ids"][r["P"]:])])
            for r in recs]


def packed_rows(recs, budget):
    rows, cur, cur_len = [], [], 0
    for r in recs:
        if cur and cur_len + len(r["ids"]) > budget:
            rows.append(cur)
            cur, cur_len = [], 0
        cur.append(r)
        cur_len += len(r["ids"])
    if cur:
        rows.append(cur)
    out = []
    for row in rows:
        ids, pos, spans, o = [], [], [], 0
        for r in row:
            ids.append(r["ids"])
            pos.append(np.arange(len(r["ids"]), dtype=np.int64))
            spans.append((r["idx"], o, r["P"], r["R"], r["ids"][r["P"]:]))
            o += len(r["ids"])
        out.append((np.concatenate(ids), np.concatenate(pos), spans))
    return out


def stage_hf(args, recs):
    import torch
    configs = [("fp32", "eager", False, False)]
    for b in args.backends:
        configs.append(("bf16", b, False, False))
        configs.append(("bf16", b, True, True))
    layout = {}
    for dtype_name, attn, packed, verl_patch in configs:
        t0 = time.time()
        model, meta = build_hf(args.model, attn, dtype_name, verl_patch)
        names = {}
        if not packed:
            name = f"{dtype_name}_{attn}_unpacked"
            names[name] = hf_forward(model, unpacked_rows(recs), autocast=(dtype_name == "bf16"))
        else:
            for tag, order in (("A", recs), ("B", recs[::-1])):
                rows = packed_rows(order, args.pack_tokens)
                layout[f"{attn}_packed{tag}"] = [[(s[0], s[1], s[2], s[3]) for s in r[2]] for r in rows]
                names[f"{dtype_name}_{attn}_packed{tag}"] = hf_forward(model, rows, autocast=True)
        for name, res in names.items():
            np.savez(os.path.join(args.out, f"hf__{name}.npz"), **{str(k): v for k, v in res.items()})
            meta_out = dict(meta, seconds=round(time.time() - t0, 1), **args.provenance)
            json.dump(meta_out, open(os.path.join(args.out, f"hf__{name}.json"), "w"), indent=1)
            print(f"[hf] {name}: {len(res)} seqs, {meta_out['seconds']} s, class={meta['class']}, "
                  f"missing_keys={meta['n_missing']}, buffers={meta['buffer_dtypes']}", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump({k: [[list(map(int, s)) for s in row] for row in v] for k, v in layout.items()},
              open(os.path.join(args.out, "packed_layout.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- vLLM
def stage_vllm(args, recs):
    os.environ.pop("RAY_ADDRESS", None)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.vllm_mem,
              max_model_len=args.max_len + 64, enable_prefix_caching=False, enable_chunked_prefill=True,
              max_num_batched_tokens=32768, logprobs_mode="processed_logprobs", seed=0)
    sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=0, logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"].tolist()) for r in recs], sp)
    res = {}
    for r, o in zip(recs, outs):
        pl = o.prompt_logprobs
        assert len(pl) == len(r["ids"]), f"seq {r['idx']}: prompt_logprobs len {len(pl)} != {len(r['ids'])}"
        res[str(r["idx"])] = np.array([pl[r["P"] + j][int(r["ids"][r["P"] + j])].logprob for j in range(r["R"])],
                                      dtype=np.float64)
    np.savez(os.path.join(args.out, "vllm__prefill.npz"), **res)
    import vllm
    json.dump({"vllm": vllm.__version__, "seconds": round(time.time() - t0, 1), "logprobs_mode": "processed_logprobs",
               "note": "prompt_logprobs = prefill scoring path, weights loaded from the checkpoint", **args.provenance},
              open(os.path.join(args.out, "vllm__prefill.json"), "w"), indent=1)
    print(f"[vllm] prefill scored {len(res)} seqs in {time.time() - t0:.1f} s", flush=True)


# ----------------------------------------------------------------------------- report
def stats(a, b):
    d = a - b
    ad = np.abs(d)
    k3 = np.exp(d) - d - 1
    return {"n_tok": int(d.size), "mean_abs": float(ad.mean()), "p50_abs": float(np.percentile(ad, 50)),
            "p99_abs": float(np.percentile(ad, 99)), "max_abs": float(ad.max()), "mean_signed": float(d.mean()),
            "prob_mae": float(np.abs(np.exp(a) - np.exp(b)).mean()), "k3_mean": float(k3.mean())}


def position_curve(per_seq_a, per_seq_b, recs, edges):
    """Mean / p99 |dlogp| binned by absolute position of the target token in its sequence."""
    pos_all, d_all = [], []
    for r in recs:
        k = r["idx"]
        if k not in per_seq_a or k not in per_seq_b:
            continue
        pos_all.append(r["P"] + np.arange(r["R"]))
        d_all.append(np.abs(per_seq_a[k] - per_seq_b[k]))
    pos, d = np.concatenate(pos_all), np.concatenate(d_all)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pos >= lo) & (pos < hi)
        if m.sum():
            out.append((int(lo), int(hi), int(m.sum()), float(d[m].mean()), float(np.percentile(d[m], 99))))
    return out


def load_checked_results(args, recs):
    """Load hf__/vllm__ results; refuse anything not computed from this fixture and selection."""
    prov = args.provenance
    lengths = {r["idx"]: r["R"] for r in recs}
    series, problems = {}, []
    for fn in existing_results(args.out):
        side = os.path.join(args.out, fn[:-4] + ".json")
        meta = json.load(open(side)) if os.path.exists(side) else {}
        if meta.get("fixture_sha256") != prov["fixture_sha256"]:
            problems.append(f"{fn}: fixture_sha256={meta.get('fixture_sha256')!r}, "
                            f"current fixture is {prov['fixture_sha256']}")
            continue
        if meta.get("selected") != prov["selected"]:
            problems.append(f"{fn}: computed for a different selection")
            continue
        z = np.load(os.path.join(args.out, fn))
        vals = {int(k): z[k] for k in z.files}
        bad = [k for k, n in lengths.items() if k not in vals or len(vals[k]) != n]
        extra = sorted(set(vals) - set(lengths))
        if bad or extra:
            problems.append(f"{fn}: missing/wrong-length sequences {bad[:5]}, unexpected sequences {extra[:5]}")
            continue
        series[fn.split("__", 1)[1][:-4]] = vals
    if problems:
        sys.exit("[report] refusing to mix results (use one --out per fixture):\n  " + "\n  ".join(problems))
    return series


def stage_report(args, recs, fx):
    series = {"online_rollout": {r["idx"]: r["online_rollout"] for r in recs},
              "online_trainer": {r["idx"]: r["online_trainer"] for r in recs}}
    if "online_trainer_repeat" in recs[0]:
        series["online_trainer_repeat"] = {r["idx"]: r["online_trainer_repeat"] for r in recs}
    series.update(load_checked_results(args, recs))
    # comparison ladder: (stage, a, b); a pair is skipped if either series is missing
    pairs = [("noise", "bf16_eager_unpacked", "fp32_eager_unpacked")]
    for b in ("sdpa", "flex_attention"):
        pairs.append(("backend", f"bf16_{b}_unpacked", "bf16_eager_unpacked"))
    for b in ("eager", "sdpa", "flex_attention"):
        pairs += [("packing", f"bf16_{b}_packedA", f"bf16_{b}_unpacked"),
                  ("packing", f"bf16_{b}_packedB", f"bf16_{b}_unpacked")]
    pairs += [("prefill", "prefill", "fp32_eager_unpacked"), ("prefill", "prefill", "bf16_eager_unpacked"),
              ("prefill", "prefill", "bf16_flex_attention_packedA"),
              ("online", "online_rollout", "prefill"), ("online", "online_rollout", "fp32_eager_unpacked"),
              ("online", "online_trainer", "bf16_flex_attention_packedA"),
              ("online", "online_trainer", "fp32_eager_unpacked"),
              ("online", "online_trainer_repeat", "online_trainer"),
              ("online", "online_trainer", "online_rollout")]
    edges = np.array([0, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384])
    rows, curves = [], {}
    for stage, a, b in pairs:
        if a not in series or b not in series:
            continue
        keys = [r["idx"] for r in recs if r["idx"] in series[a] and r["idx"] in series[b]]
        va = np.concatenate([series[a][k] for k in keys])
        vb = np.concatenate([series[b][k] for k in keys])
        s = dict(stage=stage, a=a, b=b, n_seq=len(keys), **stats(va, vb))
        rows.append(s)
        curves[f"{a}__vs__{b}"] = position_curve(series[a], series[b], recs, edges)
    # whole-batch online mismatch (all sequences in the fixture), same T=1 convention on both sides
    am = fx["response_mask"].astype(bool)
    full = stats(fx["old_log_probs"][am].astype(np.float64), fx["rollout_log_probs"][am].astype(np.float64))
    p_len = fx["attention_mask"][:, :fx["prompts"].shape[1]].astype(bool).sum(1)
    pos_full = (p_len[:, None] + np.arange(am.shape[1])[None, :])[am]
    d_full = np.abs(fx["old_log_probs"][am].astype(np.float64) - fx["rollout_log_probs"][am].astype(np.float64))
    curve_full = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pos_full >= lo) & (pos_full < hi)
        if m.sum():
            curve_full.append((int(lo), int(hi), int(m.sum()), float(d_full[m].mean()), float(np.percentile(d_full[m], 99))))
    curves["FULL_BATCH online_trainer__vs__online_rollout"] = curve_full
    rows.append(dict(stage="online", a="FULL_BATCH online_trainer", b="online_rollout",
                     n_seq=int(am.any(1).sum()), **full))
    summary = {"fixture": args.fixture, "fixture_sha256": args.provenance["fixture_sha256"],
               "parity_config": args.parity_config, "n_selected": len(recs), "selected": [r["idx"] for r in recs],
               "lengths": [len(r["ids"]) for r in recs], "pairs": rows, "full_batch_online_trainer_vs_rollout": full,
               "position_curves": curves}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Fixture `{args.fixture}` (sha256 `{args.provenance['fixture_sha256']}`), "
                f"config {json.dumps(args.parity_config)}; all result files verified against this fixture and "
                f"selection.\n\n")
        f.write("| stage | A | B | seqs | tokens | mean abs | p50 | p99 | max | mean signed (A-B) | prob MAE | k3 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s in rows:
            f.write(f"| {s['stage']} | `{s['a']}` | `{s['b']}` | {s['n_seq']} | {s['n_tok']} | {s['mean_abs']:.4f} | "
                    f"{s['p50_abs']:.4f} | {s['p99_abs']:.3f} | {s['max_abs']:.2f} | {s['mean_signed']:+.4f} | "
                    f"{s['prob_mae']:.4f} | {s['k3_mean']:.4f} |\n")
        f.write(f"\nFull batch ({int(am.sum())} response tokens), online trainer vs online rollout: "
                + ", ".join(f"{k}={v:.4g}" for k, v in full.items()) + "\n")
    print(open(os.path.join(args.out, "summary.md")).read())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib not available; skipped plots (curves are in summary.json)")
        return
    groups = {}
    for name, cur in curves.items():
        stage = next(s["stage"] for s in rows if f"{s['a']}__vs__{s['b']}" == name)
        groups.setdefault(stage, []).append((name, cur))
    for stage, items in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for name, cur in items:
            x = [(lo + hi) / 2 for lo, hi, *_ in cur]
            axes[0].plot(x, [c[3] for c in cur], marker="o", ms=3, label=name.replace("__vs__", " vs "))
            axes[1].plot(x, [c[4] for c in cur], marker="o", ms=3)
        for ax, t in zip(axes, ("mean |dlogp| (nats)", "p99 |dlogp| (nats)")):
            ax.axvline(SLIDING_WINDOW, color="k", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("absolute token position in sequence (sliding window 512 dashed)")
            ax.set_title(t)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"log-prob error vs position: {stage}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"err_vs_pos_{stage}.png"), dpi=120)
        plt.close(fig)
    print(f"[report] plots: {sorted(fn for fn in os.listdir(args.out) if fn.endswith('.png'))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "hf", "vllm", "report"])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/meta-RL/models/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--pack_tokens", type=int, default=16384, help="= ppo_max_token_len_per_gpu")
    ap.add_argument("--backends", default="eager,sdpa,flex_attention")
    ap.add_argument("--vllm_mem", type=float, default=0.6)
    ap.add_argument("--allow_nonparity_fixture", action="store_true",
                    help="run even if the fixture is not step 1 / T=1 / untruncated (results are then not a parity test)")
    args = ap.parse_args()
    args.backends = [b for b in args.backends.split(",") if b]
    os.makedirs(args.out, exist_ok=True)
    fx = load_fixture(args.fixture)
    ident = fixture_identity(args.fixture)
    cfg = check_parity_config(args.fixture, fx, args.allow_nonparity_fixture)
    sel_path = os.path.join(args.out, "selection.json")
    old = json.load(open(sel_path)) if os.path.exists(sel_path) else None
    if args.stage == "select":
        sel = select(fx, args.n, args.seed, args.max_len)
        if old is not None and (old.get("fixture_sha256"), old.get("indices")) != (ident["sha256"], sel) \
                and existing_results(args.out):
            sys.exit(f"[select] {args.out} already holds results for another fixture/selection "
                     f"({len(existing_results(args.out))} files); use a fresh --out")
        json.dump({"fixture": ident["path"], "fixture_sha256": ident["sha256"], "fixture_size": ident["size"],
                   "parity_config": cfg, "seed": args.seed, "n": args.n, "indices": sel},
                  open(sel_path, "w"), indent=1)
        print(f"[select] {len(sel)} sequences: {sel}")
    elif old is None:
        sys.exit(f"[{args.stage}] no selection.json in {args.out}; run the `select` stage first")
    elif old.get("fixture_sha256") != ident["sha256"]:
        sys.exit(f"[{args.stage}] {sel_path} was made from fixture sha256={old.get('fixture_sha256')!r}, "
                 f"not from {args.fixture} (sha256={ident['sha256']}); use the matching --out or rerun `select`")
    sel = json.load(open(sel_path))["indices"]
    args.provenance = {"fixture_sha256": ident["sha256"], "selected": sel}
    args.parity_config = cfg
    recs = [extract(fx, i) for i in sel]
    print(f"[{args.stage}] lengths={[len(r['ids']) for r in recs]} prompt={[r['P'] for r in recs]}", flush=True)
    if args.stage == "hf":
        stage_hf(args, recs)
    elif args.stage == "vllm":
        stage_vllm(args, recs)
    elif args.stage == "report":
        stage_report(args, recs, fx)


if __name__ == "__main__":
    sys.exit(main())
