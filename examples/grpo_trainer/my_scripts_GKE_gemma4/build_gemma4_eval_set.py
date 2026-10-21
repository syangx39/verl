#!/usr/bin/env python3
"""Build the frozen Gemma-4 eval set (PR #9, step 1): 5,000 OpenMathInstruct-2 problems with no training overlap, plus
the prompt-token fixture that both the GPU (verl) and TPU sides must reproduce.

Selection rule (deterministic, independent of row order):
  1. key(q) = lowercase question with every non-alphanumeric character removed (catches whitespace / punctuation /
     LaTeX-spacing variants, not paraphrases);
  2. candidates = rows of --val whose key is not the key of ANY row of --train (the training file the recipe uses,
     not only the prompts a run happens to draw), de-duplicated by key (lowest val index kept);
  3. order candidates by sha256(key) and keep the first --n.
Outputs in --out-dir:
  eval_<n>.parquet              rows copied verbatim from --val (same schema; extra_info.index = val row index),
                                in selection order -> data.val_files
  eval_<n>_prompt_tokens.parquet  eval_pos, val_index, key_sha256, n_prompt_tokens, prompt_token_ids: the token ids
                                verl's agent loop sends to vLLM (Continuous Token builder, same tokenizer/processor and
                                hf model_type as the rollout), i.e. the current double-templated input, frozen
  manifest.json                 rule, counts, input/output sha256, tokenizer file sha256, builder, verl pin, stats
--check-fixture <PR #8 fixture npz>: before writing, rebuild the prompt ids of the fixture's training samples with the
same builder and require exact equality with the ids the online run used (validates the offline builder).
Run on a pod (needs the image's verl + transformers):
  python3 build_gemma4_eval_set.py --out-dir /workspace/meta-RL/data/openmathinstruct2/gemma4_eval_5k \
      --check-fixture /workspace/meta-RL/logprob_parity/fixture_fixed_1021_0813/fixture_step1.npz
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def key_of(q):
    return _NON_ALNUM.sub("", q.lower())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_builder(model_path):
    """The rollout's prompt tokenizer: same objects and arguments as verl's AgentLoopWorker/AgentLoopBase."""
    from transformers import AutoConfig

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    tok = hf_tokenizer(model_path, trust_remote_code=False)
    proc = hf_processor(model_path, trust_remote_code=False)
    model_type = AutoConfig.from_pretrained(model_path).model_type
    builder = create_continuous_token_builder(tok, hf_model_type=model_type, chat_template_kwargs={},
                                              mm_processor_kwargs={}, processor=proc)
    return builder, tok, proc, model_type


def check_against_fixture(builder, fixture, train_rows_by_index, max_n):
    """Rebuild the prompts of the online fixture's samples and compare token ids exactly."""
    z = np.load(fixture, allow_pickle=True)
    prompts = z["prompts"]
    attn = z["attention_mask"]
    extra = z["nt__extra_info"]
    plen = prompts.shape[1]
    seen, n_ok, n_bad, bad = set(), 0, 0, []
    for i in range(prompts.shape[0]):
        info = ast.literal_eval(str(extra[i]))
        idx = int(info["index"])
        if idx in seen:
            continue
        seen.add(idx)
        online = prompts[i][attn[i][:plen].astype(bool)].tolist()
        msgs = [dict(m) for m in train_rows_by_index[idx]["prompt"]]
        offline = builder.build_initial_tokens(msgs)
        if offline == online:
            n_ok += 1
        else:
            n_bad += 1
            bad.append({"train_index": idx, "online_len": len(online), "offline_len": len(offline)})
        if len(seen) >= max_n:
            break
    return {"fixture": fixture, "fixture_sha256": sha256_file(fixture), "n_prompts_compared": len(seen),
            "n_exact": n_ok, "n_mismatch": n_bad, "mismatches": bad[:10]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="/workspace/meta-RL/data/openmathinstruct2/train.parquet")
    ap.add_argument("--val", default="/workspace/meta-RL/data/openmathinstruct2/val.parquet")
    ap.add_argument("--model", default="/workspace/meta-RL/models/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check-fixture", default=None, help="PR #8 fixture npz (online prompt ids of training samples)")
    ap.add_argument("--check-max", type=int, default=256)
    args = ap.parse_args()
    t0 = time.time()

    train = pq.read_table(args.train)
    train_q = train.column("extra_info").to_pylist()
    train_keys = {key_of(x["question"]) for x in train_q}
    val = pq.read_table(args.val)
    val_rows = val.to_pylist()
    first_by_key, n_overlap = {}, 0
    for i, r in enumerate(val_rows):
        k = key_of(r["extra_info"]["question"])
        if k in train_keys:
            n_overlap += 1
            continue
        if k not in first_by_key:
            first_by_key[k] = i
    ranked = sorted(first_by_key.items(), key=lambda kv: hashlib.sha256(kv[0].encode()).hexdigest())
    if len(ranked) < args.n:
        sys.exit(f"only {len(ranked)} non-overlapping unique val questions, need {args.n}")
    chosen = ranked[:args.n]
    for k, i in chosen:
        assert int(val_rows[i]["extra_info"]["index"]) == i, "val extra_info.index != row index"

    builder, tok, proc, model_type = make_builder(args.model)
    fixture_check = None
    if args.check_fixture:
        by_index = {int(r["extra_info"]["index"]): r for r in train.select(["prompt", "extra_info"]).to_pylist()}
        fixture_check = check_against_fixture(builder, args.check_fixture, by_index, args.check_max)
        print("FIXTURE CHECK", json.dumps({k: v for k, v in fixture_check.items() if k != "mismatches"}))
        if fixture_check["n_mismatch"] or not fixture_check["n_exact"]:
            sys.exit(f"offline builder does not reproduce the online prompt ids: {fixture_check['mismatches'][:3]}")

    os.makedirs(args.out_dir, exist_ok=True)
    eval_path = os.path.join(args.out_dir, f"eval_{args.n}.parquet")
    tok_path = os.path.join(args.out_dir, f"eval_{args.n}_prompt_tokens.parquet")
    pq.write_table(val.take(pa.array([i for _, i in chosen])), eval_path)

    ids, lens = [], []
    for k, i in chosen:
        p = builder.build_initial_tokens([dict(m) for m in val_rows[i]["prompt"]])
        ids.append(p)
        lens.append(len(p))
    tok_table = pa.table({
        "eval_pos": pa.array(list(range(len(chosen))), pa.int32()),
        "val_index": pa.array([i for _, i in chosen], pa.int64()),
        "key_sha256": pa.array([hashlib.sha256(k.encode()).hexdigest() for k, _ in chosen]),
        "n_prompt_tokens": pa.array(lens, pa.int32()),
        "prompt_token_ids": pa.array(ids, pa.list_(pa.int32())),
    })
    pq.write_table(tok_table, tok_path)

    tok_files = {}
    for fn in sorted(os.listdir(args.model)):
        if fn.startswith(("tokenizer", "special_tokens", "chat_template", "processor", "preprocessor")) or fn.endswith(".jinja"):
            tok_files[fn] = sha256_file(os.path.join(args.model, fn))
    try:
        import verl
        verl_head = subprocess.check_output(["git", "-C", os.path.dirname(os.path.dirname(verl.__file__)), "rev-parse",
                                             "HEAD"], text=True).strip()
    except Exception as e:
        verl_head = f"unknown ({e!r})"
    lens_np = np.array(lens)
    manifest = {
        "kind": "gemma4_eval_set", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": {"key": "lowercase question, every non-alphanumeric character removed ([\\W_]+ -> '')",
                 "exclude": "val rows whose key equals the key of ANY train row", "dedup": "by key, lowest val index kept",
                 "order": "sha256(key) ascending", "n": args.n},
        "inputs": {"train": {"path": args.train, "sha256": sha256_file(args.train), "rows": train.num_rows,
                             "unique_keys": len(train_keys)},
                   "val": {"path": args.val, "sha256": sha256_file(args.val), "rows": val.num_rows,
                           "rows_overlapping_train": n_overlap, "unique_non_overlapping_keys": len(first_by_key)}},
        "outputs": {"eval": {"path": eval_path, "sha256": sha256_file(eval_path), "rows": len(chosen)},
                    "prompt_tokens": {"path": tok_path, "sha256": sha256_file(tok_path)}},
        "prompt_tokens": {"builder": type(builder).__module__ + "." + type(builder).__name__, "hf_model_type": model_type,
                          "processor": type(proc).__name__ if proc is not None else None,
                          "tokenizer": type(tok).__name__, "tokenizer_files_sha256": tok_files, "model_path": args.model,
                          "verl_head": verl_head, "rendering": "current input kept: the parquet's user content already "
                          "carries a <start_of_turn> template and verl applies the Gemma-4 chat template on top",
                          "len_min": int(lens_np.min()), "len_p50": float(np.median(lens_np)),
                          "len_max": int(lens_np.max()), "len_mean": float(lens_np.mean())},
        "fixture_check": fixture_check,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(json.dumps({"eval": manifest["outputs"], "counts": manifest["inputs"]["val"],
                      "prompt_len": {k: v for k, v in manifest["prompt_tokens"].items() if k.startswith("len_")}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
