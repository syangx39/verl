#!/usr/bin/env python3
"""Meta GSM8K-boxed reproduction: build the verl parquets, the model copy with the
frozen stop set, the prompt fixture and a manifest.

Inputs (from Meta's package):
  data/train.jsonl  (7,473 rows)  data/test.jsonl (1,319 rows)
  each row: {"messages":[{"role":"system",...},{"role":"user",...}], "answer": "<gold verbatim>"}
Optionally re-verify them against public GSM8K with Meta's scripts/build_gsm8k_boxed.py --verify.

Outputs (in --out):
  gsm8k_boxed_train.parquet       7,473 rows, data_source=gsm8k_boxed_train
  gsm8k_boxed_test512.parquet     first 512 test rows, data_source=gsm8k_boxed_test512, index 20,000,000+ (Meta: max_eval_samples=512)
  gsm8k_boxed_test.parquet        all 1,319 test rows, data_source=gsm8k_boxed_test, index 10,000,000+ (our full-set diagnostic)
  (train index 0..7472; the three ranges are disjoint so val dumps pair by (source, qid))
  prompt_fixture.json             10 prompts: messages, rendered text, full token ids (Base tokenizer)
  MANIFEST.sha256
  <model_out>/                    copy of Qwen3-0.6B-Base whose generation_config.json eos_token_id is
                                  [151645, 151643]  -- reproduces Meta's vllm_stop_token_ids=[151645] on top
                                  of the tokenizer EOS 151643 (vLLM always stops on both)

verl row format: prompt = the two messages unchanged (system + user); the chat template is applied by
verl at load time; reward_model.ground_truth = the gold string VERBATIM (thousands separators kept,
exactly as Meta ships it -- normalization is the reward function's job, see boxed_math_reward.py).
"""
import argparse
import hashlib
import json
import os
import shutil

import pandas as pd

EXPECTED = {"train": 7473, "test": 1319}


def read_jsonl(p):
  return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def rows_to_df(rows, source, offset=0):
  out = []
  for i, r in enumerate(rows):
    assert [m["role"] for m in r["messages"]] == ["system", "user"], f"row {i}: unexpected roles"
    out.append({
        "data_source": source,
        "prompt": [{"role": m["role"], "content": m["content"]} for m in r["messages"]],
        "reward_model": {"style": "rule", "ground_truth": r["answer"]},   # verbatim
        "extra_info": {"question": r["messages"][1]["content"], "index": offset + i, "split": source,
                       "answer": r["answer"]},
    })
  return pd.DataFrame(out)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--meta_data", required=True, help="dir with Meta's train.jsonl / test.jsonl")
  ap.add_argument("--out", required=True)
  ap.add_argument("--model_in", required=True, help="HF Qwen/Qwen3-0.6B-Base local dir")
  ap.add_argument("--model_out", required=True, help="where to write the stop-set-patched copy")
  ap.add_argument("--n_eval", type=int, default=512)
  ap.add_argument("--overwrite", action="store_true", help="replace an existing --model_out")
  ap.add_argument("--prompt_example", default=None,
                  help="Meta's reference/prompt_example.json: row 0 of test.jsonl must render to exactly its token ids (99 for row 0)")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  tr = read_jsonl(os.path.join(args.meta_data, "train.jsonl"))
  te = read_jsonl(os.path.join(args.meta_data, "test.jsonl"))
  assert len(tr) == EXPECTED["train"] and len(te) == EXPECTED["test"], (len(tr), len(te))
  sys_prompts = {m["content"] for r in tr + te for m in r["messages"] if m["role"] == "system"}
  assert len(sys_prompts) == 1, f"system prompt not uniform: {sys_prompts}"
  print("system prompt:", repr(next(iter(sys_prompts))))

  dtr = rows_to_df(tr, "gsm8k_boxed_train")
  dte = rows_to_df(te, "gsm8k_boxed_test", offset=10_000_000)
  dte512 = dte.head(args.n_eval).copy()
  dte512["data_source"] = f"gsm8k_boxed_test{args.n_eval}"
  # its own index range (20,000,000+) so dumps can be paired by (source, qid) even though the questions
  # are a subset of the full test set (paired_eval_bootstrap.py --key qid)
  dte512["extra_info"] = [dict(e, index=20_000_000 + i, split=f"gsm8k_boxed_test{args.n_eval}") for i, e in enumerate(dte512["extra_info"])]
  # train/test overlap check on normalized question text
  norm = lambda s: " ".join(s.split())
  trq = {norm(e["question"]) for e in dtr["extra_info"]}
  overlap = sum(norm(e["question"]) in trq for e in dte["extra_info"])
  print(f"train rows {len(dtr)} (unique questions {len(trq)}) | test rows {len(dte)} | test∩train = {overlap}")
  # thousands separators in gold (Meta: 93 of 8,792)
  commas = sum("," in r["answer"] for r in tr + te)
  print(f"gold strings with thousands separators: {commas} (Meta reports 93)")

  dtr.to_parquet(f"{args.out}/gsm8k_boxed_train.parquet", index=False)
  dte512.to_parquet(f"{args.out}/gsm8k_boxed_test{args.n_eval}.parquet", index=False)
  dte.to_parquet(f"{args.out}/gsm8k_boxed_test.parquet", index=False)

  # ---- model copy with the frozen stop set (guarded: never touch the source checkpoint)
  src, dst = os.path.realpath(args.model_in), os.path.realpath(args.model_out)
  if src == dst or dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
    raise SystemExit(f"refusing: --model_out {dst} must not equal, contain or be inside --model_in {src}")
  if not os.path.exists(os.path.join(src, "model.safetensors")):
    raise SystemExit(f"--model_in {src} has no model.safetensors")
  if os.path.exists(dst):
    if not args.overwrite:
      raise SystemExit(f"--model_out {dst} exists; pass --overwrite to replace it")
    shutil.rmtree(dst)
  shutil.copytree(src, dst)
  gpath = os.path.join(args.model_out, "generation_config.json")
  gcfg = json.load(open(gpath)) if os.path.exists(gpath) else {}
  before = gcfg.get("eos_token_id")
  gcfg["eos_token_id"] = [151645, 151643]
  json.dump(gcfg, open(gpath, "w"), indent=2)
  print(f"model copy: eos_token_id {before} -> {gcfg['eos_token_id']} (Meta: vllm_stop_token_ids=[151645] + tokenizer eos)")

  # ---- prompt fixture with the Base tokenizer
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(args.model_out)
  has_tmpl = bool(getattr(tok, "chat_template", None))
  print("tokenizer has chat_template:", has_tmpl, "| eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token, tok.pad_token_id)
  fx = []
  for df, n in ((dtr, 5), (dte512, 5)):
    for _, r in df.head(n).iterrows():
      msgs = [dict(m) for m in r["prompt"]]
      text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
      ids = tok(text, add_special_tokens=False)["input_ids"]
      fx.append({"source": r["data_source"], "index": int(r["extra_info"]["index"]), "messages": msgs,
                 "rendered_text": text, "token_ids": ids, "n_tokens": len(ids), "ground_truth": r["reward_model"]["ground_truth"]})
  json.dump({"model": args.model_out, "tokenizer_sha256": hashlib.sha256(open(f"{args.model_out}/tokenizer.json", "rb").read()).hexdigest(),
             "rows": fx}, open(f"{args.out}/prompt_fixture.json", "w"), ensure_ascii=False, indent=1)
  print("fixture n_tokens:", [r["n_tokens"] for r in fx], "| head", fx[0]["token_ids"][:3], "tail", fx[0]["token_ids"][-3:])
  print("rendered example:\n" + fx[0]["rendered_text"][:400])

  # ---- gate: Meta's rendered prompt for test row 0 must be reproduced token for token
  if args.prompt_example:
    import transformers
    ref = json.load(open(args.prompt_example))
    msgs = [dict(m) for m in dte.iloc[0]["prompt"]]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    same_msgs = msgs == ref["messages"]
    same_text = text == ref["rendered_prompt"]
    same_ids = ids == ref["prompt_token_ids"]
    print(f"prompt_example check (transformers {transformers.__version__}, Meta rendered with {ref.get('transformers_version_used_to_render')}): "
          f"messages {'match' if same_msgs else 'DIFFER'} | rendered text {'match' if same_text else 'DIFFER'} | "
          f"token ids {'match' if same_ids else 'DIFFER'} ({len(ids)} vs {len(ref['prompt_token_ids'])})")
    if not same_ids:
      k = next((i for i, (x, y) in enumerate(zip(ids, ref["prompt_token_ids"])) if x != y), min(len(ids), len(ref["prompt_token_ids"])))
      print(f"  first divergence at position {k}: ours {ids[max(0,k-3):k+3]} vs meta {ref['prompt_token_ids'][max(0,k-3):k+3]}")
      if not same_text:
        import difflib
        print("  text diff:\n" + "\n".join(difflib.unified_diff(ref["rendered_prompt"].splitlines(), text.splitlines(), "meta", "ours", lineterm="", n=1)))
      raise SystemExit("prompt fixture mismatch -- fix the template/tokenizer before building anything else")
    print(f"  stop set in the model copy: {gcfg['eos_token_id']} (Meta: 151645 stop + native EOS 151643 both terminate)")

  names = [f"gsm8k_boxed_train.parquet", f"gsm8k_boxed_test{args.n_eval}.parquet", "gsm8k_boxed_test.parquet", "prompt_fixture.json"]
  with open(f"{args.out}/MANIFEST.sha256", "w") as f:
    for nme in names:
      f.write(f"{hashlib.sha256(open(f'{args.out}/{nme}', 'rb').read()).hexdigest()}  {nme}\n")
  print("wrote", names, "+ MANIFEST.sha256")


if __name__ == "__main__":
  main()