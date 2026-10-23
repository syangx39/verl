#!/usr/bin/env python3
"""Build the A2 training-data artifact: the same rows with the inner Gemma-3 turn template removed (PR #11 A2).

The reference training prompt (MaxText gsm8k_rl.json TEMPLATE, my_scripts/preprocess_openmathinstruct2.py) puts
Gemma-3 turn markers inside the user message, and verl then applies the Gemma-4 chat template on top:

    reference user content:  "<start_of_turn>user\\n{system_prompt}\\n\\n{question}<end_of_turn>\\n<start_of_turn>model"
    A2 user content:         "{system_prompt}\\n\\n{question}"

Only the inner markers and the fake model header are removed; instructions, question text, requested answer format,
row order, data_source, reward_model (ground truth) and extra_info (question, index) are unchanged. Every row is
checked against that exact form (the script aborts on any mismatch). The eval set is not touched.

The manifest records hashes, row checks, token counts of a deterministic sample rendered through the model's chat
template (reference vs A2), the decoded-text form verl writes to the rollout dumps, and a few before/after samples.

Usage:
    python3 gemma4_build_single_wrap_train.py --src train.parquet --out train_single_wrap.parquet \\
        --manifest train_single_wrap_manifest.json --model /workspace/meta-RL/models/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys

SYSTEM_PROMPT = ("You are given a problem. Think about the problem and provide your reasoning. Place it between "
                 "<reasoning> and </reasoning>. Then, provide the final answer (i.e., just one numerical value) between "
                 "<answer> and </answer>.")
INNER_PREFIX = "<start_of_turn>user\n"
INNER_SUFFIX = "<end_of_turn>\n<start_of_turn>model"


def reference_content(question: str) -> str:
    return f"{INNER_PREFIX}{SYSTEM_PROMPT}\n\n{question}{INNER_SUFFIX}"


def single_wrap_content(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{question}"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="tokenizer / chat template used for the token-count sample")
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    src = pq.read_table(args.src)
    n = src.num_rows
    prompts = src.column("prompt").to_pylist()
    extra = src.column("extra_info").to_pylist()
    new_prompts, bad = [], []
    for i, (msgs, ei) in enumerate(zip(prompts, extra)):
        q = ei.get("question") if ei else None
        ok = (isinstance(msgs, list) and len(msgs) == 1 and msgs[0].get("role") == "user" and isinstance(q, str)
              and msgs[0].get("content") == reference_content(q))
        if not ok:
            bad.append(i)
            continue
        new_prompts.append([{"role": "user", "content": single_wrap_content(q)}])
    if bad:
        print(f"ABORT: {len(bad)} rows do not have the exact reference form, e.g. rows {bad[:5]}", file=sys.stderr)
        return 1
    prompt_type = src.schema.field("prompt").type
    out = src.set_column(src.schema.get_field_index("prompt"), pa.field("prompt", prompt_type),
                         pa.array(new_prompts, type=prompt_type))
    out = out.replace_schema_metadata(src.schema.metadata)
    pq.write_table(out, args.out)

    # Re-read and check that everything except the prompt is identical, row by row.
    back = pq.read_table(args.out)
    checks = {"rows_src": n, "rows_out": back.num_rows, "rows_exact_reference_form": n - len(bad)}
    for col in ("data_source", "reward_model", "extra_info"):
        checks[f"{col}_identical"] = back.column(col).to_pylist() == src.column(col).to_pylist()
    back_prompts = back.column("prompt").to_pylist()
    checks["prompt_is_single_wrap_of_question"] = all(
        p[0]["content"] == single_wrap_content(ei["question"]) for p, ei in zip(back_prompts, extra))
    checks["no_inner_markers_left"] = not any(
        INNER_PREFIX.strip() in p[0]["content"] or "<end_of_turn>" in p[0]["content"] for p in back_prompts)
    # The Gemma-4 chat template trims the message content; with the inner markers gone, whitespace at the end of
    # a question is now at the end of the content and is trimmed when the prompt is rendered.
    checks["questions_with_trailing_whitespace"] = sum(1 for ei in extra if ei["question"] != ei["question"].rstrip())

    # Token counts and decoded forms on a deterministic sample, through the model's chat template.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    stride = max(1, n // args.sample)
    idx = list(range(0, n, stride))[: args.sample]
    ref_len, new_len, decoded_ok, samples, mismatches = [], [], 0, [], []

    def render(content):  # same path as the launcher preflight: chat template text, then token ids
        text = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        return text, tok(text, add_special_tokens=False)["input_ids"]

    for k, i in enumerate(idx):
        q = extra[i]["question"]
        text_ref, ids_ref = render(reference_content(q))
        text_new, ids_new = render(single_wrap_content(q))
        ref_len.append(len(ids_ref))
        new_len.append(len(ids_new))
        dec_ref = tok.decode(ids_ref, skip_special_tokens=True)
        dec_new = tok.decode(ids_new, skip_special_tokens=True)
        # verl's rollout dump stores this decoded form; gemma4_ab_compare.py --train-template-change relies on it.
        # .strip() = the Gemma-4 chat template's `message['content'] | trim`, which now reaches the question's end.
        pre, suf = "user\n" + INNER_PREFIX, INNER_SUFFIX + "\nmodel\n"
        expected_new = ("user\n" + dec_ref[len(pre):-len(suf)].strip() + "\nmodel\n"
                        if dec_ref.startswith(pre) and dec_ref.endswith(suf) else None)
        if expected_new == dec_new:
            decoded_ok += 1
        elif len(mismatches) < 10:
            mismatches.append({"row": i, "question_tail": q[-60:], "decoded_reference_tail": dec_ref[-90:],
                               "decoded_single_wrap_tail": dec_new[-90:],
                               "expected_tail": None if expected_new is None else expected_new[-90:]})
        if k < 3:
            samples.append({"row": i, "reference_rendered": text_ref, "single_wrap_rendered": text_new,
                            "reference_tokens": len(ids_ref), "single_wrap_tokens": len(ids_new)})
    diff = [a - b for a, b in zip(ref_len, new_len)]
    token_counts = {"sample_rows": len(idx), "stride": stride,
                    "reference": {"mean": statistics.mean(ref_len), "max": max(ref_len)},
                    "single_wrap": {"mean": statistics.mean(new_len), "max": max(new_len)},
                    "removed_tokens": {"min": min(diff), "max": max(diff), "mean": statistics.mean(diff)},
                    "decoded_form_matches_transform": f"{decoded_ok}/{len(idx)}", "decoded_form_mismatches": mismatches}

    manifest = {"source": args.src, "source_sha256": _sha256_file(args.src), "output": args.out,
                "output_sha256": _sha256_file(args.out), "transform": {
                    "reference_content": "<start_of_turn>user\\n{system_prompt}\\n\\n{question}<end_of_turn>\\n<start_of_turn>model",
                    "single_wrap_content": "{system_prompt}\\n\\n{question}", "system_prompt": SYSTEM_PROMPT},
                "checks": checks, "token_counts": token_counts, "samples": samples}
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    ok = (all(v for k, v in checks.items() if k.endswith(("identical", "single_wrap_of_question", "markers_left")))
          and checks["rows_out"] == n and decoded_ok == len(idx))
    print(json.dumps({"checks": checks, "token_counts": token_counts, "output_sha256": manifest["output_sha256"]}, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
