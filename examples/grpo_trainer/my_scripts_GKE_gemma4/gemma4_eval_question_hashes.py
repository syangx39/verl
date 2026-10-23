#!/usr/bin/env python3
"""Build the held-out question-hash manifest used to scope reward knobs to training rows (PR #11 A1).

maxtext_math_reward.py cannot tell training rows from validation rows by its arguments: both carry the same
data_source and extra_info fields, and verl's `validate` flag does not reach compute_score. With
REWARD_FMT_WEIGHT_TRAIN set, the scorer keeps the canonical format coefficient for rows whose
extra_info["question"] hashes into this manifest (the frozen eval set) and applies the training coefficient to
every other row.

This script writes {source, source_sha256, n, hash, question_sha256: [...]} for the eval parquet and, with
--train, checks that no training question hashes into the eval set (exit 1 otherwise).

Usage:
    python3 gemma4_eval_question_hashes.py --eval eval_5000.parquet --out eval_5k/eval_5000_question_sha256.json \\
        [--train train.parquet]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _question_hashes(path: str) -> list[str]:
    import pyarrow.parquet as pq

    out = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["extra_info"], batch_size=65536):
        for r in batch.column("extra_info").to_pylist():
            q = r.get("question") if r else None
            if not isinstance(q, str):
                raise SystemExit(f"{path}: row without extra_info.question")
            out.append(hashlib.sha256(q.encode("utf-8")).hexdigest())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="frozen eval parquet (eval_5000.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="", help="training parquet to check for overlap")
    args = ap.parse_args()

    ev = _question_hashes(args.eval)
    uniq = sorted(set(ev))
    if len(uniq) != len(ev):
        raise SystemExit(f"{args.eval}: {len(ev) - len(uniq)} duplicate questions")
    manifest = {"source": os.path.basename(args.eval), "source_sha256": _sha256_file(args.eval), "n": len(uniq),
                "hash": "sha256 of the UTF-8 extra_info.question string", "question_sha256": uniq}
    if args.train:
        tr = _question_hashes(args.train)
        overlap = set(tr) & set(uniq)
        manifest["train_check"] = {"source": os.path.basename(args.train), "source_sha256": _sha256_file(args.train),
                                   "train_rows": len(tr), "overlapping_eval_questions": len(overlap)}
        print(f"train rows {len(tr)}, eval questions {len(uniq)}, overlap {len(overlap)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=0)
        f.write("\n")
    print(f"wrote {args.out}: n={len(uniq)} source_sha256={manifest['source_sha256']}")
    return 1 if manifest.get("train_check", {}).get("overlapping_eval_questions") else 0


if __name__ == "__main__":
    sys.exit(main())
