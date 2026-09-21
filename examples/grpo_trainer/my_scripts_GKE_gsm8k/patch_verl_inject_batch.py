#!/usr/bin/env python3
"""Patch the verl fork to INJECT a fixed rollout batch at one training step (env-gated, off by default).

Purpose: compare two trainer paths (e.g. single-forward bypass+reinforce vs two-pass) on IDENTICAL tokens inside
the native trainer, so their step-1 gradients (exp_avg/(1-beta1)) and post-update weights can be compared directly.

When INJECT_BATCH_NPZ is set, at global step INJECT_BATCH_STEP (default 1), right after the generated output is
merged into the batch (and before response_mask / balance_batch), the trainer replaces, per prompt, the 16 generated
responses with the 16 rows of the dump that carry the same question index (extra_info.index == dump nt__qid):
    responses, attention_mask, position_ids, response_mask, rollout_log_probs, input_ids (= prompts ++ responses)
Prompts are asserted equal token-for-token (same data.seed -> same prompts at that step), uids stay those of the
current batch, and rewards are recomputed by the trainer's own reward function on the injected responses.
Requirements: the dump must come from a run with the same data.seed, batch size, rollout.n, prompt/response lengths
(patch_verl_logprob_fixture.py produces it); every current prompt must have exactly rollout.n dump rows.

Idempotent. Prints: INJECT-BATCH PATCH: APPLIED | already patched | FAILED <reason>
Usage:
  python3 patch_verl_inject_batch.py /workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py"
src = open(path, encoding="utf-8").read()
MARK = "_INJECT_BATCH"
if MARK in src:
  print("INJECT-BATCH PATCH: already patched")
  sys.exit(0)

CALL_ANCHOR = '''                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
'''
METHOD_ANCHOR = '''    def _update_actor(self, batch: DataProto) -> DataProto:
'''
for name, anchor in (("call", CALL_ANCHOR), ("method", METHOD_ANCHOR)):
  if src.count(anchor) != 1:
    print(f"INJECT-BATCH PATCH: FAILED {name} anchor found {src.count(anchor)} times (expected 1)")
    sys.exit(1)

CALL_INSERT = '''                    batch = batch.union(gen_batch_output)

                    # _INJECT_BATCH: env-gated replacement of the generated responses by a fixed dump (see _inject_batch)
                    import os as _os
                    if _os.environ.get("INJECT_BATCH_NPZ") and self.global_steps == int(_os.environ.get("INJECT_BATCH_STEP", "1")):
                        batch = self._inject_batch(batch)

                    if "response_mask" not in batch.batch.keys():
'''
METHOD_INSERT = '''    def _inject_batch(self, batch: DataProto) -> DataProto:  # _INJECT_BATCH
        """Replace generated responses with the rows of INJECT_BATCH_NPZ that carry the same question index."""
        import os
        import numpy as np
        import torch

        z = np.load(os.environ["INJECT_BATCH_NPZ"], allow_pickle=False)
        need = ["prompts", "responses", "attention_mask", "position_ids", "response_mask", "rollout_log_probs", "nt__qid"]
        missing = [k for k in need if k not in z.files]
        if missing:
            raise RuntimeError(f"_INJECT_BATCH: dump lacks {missing}")
        n_rep = int(self.config.actor_rollout_ref.rollout.n)
        cur_qid = np.array([int(e["index"]) for e in batch.non_tensor_batch["extra_info"]])
        dump_qid = z["nt__qid"].astype(np.int64)
        by_q = {}
        for i, q in enumerate(dump_qid):
            by_q.setdefault(int(q), []).append(i)
        cur_q_set, dump_q_set = set(cur_qid.tolist()), set(by_q)
        if cur_q_set != dump_q_set:
            raise RuntimeError(f"_INJECT_BATCH: question sets differ (cur-only {len(cur_q_set - dump_q_set)}, dump-only {len(dump_q_set - cur_q_set)}); use the same data.seed / step")
        bad = {q: len(v) for q, v in by_q.items() if len(v) != n_rep}
        if bad:
            raise RuntimeError(f"_INJECT_BATCH: {len(bad)} questions do not have {n_rep} dump rows: {list(bad.items())[:3]}")
        # order: for each current row, take the next unused dump row of the same question
        used = {q: 0 for q in by_q}
        src_idx = np.empty(len(cur_qid), dtype=np.int64)
        for i, q in enumerate(cur_qid.tolist()):
            src_idx[i] = by_q[q][used[q]]; used[q] += 1
        dev = batch.batch["responses"].device
        P_cur = batch.batch["prompts"]
        P_dump = torch.as_tensor(z["prompts"][src_idx], device=dev, dtype=P_cur.dtype)
        if P_cur.shape != P_dump.shape or not torch.equal(P_cur, P_dump):
            n_diff = int((P_cur != P_dump).any(dim=1).sum()) if P_cur.shape == P_dump.shape else -1
            raise RuntimeError(f"_INJECT_BATCH: prompt tokens differ from the dump (rows differing: {n_diff}; shapes {tuple(P_cur.shape)} vs {tuple(P_dump.shape)})")
        for k in ("responses", "attention_mask", "position_ids", "response_mask", "rollout_log_probs"):
            t = torch.as_tensor(z[k][src_idx], device=dev, dtype=batch.batch[k].dtype)
            if t.shape != batch.batch[k].shape:
                raise RuntimeError(f"_INJECT_BATCH: shape mismatch for {k}: dump {tuple(t.shape)} vs batch {tuple(batch.batch[k].shape)}")
            batch.batch[k] = t
        batch.batch["input_ids"] = torch.cat([batch.batch["prompts"], batch.batch["responses"]], dim=1)
        n_tok = int(batch.batch["response_mask"].sum())
        print(f"[_INJECT_BATCH] step {self.global_steps}: replaced {len(cur_qid)} rows ({len(by_q)} questions x {n_rep}) from {os.environ['INJECT_BATCH_NPZ']}; "
              f"prompts identical; injected completion tokens {n_tok}")
        return batch

'''
new = src.replace(CALL_ANCHOR, CALL_INSERT, 1).replace(METHOD_ANCHOR, METHOD_INSERT + METHOD_ANCHOR, 1)
try:
  ast.parse(new)
except SyntaxError as e:
  print(f"INJECT-BATCH PATCH: FAILED patched file does not parse: {e}")
  sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("INJECT-BATCH PATCH: APPLIED")
