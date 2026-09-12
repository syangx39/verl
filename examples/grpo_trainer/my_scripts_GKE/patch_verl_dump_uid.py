#!/usr/bin/env python3
"""Patch the verl fork so trainer.rollout_data_dir dumps carry the GRPO group uid.

verl assigns batch.non_tensor_batch["uid"] (one uuid per prompt, before the n=8
repeat) at the top of every training step, but _log_rollout_data() does not
write it. Grouping the dump by extra_info.index is not safe (indices from the
old train/val files collide), so add "uid" next to the existing "request_id".

Usage (on the head pod):
  python3 patch_verl_dump_uid.py /workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py
Idempotent. Prints exactly one of:
  DUMP-UID PATCH: APPLIED
  DUMP-UID PATCH: already patched
  DUMP-UID PATCH: FAILED <reason>
Record the resulting `git -C /workspace/meta-RL/verl diff` in the package.
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py"
src = open(path, encoding="utf-8").read()

MARK = "# _DUMP_UID"
if MARK in src:
  print("DUMP-UID PATCH: already patched")
  sys.exit(0)

ANCHOR = '''            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
'''
if src.count(ANCHOR) != 1:
  print(f"DUMP-UID PATCH: FAILED anchor found {src.count(ANCHOR)} times (expected 1) -- fork changed?")
  sys.exit(1)

INSERT = ANCHOR + '''            if "uid" in batch.non_tensor_batch:  # _DUMP_UID: GRPO group id (one uuid per prompt)
                reward_extra_infos_to_dump.setdefault(
                    "uid",
                    batch.non_tensor_batch["uid"].tolist(),
                )
'''
new = src.replace(ANCHOR, INSERT, 1)
try:
  ast.parse(new)
except SyntaxError as e:
  print(f"DUMP-UID PATCH: FAILED patched file does not parse: {e}")
  sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("DUMP-UID PATCH: APPLIED")
