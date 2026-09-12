#!/usr/bin/env python3
"""Patch the verl fork's reward-loop NaiveRewardManager to pass the response
token length to the custom reward function.

The reward-loop path (verl/experimental/reward_loop/reward_manager/naive.py)
computes `valid_response_length` but does not hand it to compute_score(), so a
length-aware reward (DAPO overlong soft penalty) cannot be implemented in the
custom reward. This adds one key to extra_info:

    extra_info["response_len"] = int(valid_response_length)   # _RESP_LEN

Idempotent. Prints exactly one of:
  RESP-LEN PATCH: APPLIED | already patched | FAILED <reason>
Usage:
  python3 patch_verl_reward_response_len.py \
      /workspace/meta-RL/verl/verl/experimental/reward_loop/reward_manager/naive.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/meta-RL/verl/verl/experimental/reward_loop/reward_manager/naive.py"
src = open(path, encoding="utf-8").read()
MARK = "# _RESP_LEN"
if MARK in src:
  print("RESP-LEN PATCH: already patched")
  sys.exit(0)
ANCHOR = '''        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
'''
if src.count(ANCHOR) != 1:
  print(f"RESP-LEN PATCH: FAILED anchor found {src.count(ANCHOR)} times (expected 1)")
  sys.exit(1)
INSERT = ANCHOR + '''        extra_info["response_len"] = int(valid_response_length)  # _RESP_LEN: for length-aware rewards
'''
new = src.replace(ANCHOR, INSERT, 1)
try:
  ast.parse(new)
except SyntaxError as e:
  print(f"RESP-LEN PATCH: FAILED patched file does not parse: {e}")
  sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("RESP-LEN PATCH: APPLIED")
