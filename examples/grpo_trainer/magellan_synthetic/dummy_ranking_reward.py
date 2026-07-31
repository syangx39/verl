#!/usr/bin/env python3
"""Dummy diversity ranking reward for the Magellan synthetic prefill workload.

Faithful copy of the customer's dummy reward, wrapped for verl's
custom_reward_function interface:
    fn(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float

Scoring (unchanged from the provided function):
    0.0  no integer list found in the output
    0.3  valid integers but wrong item count vs label_arr/fm_labels length
    0.3  identity permutation (1,2,...,n)  — lazy output
    0.2  duplicates present
    0.3 + 0.7 * normalized displacement from identity, otherwise

KNOWN LIMITATION (documented in run_meta_prefill_16k.sh): this reward is
RL-gameable (max displacement = fixed reversal); expect response distribution
drift after ~10-20 steps. For perf testing only.
"""

import re


def _extract_int_list(text):
  """Parse the ranking list from model output.

  Priority: (1) last line matching the comma-separated format the system
  prompt demands ("1,3,2,4"), (2) last whitespace-separated integer line,
  (3) all integers anywhere (lenient fallback). Returns list[int] or None.
  """
  if not text:
    return None
  MAX_DIGITS = 12  # ranking indices are small; anything longer is garbage
  def _safe_ints(strs):
    return [int(t) for t in strs if len(t.lstrip("-")) <= MAX_DIGITS]
  comma_candidates = []
  space_candidates = []
  for line in text.strip().splitlines():
    stripped = line.strip()
    if re.fullmatch(r"-?\d+(\s*,\s*-?\d+)+\s*,?", stripped):
      comma_candidates.append(_safe_ints(re.findall(r"-?\d+", stripped)))
      continue
    tokens = stripped.split()
    if tokens and all(re.fullmatch(r"-?\d+", t) for t in tokens):
      space_candidates.append(_safe_ints(tokens))
  if comma_candidates:
    return comma_candidates[-1]
  if space_candidates:
    return space_candidates[-1]
  nums = _safe_ints(re.findall(r"-?\d+", text))
  if len(nums) >= 2:
    return nums
  return None


def dummy_reward(data_source=None, solution_str=None, ground_truth=None,
                 extra_info=None, **kwargs):
  """verl entry point. Signature superset of the customer's version.

  Hardened: never raises — any parsing pathology scores 0.0. (A single
  garbage response must not kill the whole step; observed: 8000-digit
  number blob hit CPython's int-conversion digit limit and crashed a
  32-GPU run at step 5.)"""
  try:
    return _dummy_reward_inner(solution_str, extra_info or {})
  except Exception:
    return 0.0


def _dummy_reward_inner(solution_str, extra_info):
  pred_list = _extract_int_list(solution_str or "")
  if pred_list is None or len(pred_list) == 0:
    return 0.0

  label_arr = extra_info.get("label_arr")
  fm_labels = extra_info.get("fm_labels")
  expected_n = len(label_arr or fm_labels or [])

  if expected_n > 0 and len(pred_list) != expected_n:
    return 0.3

  n = len(pred_list)
  identity = list(range(1, n + 1))
  if pred_list == identity:
    return 0.3

  if len(set(pred_list)) != n:
    return 0.2

  displacement = sum(abs(pred_list[i] - identity[i]) for i in range(n))
  max_displacement = n * n // 2
  diversity_score = min(1.0, displacement / max(max_displacement, 1))

  return 0.3 + 0.7 * diversity_score


if __name__ == "__main__":
  cases = [
      ("3 1 2 5 4", {"label_arr": [0, 1, 0, 1, 1]}, "valid rerank"),
      ("1 2 3 4 5", {"label_arr": [0, 1, 0, 1, 1]}, "identity -> 0.3"),
      ("5 4 3 2 1", {"label_arr": [0, 1, 0, 1, 1]}, "reversal (max score)"),
      ("2 2 3 4 5", {"label_arr": [0, 1, 0, 1, 1]}, "duplicates -> 0.2"),
      ("1 2 3", {"label_arr": [0, 1, 0, 1, 1]}, "wrong count -> 0.3"),
      ("no numbers here", {"label_arr": [0, 1]}, "invalid -> 0.0"),
      ("The ranking is:\n4 2 1 5 3", {"fm_labels": [0.1] * 5}, "prose + list"),
  ]
  for out, extra, note in cases:
    print(f"{dummy_reward('x', out, '', extra):.3f}  {note}")
