#!/usr/bin/env python3
"""Meta `boxed_math` reward + overlong penalty, ported from REPRODUCTION.md v1.0 (2026-09-18).

Scoring is a three-way EXCLUSIVE branch (a correct answer scores 1.0, not 1.1):
    pred = last \\boxed{...} (rfind of the literal "\\boxed{", then forward brace matching; unbalanced -> None)
    if pred is None or pred == "":      reward = 0.0      # no parseable box (truncation lands here)
    elif normalize(pred) == gold_norm:  reward = 1.0      # score
    else:                               reward = 0.1      # format_score
    normalize(v) = v.strip().rstrip(".").replace(",", "").replace("$", "").strip()   -- deliberately minimal:
        no decimal/integer equivalence (18.0 vs 18 -> 0.1), no \\text{} unwrapping, no LaTeX parsing.
    gold: REPRODUCTION.md says the gold is NOT normalized, but its own fixture scores \\boxed{1080} vs gold "1,080"
    as 1.0, which is only possible if the gold's thousands separator is also removed. We follow the FIXTURES
    (gold gets the same minimal normalize) and have asked Meta to resolve the contradiction.  [PENDING]
Overlong penalty (TRAINING ONLY -- Meta evaluates with the raw reward):
    penalty = min(0, -(completion_len - (2048 - 512)) / 512 * 1.0)      # added to the reward; can go below -1 in fixtures
verl interface: compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict
    score          training reward (raw + penalty) for sources listed in REWARD_PENALTY_SOURCES; raw reward otherwise (eval)
    reward_raw     boxed_math reward without penalty (== Meta eval/mean_reward semantics)
    acc            1 if reward_raw == 1.0 (Meta eval/accuracy semantics), fmt: 1 if a box was parsed
    format_credit, length_penalty, overlong, qid
Knobs (env): REWARD_FORMAT_SCORE 0.1, REWARD_OVERLONG_BUFFER 512 (0 = off), REWARD_OVERLONG_PENALTY 1.0,
             REWARD_MAX_RESP_LEN 2048, REWARD_PENALTY_SOURCES "gsm8k_boxed_train" (comma list).
Self-test: `python3 boxed_math_reward.py [--fixtures reference/reward_fixtures.json]` runs Meta's 15 reward + 6
overlong fixtures (embedded copy by default); non-zero exit on any mismatch.
"""
import json
import os
import sys

_FORMAT_SCORE = float(os.environ.get("REWARD_FORMAT_SCORE", "0.1"))
_SCORE = 1.0
_OVERLONG_BUFFER = int(os.environ.get("REWARD_OVERLONG_BUFFER", "512"))
_OVERLONG_PENALTY = float(os.environ.get("REWARD_OVERLONG_PENALTY", "1.0"))
_MAX_RESP_LEN = int(os.environ.get("REWARD_MAX_RESP_LEN", "2048"))
_PENALTY_SOURCES = {s.strip() for s in os.environ.get("REWARD_PENALTY_SOURCES", "gsm8k_boxed_train").split(",") if s.strip()}


def extract_boxed(text: str):
  """Content of the LAST \\boxed{...}; None if absent or if the braces never balance (truncated)."""
  idx = text.rfind("\\boxed{")
  if idx < 0:
    return None
  i = idx + len("\\boxed")            # position of '{'
  depth = 0
  for j in range(i, len(text)):
    if text[j] == "{":
      depth += 1
    elif text[j] == "}":
      depth -= 1
      if depth == 0:
        return text[i + 1:j]
  return None


def normalize(v: str) -> str:
  return v.strip().rstrip(".").replace(",", "").replace("$", "").strip()


def boxed_math(response: str, gold: str):
  """Returns (reward, acc, fmt, format_credit)."""
  pred = extract_boxed(response)
  if pred is None or pred == "":
    return 0.0, 0.0, 0.0, 0.0
  if normalize(pred) == normalize(str(gold)):      # gold normalized per the fixtures (see docstring, PENDING)
    return _SCORE, 1.0, 1.0, 0.0
  return _FORMAT_SCORE, 0.0, 1.0, _FORMAT_SCORE


def overlong_penalty(completion_len: int) -> float:
  if _OVERLONG_BUFFER == 0:
    return 0.0
  return min(0.0, -(float(completion_len) - (_MAX_RESP_LEN - _OVERLONG_BUFFER)) / _OVERLONG_BUFFER * _OVERLONG_PENALTY)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
  reward, acc, fmt, credit = boxed_math(solution_str, ground_truth)
  pen = 0.0
  if data_source in _PENALTY_SOURCES and _OVERLONG_BUFFER > 0:
    n = extra_info.get("response_len") if isinstance(extra_info, dict) else None
    if n is None:
      raise RuntimeError("overlong penalty enabled but extra_info has no 'response_len' (apply patch_verl_reward_response_len.py)")
    pen = overlong_penalty(int(n))
  qid = float(extra_info.get("index", -1)) if isinstance(extra_info, dict) else -1.0
  return {"score": reward + pen, "reward_raw": reward, "acc": acc, "fmt": fmt, "format_credit": credit,
          "length_penalty": pen, "overlong": 1.0 if pen < 0 else 0.0, "qid": qid}


# ---- Meta's fixtures (reference/reward_fixtures.json, copied verbatim; --fixtures overrides with the file)
META_FIXTURES = {
    "reward_fixtures": [
        {"case": "exact match", "response": "The answer is \\boxed{18}", "gold": "18", "reward": 1.0},
        {"case": "gold has thousands separator", "response": "So \\boxed{1080}", "gold": "1,080", "reward": 1.0},
        {"case": "prediction has separator", "response": "So \\boxed{1,080}", "gold": "1080", "reward": 1.0},
        {"case": "dollar sign in prediction", "response": "\\boxed{$42}", "gold": "42", "reward": 1.0},
        {"case": "trailing period", "response": "\\boxed{18.}", "gold": "18", "reward": 1.0},
        {"case": "surrounding whitespace", "response": "\\boxed{  18  }", "gold": "18", "reward": 1.0},
        {"case": "nested braces", "response": "\\boxed{\\frac{1}{2}}", "gold": "\\frac{1}{2}", "reward": 1.0},
        {"case": "multiple boxed -> LAST wins", "response": "first \\boxed{7} then \\boxed{18}", "gold": "18", "reward": 1.0},
        {"case": "multiple boxed, last is wrong", "response": "first \\boxed{18} then \\boxed{7}", "gold": "18", "reward": 0.1},
        {"case": "well-formed but wrong value", "response": "\\boxed{17}", "gold": "18", "reward": 0.1},
        {"case": "no boxed at all", "response": "The answer is 18.", "gold": "18", "reward": 0.0},
        {"case": "TRUNCATED: unclosed brace", "response": "... so the answer is \\boxed{18", "gold": "18", "reward": 0.0},
        {"case": "TRUNCATED: unclosed nested", "response": "\\boxed{\\frac{1}{2", "gold": "\\frac{1}{2}", "reward": 0.0},
        {"case": "empty boxed", "response": "\\boxed{}", "gold": "18", "reward": 0.0},
        {"case": "decimal vs integer gold", "response": "\\boxed{18.0}", "gold": "18", "reward": 0.1},
    ],
    "overlong_fixtures": [
        {"completion_tokens": 100, "penalty": 0.0}, {"completion_tokens": 1536, "penalty": 0.0},
        {"completion_tokens": 1537, "penalty": -0.001953}, {"completion_tokens": 1792, "penalty": -0.5},
        {"completion_tokens": 2048, "penalty": -1.0}, {"completion_tokens": 2560, "penalty": -2.0},
    ],
}


def run_fixtures(fx):
  fails = []
  for c in fx["reward_fixtures"]:
    r, *_ = boxed_math(c["response"], c["gold"])
    ok = abs(r - c["reward"]) < 1e-9
    print(f"{'OK ' if ok else 'FAIL'} {c['case']:<36} -> {r} (expected {c['reward']})")
    fails += [] if ok else [c["case"]]
  for c in fx["overlong_fixtures"]:
    p = overlong_penalty(c["completion_tokens"])
    ok = abs(p - c["penalty"]) < 1e-5
    print(f"{'OK ' if ok else 'FAIL'} overlong {c['completion_tokens']:>5} tokens -> {p:+.6f} (expected {c['penalty']:+})")
    fails += [] if ok else [f"overlong {c['completion_tokens']}"]
  # eval vs train semantics
  tr = compute_score("gsm8k_boxed_train", "\\boxed{18}", "18", extra_info={"index": 0, "response_len": 2048})
  ev = compute_score("gsm8k_boxed_test512", "\\boxed{18}", "18", extra_info={"index": 0, "response_len": 2048})
  ok = tr["score"] == 0.0 and tr["reward_raw"] == 1.0 and ev["score"] == 1.0 and ev["length_penalty"] == 0.0 and ev["acc"] == 1.0
  print(f"{'OK ' if ok else 'FAIL'} penalty applies to train source only: train score {tr['score']} (raw 1.0, pen -1.0); eval score {ev['score']} (no penalty)")
  fails += [] if ok else ["penalty scope"]
  return fails


if __name__ == "__main__":
  fx = META_FIXTURES
  if len(sys.argv) > 2 and sys.argv[1] == "--fixtures":
    fx = json.load(open(sys.argv[2]))
    print(f"using fixtures from {sys.argv[2]}")
  fails = run_fixtures(fx)
  print("knobs:", dict(format_score=_FORMAT_SCORE, buffer=_OVERLONG_BUFFER, penalty=_OVERLONG_PENALTY, cap=_MAX_RESP_LEN, penalty_sources=sorted(_PENALTY_SOURCES)))
  print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
  raise SystemExit(0 if not fails else 1)
