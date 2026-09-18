#!/usr/bin/env python3
"""Meta `boxed_math` reward -- PROVISIONAL port for the GSM8K-boxed reproduction.

STATUS: the extraction / normalization rules below are our reading of Meta's config and README notes
("format_score: 0.1 -- partial credit for a well-formed \\boxed{} with a wrong value"; gold shipped
verbatim incl. thousands separators, "the reward function's own normalization is what decides whether
\\boxed{1080} matches gold 1,080"). They MUST be replaced by a verbatim port of Meta's boxed_math once
they send the source, and validated on a shared completion fixture. Every rule that is a guess is
marked GUESS.

Score (Meta semantics, weight 1.0):
    correct boxed answer                          -> 1.0
    well-formed \\boxed{...} with a wrong value    -> format_score (0.1)      [GUESS: "well-formed" = non-empty box]
    no \\boxed{}                                  -> 0.0
  + overlong soft penalty (DAPO):  buffer 512 tokens, cap 2048, penalty 1.0
        r += min(0, -(L - (2048-512)) / 512 * 1.0)   -- needs extra_info["response_len"] (fork patch)
verl interface: compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict
Returned keys: score, acc, fmt(=boxed present), format_credit, length_penalty, overlong, qid.
Knobs (env): REWARD_FORMAT_SCORE (0.1), REWARD_OVERLONG_BUFFER (512; 0=off), REWARD_OVERLONG_PENALTY (1.0),
             REWARD_MAX_RESP_LEN (2048). No math_verify, no sympy -> deterministic, thread-safe, no timeouts.
"""
import os
import re

_FORMAT_SCORE = float(os.environ.get("REWARD_FORMAT_SCORE", "0.1"))
_OVERLONG_BUFFER = int(os.environ.get("REWARD_OVERLONG_BUFFER", "512"))
_OVERLONG_PENALTY = float(os.environ.get("REWARD_OVERLONG_PENALTY", "1.0"))
_MAX_RESP_LEN = int(os.environ.get("REWARD_MAX_RESP_LEN", "2048"))


def last_boxed(text: str):
  """Content of the LAST \\boxed{...} with balanced braces; None if absent.  [GUESS: last, not first]"""
  idx = text.rfind("\\boxed{")                 # strict: the literal token \boxed{ (GUESS: no whitespace, no \boxed[...]{ } variants)
  if idx < 0:
    return None
  i = idx + len("\\boxed")
  depth = 0
  for j in range(i, len(text)):
    if text[j] == "{":
      depth += 1
    elif text[j] == "}":
      depth -= 1
      if depth == 0:
        return text[i + 1:j]
  return None                                   # unbalanced -> treat as not well-formed


def normalize(s: str) -> str:
  """GUESS at Meta's normalization: strip, drop $ and thousands separators, \\text{} wrappers,
  trailing period, surrounding whitespace; keep sign and decimal point."""
  s = s.strip()
  s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
  s = s.replace("\\$", "").replace("$", "").replace(",", "").replace(" ", "")   # \$ (escaped) and bare $
  s = s.rstrip(".")
  s = re.sub(r"^\\?\((.*)\\?\)$", r"\1", s)
  return s


def numeric_equal(a: str, b: str) -> bool:
  try:
    return abs(float(a) - float(b)) < 1e-6
  except ValueError:
    return False


def _overlong(extra_info):
  if _OVERLONG_BUFFER == 0:
    return 0.0, 0.0
  n = extra_info.get("response_len") if isinstance(extra_info, dict) else None
  if n is None:
    raise RuntimeError("REWARD_OVERLONG_BUFFER>0 but extra_info has no 'response_len' (apply patch_verl_reward_response_len.py)")
  pen = min(0.0, -(float(n) - (_MAX_RESP_LEN - _OVERLONG_BUFFER)) / _OVERLONG_BUFFER * _OVERLONG_PENALTY)
  return pen, (1.0 if pen < 0 else 0.0)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
  boxed = last_boxed(solution_str)
  gold = str(ground_truth)
  if boxed is None or boxed.strip() == "":
    acc, credit, fmt = 0.0, 0.0, 0.0
  else:
    fmt = 1.0
    ng, ga = normalize(boxed), normalize(gold)
    acc = 1.0 if (ng == ga or numeric_equal(ng, ga)) else 0.0
    credit = 0.0 if acc else _FORMAT_SCORE
  pen, overlong = _overlong(extra_info)
  qid = float(extra_info.get("index", -1)) if isinstance(extra_info, dict) else -1.0
  return {"score": acc + credit + pen, "acc": acc, "fmt": fmt, "format_credit": credit,
          "length_penalty": pen, "overlong": overlong, "qid": qid}


if __name__ == "__main__":
  cases = [
      ("... so the answer is \\boxed{72}.", "72", 1.0, 1.0, "exact"),
      ("\\boxed{1080}", "1,080", 1.0, 1.0, "thousands separator in gold (GUESS: dropped)"),
      ("\\boxed{1,080}", "1080", 1.0, 1.0, "thousands separator in answer"),
      ("\\boxed{\\$18}", "18", 1.0, 1.0, "dollar sign"),
      ("\\boxed{18.0}", "18", 1.0, 1.0, "numeric equality (GUESS)"),
      ("\\boxed{7}", "72", 0.1, 0.0, "boxed but wrong -> format_score"),
      ("the answer is 72", "72", 0.0, 0.0, "no box -> 0 even if correct"),
      ("\\boxed{}", "72", 0.0, 0.0, "empty box -> not well-formed (GUESS)"),
      ("\\boxed{5} ... \\boxed{72}", "72", 1.0, 1.0, "last box wins (GUESS)"),
      ("\\boxed{\\text{72}}", "72", 1.0, 1.0, "\\text wrapper (GUESS)"),
      ("\\boxedgarbage{72}", "72", 0.0, 0.0, "not the literal \\boxed{ -> no box"),
      ("\\boxed {72}", "72", 0.0, 0.0, "space before brace -> no box (GUESS: strict)"),
  ]
  ok_all = True
  for comp, gt, exp_score, exp_acc, note in cases:
    o = compute_score("gsm8k", comp, gt, extra_info={"index": 0, "response_len": 100})
    ok = abs(o["score"] - exp_score) < 1e-9 and o["acc"] == exp_acc
    ok_all &= ok
    print(f"{'OK ' if ok else 'FAIL'} {note}: score={o['score']} acc={o['acc']} fmt={o['fmt']}")
  for n, exp in ((1536, 0.0), (1792, -0.5), (2048, -1.0)):
    o = compute_score("gsm8k", "\\boxed{72}", "72", extra_info={"index": 0, "response_len": n})
    ok = abs(o["length_penalty"] - exp) < 1e-9 and abs(o["score"] - (1.0 + exp)) < 1e-9
    ok_all &= ok
    print(f"{'OK ' if ok else 'FAIL'} length {n}: penalty={o['length_penalty']:+.2f} score={o['score']:+.2f}")
  print("knobs:", dict(format_score=_FORMAT_SCORE, buffer=_OVERLONG_BUFFER, penalty=_OVERLONG_PENALTY, cap=_MAX_RESP_LEN))
  print("RESULT:", "PASS" if ok_all else "FAIL", "-- PROVISIONAL rules; replace with Meta's verbatim boxed_math")
  raise SystemExit(0 if ok_all else 1)