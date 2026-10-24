#!/usr/bin/env python3
"""GSM8K boxed scorer v2 (`boxed_v2`): Meta `boxed_math` (v1) + ONE explicit, model-agnostic rule for decorated numbers.

v1 is boxed_math_reward.py in this directory. It is not modified; it is loaded from its file and its sha256 is
checked at import, so the v1 parts below are byte-for-byte the frozen Meta port the Qwen GSM8K 8K runs trained with.

What changes vs v1: only the "is the boxed answer correct" test. Extraction, the three reward levels, the format
score, the overlong penalty and its scope are v1's code and read the same environment knobs (REWARD_FORMAT_SCORE,
REWARD_OVERLONG_BUFFER, REWARD_OVERLONG_PENALTY, REWARD_MAX_RESP_LEN, REWARD_PENALTY_SOURCES).

    pred = v1.extract_boxed(response)       LAST \\boxed{...}; braces never balance (truncated) -> None
    pred is None or pred == ""              -> 0.0                          (v1)
    v1 correct                              -> 1.0                          (v1; v2 never removes credit)
    v1 parseable-wrong but number_equal     -> 1.0                          (the only v2 change)
    otherwise                               -> REWARD_FORMAT_SCORE (0.1)    (v1)

    v1 correct:   v1.normalize(pred) == v1.normalize(gold)   (strip, trailing '.', every ',' and '$' removed)
    number_equal: number(pred) and number(gold) both exist and have the same exact value

number(s) is a whitelist. Each step runs at most once, in this order. If what is left after step 5 is not a plain
decimal literal, the answer is "not a number" and only v1 can credit it. Contents longer than 200 characters are
never numbers.
  0. spacing  `{,}` -> ','; `\\,` between a digit and exactly three more digits -> ',' (both LaTeX thousands
              separators; step 5 still checks the grouping); the LaTeX spacing commands `\\,` `\\;` `\\:` `\\!` `\\ `
              `\\quad` `\\qquad` and `~` -> one space; strip; trailing '.' removed
  1. wrapper  if the WHOLE answer is a single \\text / \\textrm / \\textit / \\textbf / \\mathrm / \\mathbf / \\mbox
              group, use its content (e.g. \\text{18})
  2. unit     drop ONE trailing unit phrase, written as a group from the step-1 macro family or as bare words, with
              an optional ^2 / ^3 after it. A unit phrase is 1-4 words of ASCII letters. Inside a word only / . ' -
              may appear (Mb/hr, o'clock, sq.). No digits or other symbols are allowed, and no word may be in
              BLOCKED_WORDS: connectives, approximations, bounds, operations and multipliers, i.e. words that could
              change the value ("and", "about", "than", "squared", "thousand", "k"). A group that is not a unit
              phrase is NOT dropped, e.g. "\\text{ and } 20" or "\\text{ (approx 18)}".
              Unit words are not checked against the question: "18 hours" for gold 18 minutes is accepted.
  3. suffix   drop ONE trailing \\% / % / ^\\circ / ^{\\circ} / U+00B0 (degree sign)
  4. prefix   an optional sign (+ or -), then ONE currency symbol: \\$ / $ / U+20AC (euro) / U+00A3 (pound). A sign
              written right after the symbol is also read.
  5. literal  the rest must be digits, optionally grouped in thousands (1,234,567, groups of exactly three), with
              an optional fractional part (.5). Fractions, expressions, exponents, scientific notation, times,
              ranges, lists and words are not numbers.
  value       exact decimal value (decimal.Decimal). 28.00 == 28 and 18.0 == 18; no tolerance, so 18.5 != 18.

Intentional divergence from Meta's fixtures (PR #14 option B): "decimal vs integer gold", \\boxed{18.0} vs 18, is 1.0
here (Meta / v1: 0.1). Every other Meta fixture scores as in v1.
Inherited from v1, because a v1-correct answer stays correct: v1 deletes every ',' and '$' anywhere, so
\\boxed{1,00} vs 100 is correct under both versions even though step 5 alone would reject that grouping.

compute_score(...) returns v1's fields with v2 values (score, reward_raw, acc, fmt, format_credit, length_penalty,
overlong, qid) plus diagnostics. acc_v1 is the v1 (Meta) accuracy of the same response. v2_only = acc - acc_v1,
i.e. credited only by the number rule. scorer_version = 2.0. All values are floats because verl averages every
key.

Self-test: `python3 boxed_math_reward_v2.py` runs Meta's fixtures (with the divergence above), the overlong
fixtures, the penalty scope and RULE_CASES; non-zero exit on any mismatch. Unit tests: test_boxed_math_reward_v2.py.
"""
import decimal
import hashlib
import importlib.util
import json
import os
import re
import sys

SCORER_VERSION = 2.0
V1_SHA256 = "a743cf71e2094b0ebee67506460028b5a27bb9b1d86928f6c19b152561c7015b"   # boxed_math_reward.py (frozen)
V1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boxed_math_reward.py")


def _load_v1():
  with open(V1_PATH, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
  if sha != V1_SHA256:
    raise RuntimeError(f"{V1_PATH}: sha256 {sha} != frozen v1 {V1_SHA256}")
  spec = importlib.util.spec_from_file_location("boxed_math_reward_v1", V1_PATH)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


v1 = _load_v1()

TEXT_MACROS = ("\\text", "\\textrm", "\\textit", "\\textbf", "\\mathrm", "\\mathbf", "\\mbox")
BLOCKED_WORDS = frozenset("""
    and or nor not no plus minus
    approximately approx about around nearly almost roughly circa estimated
    than least most
    half twice thrice squared cubed
    hundred hundreds thousand thousands million millions billion billions trillion trillions dozen dozens
    k bn mn mil mln
""".split())                                           # compared lower-cased
BLOCKED_WORDS_CASED = frozenset({"M", "B", "MM"})      # million / billion abbreviations (lower-case mm = millimetres)
SUFFIXES = (("\\%", "percent"), ("%", "percent"), ("^{\\circ}", "degree"), ("^\\circ", "degree"), ("\u00b0", "degree"))
CURRENCIES = ("\\$", "$", "\u20ac", "\u00a3")
MAX_NUMBER_CHARS = 200   # longer boxed contents are never numbers (also bounds the regex work per answer)

_THIN_GROUP = re.compile(r"(?<=\d)\\,(?=\d{3}(?!\d))")
_SPACING = re.compile(r"\\(?:qquad|quad|[,;:! ])|~")
_WORD = r"[A-Za-z]+(?:[/.'\-][A-Za-z]+)*\.?"
_UNIT_PHRASE = re.compile(rf"{_WORD}(?: {_WORD}){{0,3}}")
_BARE_UNIT = re.compile(rf"(?P<head>.*?)(?:(?<=[0-9%\u00b0}}])|\s+)(?P<unit>{_WORD}(?:\s+{_WORD}){{0,3}})", re.S)
_UNIT_EXP = re.compile(r"\^(?:[23]|\{[23]\})$")
_LITERAL = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _matching_close(s, open_idx):
  depth = 0
  for j in range(open_idx, len(s)):
    if s[j] == "{":
      depth += 1
    elif s[j] == "}":
      depth -= 1
      if depth == 0:
        return j
  return -1


def _matching_open(s, close_idx):
  depth = 0
  for j in range(close_idx, -1, -1):
    if s[j] == "}":
      depth += 1
    elif s[j] == "{":
      depth -= 1
      if depth == 0:
        return j
  return -1


def is_unit_phrase(phrase):
  phrase = " ".join(phrase.split())
  if not _UNIT_PHRASE.fullmatch(phrase):
    return False
  words = [w.rstrip(".") for w in phrase.split(" ")]
  return not any(w.lower() in BLOCKED_WORDS or w in BLOCKED_WORDS_CASED for w in words)


def _unwrap(s):
  for mac in TEXT_MACROS:
    if s.startswith(mac + "{") and _matching_close(s, len(mac)) == len(s) - 1:
      return s[len(mac) + 1:-1]
  return s


def _strip_unit(s):
  """Step 2: drop ONE trailing unit phrase. Returns (rest, dropped)."""
  m = _UNIT_EXP.search(s)
  body = s[:m.start()].rstrip() if m else s      # ^2 / ^3 is dropped only together with a unit
  if body.endswith("}"):
    k = _matching_open(body, len(body) - 1)
    for mac in TEXT_MACROS:
      if k >= len(mac) and body[k - len(mac):k] == mac:
        if is_unit_phrase(body[k + 1:-1]):
          return body[:k - len(mac)].rstrip(), True
        break
    return s, False
  mb = _BARE_UNIT.fullmatch(body)
  if mb and is_unit_phrase(mb.group("unit")):
    return mb.group("head").rstrip(), True
  return s, False


def parse_number(s):
  """(exact value as decimal.Decimal or None, names of the whitelist steps used). See the module docstring."""
  used = []
  s = str(s)
  if len(s) > MAX_NUMBER_CHARS:
    return None, used
  s = _SPACING.sub(" ", _THIN_GROUP.sub(",", s.replace("{,}", ",")))
  s = s.strip().rstrip(".").strip()
  t = _unwrap(s)
  if t != s:
    used.append("wrapper")
    s = t.strip().rstrip(".").strip()
  s, dropped = _strip_unit(s)
  if dropped:
    used.append("unit")
  for suf, name in SUFFIXES:
    if s.endswith(suf):
      s = s[:-len(suf)].rstrip()
      used.append(name)
      break
  sign = ""
  if s[:1] in ("+", "-"):
    sign, s = s[0], s[1:].lstrip()
  for cur in CURRENCIES:
    if s.startswith(cur):
      s = s[len(cur):].lstrip()
      used.append("currency")
      if not sign and s[:1] in ("+", "-"):
        sign, s = s[0], s[1:].lstrip()
      break
  if not _LITERAL.fullmatch(s):
    return None, used
  if "," in s:
    used.append("grouping")
  return decimal.Decimal(sign + s.replace(",", "")), used


def number(s):
  return parse_number(s)[0]


def number_equal(pred, gold):
  a, b = number(pred), number(gold)
  return a is not None and b is not None and a == b


def boxed_math_v2(response, gold):
  """(reward, acc, fmt, format_credit, acc_v1) for one response; v1's boxed_math plus the number rule."""
  reward, acc_v1, fmt, credit = v1.boxed_math(response, gold)
  if fmt == 1.0 and acc_v1 == 0.0 and number_equal(v1.extract_boxed(response), str(gold)):
    return v1._SCORE, 1.0, 1.0, 0.0, acc_v1
  return reward, acc_v1, fmt, credit, acc_v1


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
  out = v1.compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
  acc_v1 = out["acc"]
  if out["fmt"] == 1.0 and acc_v1 == 0.0 and number_equal(v1.extract_boxed(solution_str), str(ground_truth)):
    out.update(score=v1._SCORE + out["length_penalty"], reward_raw=v1._SCORE, acc=1.0, format_credit=0.0)
  out.update(acc_v1=acc_v1, v2_only=out["acc"] - acc_v1, scorer_version=SCORER_VERSION)
  return out


# Meta fixture cases whose v2 reward intentionally differs from Meta's expected value: case -> v2 reward.
INTENTIONAL_DIVERGENCES = {"decimal vs integer gold": 1.0}

# The rule by example: (boxed content, gold, expected v2 reward, expected v1 reward). Each content is scored as
# "... \\boxed{<content>}". Shared by the self-test below and test_boxed_math_reward_v2.py.
RULE_CASES = [
    # --- credited by the v2 number rule (v1 gives the format score only)
    ("\\$18", "18", 1.0, 0.1),                              # currency \$ (v1 drops '$' but keeps the backslash)
    ("\\$26.00", "26", 1.0, 0.1),                           # currency + exact value 26.00 == 26
    ("\\$70,000", "70000", 1.0, 0.1),                       # currency + thousands grouping
    ("\\$1,450,000", "1,450,000", 1.0, 0.1),                # gold with grouping
    ("-\\$5", "-5", 1.0, 0.1),                              # sign before currency
    ("\\$-5", "-5", 1.0, 0.1),                              # sign after currency
    ("\u20ac77", "77", 1.0, 0.1),                           # euro
    ("\u00a377", "77", 1.0, 0.1),                           # pound
    ("28.00", "28", 1.0, 0.1),                              # exact value
    ("18.0", "18", 1.0, 0.1),                               # exact value (Meta fixture says 0.1: divergence)
    ("60\\%", "60", 1.0, 0.1),                              # percent
    ("60 \\%", "60", 1.0, 0.1),
    ("49^\\circ", "49", 1.0, 0.1),                          # degree
    ("49^{\\circ}", "49", 1.0, 0.1),
    ("49\u00b0", "49", 1.0, 0.1),
    ("-10 \\text{ degrees Fahrenheit}", "-10", 1.0, 0.1),   # unit, two words
    ("49^\\circ\\text{C}", "49", 1.0, 0.1),                 # unit then degree
    ("540 \\text{ meters}", "540", 1.0, 0.1),               # unit
    ("180,000 \\text{ meters}", "180000", 1.0, 0.1),        # grouping + unit
    ("55 \\text{ meters per second}", "55", 1.0, 0.1),      # unit, three words
    ("72 \\text{ Mb/hr}", "72", 1.0, 0.1),                  # '/' inside a unit word
    ("72 \\text{ in}^3", "72", 1.0, 0.1),                   # unit with exponent
    ("18 \\mathrm{kg}", "18", 1.0, 0.1),
    ("18\\,\\text{kg}", "18", 1.0, 0.1),                    # LaTeX thin space
    ("18~\\text{kg}", "18", 1.0, 0.1),
    ("18 kg", "18", 1.0, 0.1),                              # bare unit word
    ("18kg", "18", 1.0, 0.1),
    ("18 more apples", "18", 1.0, 0.1),                     # "more" alone is allowed ("than" is blocked)
    ("12 \\text{ quarters}", "12", 1.0, 0.1),
    ("\\$18 \\text{ per hour}", "18", 1.0, 0.1),            # unit + currency
    ("60\\% \\text{ off}", "60", 1.0, 0.1),                 # unit + percent
    ("18 \\text{ hours}.", "18", 1.0, 0.1),                 # trailing period
    ("\\text{18}", "18", 1.0, 0.1),                         # wrapper
    ("\\textbf{\\$18}", "18", 1.0, 0.1),                    # wrapper + currency
    ("\\text{18 hours}", "18", 1.0, 0.1),                   # wrapper + bare unit
    ("1{,}000", "1000", 1.0, 0.1),                          # LaTeX grouping
    ("1\\,000", "1000", 1.0, 0.1),                          # thin-space grouping
    # --- correct under v1 already (unchanged)
    ("18", "18", 1.0, 1.0),
    ("1,080", "1080", 1.0, 1.0),
    ("$42", "42", 1.0, 1.0),
    ("\\frac{1}{2}", "\\frac{1}{2}", 1.0, 1.0),             # v1 string match; v2's number rule never runs
    ("1,00", "100", 1.0, 1.0),                              # inherited v1 comma deletion (step 5 alone rejects)
    # --- format score under both: v2 must not turn these into correct answers
    ("\\$17", "18", 0.1, 0.1),                              # wrong value
    ("18.5", "18", 0.1, 0.1),                               # no tolerance
    ("\\$18.50", "18", 0.1, 0.1),
    ("-\\$10,000", "10000", 0.1, 0.1),                      # the sign counts
    ("\\$1,00", "100", 0.1, 0.1),                           # invalid grouping (v1 keeps the backslash)
    ("\\$10,0000", "100000", 0.1, 0.1),
    ("18 \\text{ and } 20", "18", 0.1, 0.1),                # two numbers
    ("18 \\text{ and } 20", "20", 0.1, 0.1),
    ("86 \\text{ glasses and } 42 \\text{ plates}", "42", 0.1, 0.1),
    ("5 \\text{ (approx 18)}", "5", 0.1, 0.1),              # not a unit phrase (digits, parentheses)
    ("5 \\text{ (approx 18)}", "18", 0.1, 0.1),
    ("7 \\frac{1}{12} \\text{ hours (or approximately 7.08 hours)}", "7", 0.1, 0.1),
    ("18 \\text{ hours and } 30 \\text{ minutes}", "18", 0.1, 0.1),
    ("18 \\text{ or more}", "18", 0.1, 0.1),                # blocked words
    ("18 \\text{ more than Bob}", "18", 0.1, 0.1),
    ("18 \\text{ approximately}", "18", 0.1, 0.1),
    ("18 \\text{ thousand}", "18", 0.1, 0.1),
    ("18 \\text{ dozen}", "18", 0.1, 0.1),
    ("18 \\text{ squared}", "18", 0.1, 0.1),
    ("18k", "18", 0.1, 0.1),
    ("18 M", "18", 0.1, 0.1),
    ("18^2", "18", 0.1, 0.1),                               # an exponent without a unit is not decoration
    ("\\frac{36}{2}", "18", 0.1, 0.1),                      # fractions are not evaluated
    ("\\dfrac{88}{3}", "22", 0.1, 0.1),
    ("6\\frac{2}{3}", "6", 0.1, 0.1),
    ("1.8e1", "18", 0.1, 0.1),
    ("2:00 \\text{ PM}", "2", 0.1, 0.1),                    # times are not numbers
    ("x = 18", "18", 0.1, 0.1),                             # no leading text
    ("\\text{about } 18", "18", 0.1, 0.1),
    ("\\text{Blue car: \\$100, Red car: \\$60}", "160", 0.1, 0.1),
    ("67\"", "67", 0.1, 0.1),                               # symbol not on the whitelist
    ("18 \\text{ hours} \\text{ per day}", "18", 0.1, 0.1),  # only ONE unit group is dropped
    ("\\$\\text{18}", "18", 0.1, 0.1),
    ("...", "18", 0.1, 0.1),
]


def run_selftest(fx=None):
  fx = fx or v1.META_FIXTURES
  fails = []
  for c in fx["reward_fixtures"]:
    want = INTENTIONAL_DIVERGENCES.get(c["case"], c["reward"])
    r = boxed_math_v2(c["response"], c["gold"])[0]
    ok = abs(r - want) < 1e-9
    note = f" (Meta {c['reward']}: intentional v2 divergence)" if c["case"] in INTENTIONAL_DIVERGENCES else ""
    print(f"{'OK ' if ok else 'FAIL'} {c['case']:<36} -> {r} (expected {want}){note}")
    fails += [] if ok else [c["case"]]
  for c in fx["overlong_fixtures"]:
    p = v1.overlong_penalty(c["completion_tokens"])
    ok = abs(p - c["penalty"]) < 1e-5
    print(f"{'OK ' if ok else 'FAIL'} overlong {c['completion_tokens']:>5} tokens -> {p:+.6f} "
          f"(expected {c['penalty']:+})")
    fails += [] if ok else [f"overlong {c['completion_tokens']}"]
  tr = compute_score("gsm8k_boxed_train", "\\boxed{\\$18}", "18", extra_info={"index": 0, "response_len": 2048})
  ev = compute_score("gsm8k_boxed_test512", "\\boxed{\\$18}", "18", extra_info={"index": 0, "response_len": 2048})
  pen_on = "gsm8k_boxed_train" in v1._PENALTY_SOURCES and v1._OVERLONG_BUFFER > 0 and v1._MAX_RESP_LEN == 2048
  want_tr = 0.0 if pen_on else 1.0
  ok = (tr["score"] == want_tr and tr["reward_raw"] == 1.0 and tr["acc"] == 1.0 and tr["acc_v1"] == 0.0
        and tr["v2_only"] == 1.0 and ev["score"] == 1.0 and ev["length_penalty"] == 0.0
        and tr["scorer_version"] == SCORER_VERSION)
  print(f"{'OK ' if ok else 'FAIL'} v2 fields + penalty scope: train score {tr['score']} "
        f"(raw 1.0, acc_v1 0, v2_only 1); eval score {ev['score']} (no penalty)")
  fails += [] if ok else ["fields/penalty scope"]
  n_ok = 0
  for content, gold, want2, want1 in RULE_CASES:
    resp = f"so the answer is \\boxed{{{content}}}"
    r2, r1 = boxed_math_v2(resp, gold)[0], v1.boxed_math(resp, gold)[0]
    ok = abs(r2 - want2) < 1e-9 and abs(r1 - want1) < 1e-9
    n_ok += ok
    if not ok:
      print(f"FAIL rule case {content!r} vs {gold!r}: v2 {r2} (want {want2}), v1 {r1} (want {want1})")
      fails.append(f"rule {content!r}/{gold!r}")
  print(f"{'OK ' if n_ok == len(RULE_CASES) else 'FAIL'} rule cases {n_ok}/{len(RULE_CASES)}")
  return fails


if __name__ == "__main__":
  fx = None
  if len(sys.argv) > 2 and sys.argv[1] == "--fixtures":
    fx = json.load(open(sys.argv[2]))
    print(f"using fixtures from {sys.argv[2]}")
  fails = run_selftest(fx)
  print("scorer:", dict(version=SCORER_VERSION, v1_sha256=V1_SHA256, format_score=v1._FORMAT_SCORE,
                        buffer=v1._OVERLONG_BUFFER, cap=v1._MAX_RESP_LEN, penalty_sources=sorted(v1._PENALTY_SOURCES)))
  print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
  raise SystemExit(0 if not fails else 1)
