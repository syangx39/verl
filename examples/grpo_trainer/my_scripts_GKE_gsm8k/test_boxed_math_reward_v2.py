#!/usr/bin/env python3
"""Tests for boxed_math_reward_v2.py (GSM8K boxed scorer v2). Run: python3 test_boxed_math_reward_v2.py [-v]

Runs with Meta's default reward knobs (cap 2048, buffer 512, penalty on gsm8k_boxed_train): REWARD_* variables are
removed from the environment before the scorer modules are imported.
"""
import decimal
import hashlib
import importlib.util
import os
import unittest

for _k in [k for k in os.environ if k.startswith("REWARD_")]:
  del os.environ[_k]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
  spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


v2 = _load("boxed_math_reward_v2_under_test", "boxed_math_reward_v2.py")
v1 = v2.v1
D = decimal.Decimal


def box(content):
  return f"Step by step ... so the answer is \\boxed{{{content}}}"


class RuleCases(unittest.TestCase):

  def test_rule_cases(self):
    for content, gold, want2, want1 in v2.RULE_CASES:
      with self.subTest(content=content, gold=gold):
        self.assertAlmostEqual(v2.boxed_math_v2(box(content), gold)[0], want2)
        self.assertAlmostEqual(v1.boxed_math(box(content), gold)[0], want1)

  def test_rule_cases_cover_both_directions(self):
    promoted = [c for c in v2.RULE_CASES if c[2] == 1.0 and c[3] == 0.1]
    rejected = [c for c in v2.RULE_CASES if c[2] == 0.1]
    self.assertGreaterEqual(len(promoted), 30)
    self.assertGreaterEqual(len(rejected), 30)

  def test_observed_gemma_forms(self):
    # The step-0 forms behind the 44.81% (v1) vs 80.67% (lenient audit) gap, PR #14 smoke.
    for content, gold in [("\\$18", "18"), ("\\$243.00", "243"), ("\\$57,500", "57500"), ("35 \\text{ hours}", "35"),
                          ("60\\%", "60"), ("82^\\circ", "82"), ("68 \\text{ square feet}", "68"),
                          ("43,200 \\text{ pounds}", "43200"), ("\\$2,880,000", "2880000")]:
      with self.subTest(content=content):
        self.assertEqual(v2.boxed_math_v2(box(content), gold)[:2], (1.0, 1.0))

  def test_observed_wrong_or_ambiguous_forms_stay_wrong(self):
    for content, gold in [("\\$50.00", "64"), ("\\$75,000", "25000"), ("1200 \\text{ feet}", "200"),
                          ("135 \\text{ kWh}", "81"), ("2:00 \\text{ PM}", "2"), ("2:00\\ \\text{pm}", "2"),
                          ("18 \\text{ portraits and } 27 \\text{ selfies}", "45"), ("\\$9,583.20", "9360"),
                          ("522.39 \\text{ calories}", "525"), ("-\\$10,000", "10000")]:
      with self.subTest(content=content):
        self.assertEqual(v2.boxed_math_v2(box(content), gold)[:2], (0.1, 0.0))


class MetaFixturesAndV1(unittest.TestCase):

  def test_v1_file_is_the_frozen_port(self):
    with open(v2.V1_PATH, "rb") as f:
      self.assertEqual(hashlib.sha256(f.read()).hexdigest(), v2.V1_SHA256)
    self.assertEqual(v1.run_fixtures(v1.META_FIXTURES), [])

  def test_meta_fixtures_with_documented_divergence_only(self):
    self.assertEqual(set(v2.INTENTIONAL_DIVERGENCES), {"decimal vs integer gold"})
    for c in v1.META_FIXTURES["reward_fixtures"]:
      with self.subTest(case=c["case"]):
        want = v2.INTENTIONAL_DIVERGENCES.get(c["case"], c["reward"])
        self.assertAlmostEqual(v2.boxed_math_v2(c["response"], c["gold"])[0], want)
    self.assertEqual(v2.run_selftest(), [])

  def test_never_removes_v1_credit(self):
    responses = [c["response"] for c in v1.META_FIXTURES["reward_fixtures"]] + [box(c[0]) for c in v2.RULE_CASES]
    golds = sorted({c["gold"] for c in v1.META_FIXTURES["reward_fixtures"]} | {c[1] for c in v2.RULE_CASES})
    for r in responses:
      for g in golds:
        reward1, acc1, fmt1, _ = v1.boxed_math(r, g)
        reward2, acc2, fmt2, _, acc_v1 = v2.boxed_math_v2(r, g)
        self.assertEqual(acc_v1, acc1)
        self.assertEqual(fmt2, fmt1)
        self.assertGreaterEqual(acc2, acc1, (r, g))
        self.assertGreaterEqual(reward2, reward1, (r, g))

  def test_extraction_is_v1(self):
    self.assertEqual(v2.boxed_math_v2("first \\boxed{\\$18} then \\boxed{7}", "18")[0], 0.1)   # last box wins
    self.assertEqual(v2.boxed_math_v2("first \\boxed{7} then \\boxed{\\$18}", "18")[0], 1.0)
    self.assertEqual(v2.boxed_math_v2("so \\boxed{\\$18", "18")[0], 0.0)                      # truncated
    self.assertEqual(v2.boxed_math_v2("\\boxed{}", "18")[0], 0.0)
    self.assertEqual(v2.boxed_math_v2("The answer is \\$18.", "18")[0], 0.0)                   # no box


class ScoreFields(unittest.TestCase):

  def test_fields_and_types(self):
    out = v2.compute_score("gsm8k_boxed_test", box("\\$18"), "18", extra_info={"index": 10000005})
    ref = v1.compute_score("gsm8k_boxed_test", box("\\$18"), "18", extra_info={"index": 10000005})
    self.assertEqual(set(out), set(ref) | {"acc_v1", "v2_only", "scorer_version"})
    self.assertTrue(all(isinstance(v, float) for v in out.values()), out)
    self.assertEqual((out["score"], out["reward_raw"], out["acc"], out["fmt"], out["format_credit"]),
                     (1.0, 1.0, 1.0, 1.0, 0.0))
    self.assertEqual((out["acc_v1"], out["v2_only"], out["scorer_version"], out["qid"]), (0.0, 1.0, 2.0, 10000005.0))
    self.assertEqual((ref["score"], ref["acc"], ref["format_credit"]), (0.1, 0.0, 0.1))

  def test_unchanged_answers_keep_v1_values(self):
    for resp, gold in [(box("18"), "18"), (box("17"), "18"), ("no box 18", "18"), (box("\\$17"), "18")]:
      out = v2.compute_score("gsm8k_boxed_test", resp, gold, extra_info={"index": 1})
      ref = v1.compute_score("gsm8k_boxed_test", resp, gold, extra_info={"index": 1})
      self.assertEqual({k: out[k] for k in ref}, ref)
      self.assertEqual((out["acc_v1"], out["v2_only"]), (ref["acc"], 0.0))

  def test_penalty_mirrors_v1(self):
    for src in ("gsm8k_boxed_train", "gsm8k_boxed_test"):
      for n in (100, 1536, 1792, 2048, 2560):
        for resp in (box("\\$18"), box("18"), box("17"), "no box"):
          out = v2.compute_score(src, resp, "18", extra_info={"index": 0, "response_len": n})
          ref = v1.compute_score(src, resp, "18", extra_info={"index": 0, "response_len": n})
          self.assertEqual((out["length_penalty"], out["overlong"]), (ref["length_penalty"], ref["overlong"]))
          self.assertAlmostEqual(out["score"], out["reward_raw"] + out["length_penalty"])
    with self.assertRaises(RuntimeError):   # same guard as v1 when the penalty is on and response_len is missing
      v2.compute_score("gsm8k_boxed_train", box("18"), "18", extra_info={"index": 0})


class NumberGrammar(unittest.TestCase):

  def test_values_and_steps(self):
    cases = {
        "18": (D(18), []), "\\$1,450,000": (D(1450000), ["currency", "grouping"]),
        "-\\$5": (D(-5), ["currency"]), "28.00": (D("28.00"), []), "60\\%": (D(60), ["percent"]),
        "49^{\\circ}": (D(49), ["degree"]), "\\text{18}": (D(18), ["wrapper"]),
        "540 \\text{ meters}": (D(540), ["unit"]), "\\textbf{\\$18}": (D(18), ["wrapper", "currency"]),
        "\\$18 \\text{ per hour}": (D(18), ["unit", "currency"]), "1{,}000": (D(1000), ["grouping"]),
    }
    for s, want in cases.items():
      with self.subTest(s=s):
        self.assertEqual(v2.parse_number(s), want)

  def test_not_numbers(self):
    for s in ["", "...", "18 \\text{ and } 20", "1,00", "10,0000", ",100", "18.", "1.8e1", "\\frac{1}{2}", "2:00",
              "18^2", "18 \\times 2", "x = 18", "18-20", "18, 20", "eighteen", "\u221218", "18\\%\\%", "\\$\\$18"]:
      with self.subTest(s=s):
        if s == "18.":   # trailing period is step 0, so this one IS a number
          self.assertEqual(v2.number(s), D(18))
        else:
          self.assertIsNone(v2.number(s))

  def test_exact_value_no_tolerance(self):
    self.assertTrue(v2.number_equal("28.00", "28"))
    self.assertTrue(v2.number_equal("0.50", "0.5"))
    self.assertFalse(v2.number_equal("18.5", "18"))
    self.assertFalse(v2.number_equal("0.333", "0.3333"))
    self.assertFalse(v2.number_equal("0.30000000000000004", "0.3"))
    self.assertFalse(v2.number_equal("18.000001", "18"))
    self.assertFalse(v2.number_equal("18", "\\frac{36}{2}"))   # gold that is not a number: rule never fires

  def test_every_blocked_word_blocks(self):
    for w in sorted(v2.BLOCKED_WORDS):
      for form in (f"18 \\text{{ {w}}}", f"18 {w}", f"18 \\text{{ {w.upper()}}}", f"18 \\text{{ kg {w}}}"):
        with self.subTest(form=form):
          self.assertIsNone(v2.number(form))
    for w in sorted(v2.BLOCKED_WORDS_CASED):
      with self.subTest(w=w):
        self.assertIsNone(v2.number(f"18 \\text{{ {w}}}"))
    self.assertEqual(v2.number("18 \\text{ mm}"), D(18))       # lower-case mm is millimetres

  def test_unit_phrase_limits(self):
    self.assertEqual(v2.number("18 \\text{ cubic feet per minute}"), D(18))          # 4 words
    self.assertIsNone(v2.number("18 \\text{ cubic feet per minute each}"))           # 5 words
    self.assertIsNone(v2.number("18 \\text{ hours, total}"))                          # comma
    self.assertIsNone(v2.number("18 \\text{ (eighteen)}"))                            # parentheses
    self.assertIsNone(v2.number("18 \\text{ 2nd}"))                                   # digit
    self.assertIsNone(v2.number("18 \\text{ }"))                                      # empty
    self.assertEqual(v2.number("18 \\text{ m/s}^2"), D(18))
    self.assertEqual(v2.number("18 \\text{ o'clock}"), D(18))
    self.assertEqual(v2.number("18 sq. ft."), D(18))

  def test_each_step_at_most_once(self):
    self.assertIsNone(v2.number("18 \\text{ kg} \\text{ total}"))   # two unit groups
    self.assertIsNone(v2.number("\\text{\\text{18}}"))              # two wrappers
    self.assertIsNone(v2.number("\\$\u20ac18"))                     # two currencies
    self.assertIsNone(v2.number("18\\%^\\circ"))                    # two suffixes
    self.assertIsNone(v2.number("--18"))                            # two signs
    self.assertIsNone(v2.number("-\\$-18"))

  def test_long_content_is_not_a_number(self):
    self.assertIsNone(v2.number("1" * (v2.MAX_NUMBER_CHARS + 1)))
    self.assertEqual(v2.number("1" * v2.MAX_NUMBER_CHARS), D("1" * v2.MAX_NUMBER_CHARS))


if __name__ == "__main__":
  unittest.main()
