#!/usr/bin/env python3
"""verl custom reward: faithful port of the MaxText RL reward stack.

Source of truth: maxtext/trainers/post_train/rl/utils_rl.py with the TPU run's
effective config (rl.yml defaults, nothing overridden in the JobSet):

    reward_exact_answer            = 1.0
    reward_white_space_format_match= 1.0
    reward_exact_format_match      = 0.1
    reward_partial_format_match    = 0.0   -> match_format_approximately == no-op
    penalty_incorrect_format       = 0.0
    penalty_incorrect_answer       = 0.0

MaxText total reward per completion = sum of three fns:
    match_format_exactly        -> +0.1 iff <reasoning>..</reasoning>..<answer>..</answer>
    match_format_approximately  -> +0.0 always (all weights zero)  [omitted]
    check_numbers               -> +1.0 iff answer correct (exact / whitespace /
                                   math_verify equivalence), else 0.0
Reward support: {0.0, 0.1, 1.0, 1.1}.

verl interface (custom_reward_function.path/.name):
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float
  - solution_str : the decoded completion (response only)  == MaxText `completion`
  - ground_truth : json.dumps([answer, answer]) written by our preprocess script
                   == MaxText `answer` element

Porting notes / deltas:
  * normalize chain (SUBSTITUTIONS/UNITS/REMOVED_EXPRESSIONS, fix_latex_escaping,
    normalize_final_answer, extract_answer) copied verbatim — these define which
    answers count as correct; any drift breaks curve overlay.
  * math_verify: MaxText runs it in a kill-able spawn pool (hung sympy). Here we
    call in-process with try/except. Pathological hangs are rarer than crashes;
    if a hang is observed, add signal.alarm or a pebble pool. Pin the SAME
    math-verify version as the TPU image (record in §3.5 version table).
  * debug logging / MCQ path / gsm8k hash path dropped (not exercised by
    OpenMathInstruct-2 default question_type).
"""

import itertools
import json
import re

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

EPSILON = 1e-6
FALLBACK_ANSWER = "-1000000"

# --- rl.yml effective values (frozen) --------------------------------------
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

REWARD_EXACT_ANSWER = 1.0
REWARD_WHITE_SPACE_FORMAT_MATCH = 1.0
REWARD_EXACT_FORMAT_MATCH = 0.1
PENALTY_INCORRECT_FORMAT = 0.0
PENALTY_INCORRECT_ANSWER = 0.0

# --- regexes (utils_rl.get_match_format_regex / get_answer_fallback_regex) --
MATCH_FORMAT = re.compile(
    rf"{REASONING_START}.+{REASONING_END}.*?{SOLUTION_START}(.+?){SOLUTION_END}",
    flags=re.MULTILINE | re.DOTALL,
)
ANSWER_TAG = re.compile(
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}",
    flags=re.MULTILINE | re.DOTALL,
)

# --- normalization tables: copied VERBATIM from utils_rl.py ----------------
SUBSTITUTIONS = [
    ("\\\\", "\\"),
    ("\\tfrac", "\\frac"),
    ("\\dfrac", "\\frac"),
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" or ", ","),
    (" and ", ","),
    ("million", "*10^6"),
    ("billion", "*10^9"),
    ("trillion", "*10^12"),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

UNITS = [
    "yard", "foot", "feet", "mile", "day", "week", "month", "year", "hour",
    "minute", "second", "centimeter", "meter", "cm", "mm", "km", "inch",
    "degree", "pound", "cent", "mph",
]

REMOVED_EXPRESSIONS = [
    "\\left", "\\right", "\\!", "square", "ways", "integers", "dollars",
    "units", "\\ldots", "sue", "points", "digits", "gm", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]

LATEX_COMMANDS = [
    "frac", "sqrt", "pi", "theta", "alpha", "beta", "gamma", "delta", "sum",
    "int", "infty", "cdot", "times", "div", "pm", "mp", "leq", "geq", "neq",
    "approx", "equiv", "sin", "cos", "tan", "log", "ln", "exp", "lim", "to",
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "overline",
    "underline", "hat", "bar", "vec", "dot", "ddot", "mathbb", "mathbf",
    "mathrm", "text", "textbf", "textit", "boxed", "left", "right", "choose",
    "binom",
]

ESCAPE_FIXES = [
    ("\f", "rac", r"\frac"),
    ("\n", "ewline", r"\newline"),
    ("\n", "e", r"\ne"),
    ("\t", "heta", r"\theta"),
    ("\t", "an", r"\tan"),
    ("\t", "o", r"\to"),
    ("\t", "imes", r"\times"),
    ("\t", "ext", r"\text"),
    ("\t", "extbf", r"\textbf"),
    ("\t", "extit", r"\textit"),
    ("\r", "ightarrow", r"\rightarrow"),
    ("\r", "ightarrow", r"\Rightarrow"),
    ("\b", "eta", r"\beta"),
    ("\b", "ar", r"\bar"),
    ("\b", "inom", r"\binom"),
    ("\b", "oxed", r"\boxed"),
    ("\a", "lpha", r"\alpha"),
    ("\a", "pprox", r"\approx"),
    ("\v", "ec", r"\vec"),
]


def boxed(x: str) -> str:
  return "\\boxed{" + x + "}" if not x.startswith("\\boxed{") else x


def normalize_final_answer(final_answer: str) -> str:
  """Verbatim port of utils_rl.normalize_final_answer."""
  final_answer = final_answer.split("=")[-1]
  final_answer = re.sub(r"([0-9]) +([0-9])", r"\1+\2", final_answer)
  for before, after in SUBSTITUTIONS:
    final_answer = final_answer.replace(before, after)
  for unit in UNITS:
    final_answer = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", final_answer)
  for expr in REMOVED_EXPRESSIONS:
    final_answer = final_answer.replace(expr, "")
  final_answer = re.sub(
      r".*?(\d+)?\s*\$\s*(\d+)?\s*(\\frac\{.*?\}\{.*?\}|\d+/\d+)\s*\$.*",
      lambda m: f"${w}{m.group(3)}$" if (w := (m.group(1) or m.group(2))) else f"${m.group(3)}$",
      final_answer,
  )
  final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
  final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
  final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
  final_answer = final_answer.replace("$", "")
  if final_answer.startswith("."):
    final_answer = "0" + final_answer
  final_answer = final_answer.replace("{.", "{0.")
  if len(final_answer) >= 2 and final_answer[0] == "{" and final_answer[-1] == "}":
    final_answer = final_answer[1:-1]
  try:
    f = float(final_answer)
    if abs(f - round(f)) < 1e-7:
      final_answer = str(int(round(f)))
  except (ValueError, OverflowError):
    pass
  if final_answer.replace(",", "").isdigit():
    final_answer = final_answer.replace(",", "")
  return final_answer


def fix_latex_escaping(text: str) -> str:
  """Verbatim port of utils_rl.fix_latex_escaping."""
  for escape_char, suffix, latex_cmd in ESCAPE_FIXES:
    if escape_char in text:
      text = text.replace(escape_char + suffix, latex_cmd)
  for cmd in LATEX_COMMANDS:
    text = re.sub(rf"(?<!\\)\b{cmd}\b", rf"\\{cmd}", text)
  return text


def preprocess_math_string(text: str) -> str:
  return fix_latex_escaping(normalize_final_answer(text).strip())


def extract_answer(response: str) -> str:
  """Verbatim port of utils_rl.extract_answer (config tokens inlined)."""
  answer_matches = ANSWER_TAG.findall(response)
  content = answer_matches[-1] if answer_matches else response
  boxed_matches = []
  stack = []
  for i, ch in enumerate(content):
    if ch == "{":
      stack.append(i)
    elif ch == "}":
      if not stack:
        continue
      op = stack.pop()
      if content[:op].endswith(r"\boxed"):
        boxed_matches.append(content[op + 1: i].strip())
  if boxed_matches:
    return boxed_matches[-1]
  m = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if m:
    return m.group(1).strip()
  fallback_matches = ANSWER_TAG.findall(response)
  if fallback_matches:
    return fallback_matches[-1].strip()
  return FALLBACK_ANSWER


def _math_verify_equal(gold_boxed_list, guess_boxed: str) -> bool:
  """In-process equivalent of MaxText verify_math_worker (spawn pool dropped).

  math_verify.verify(gold, target): order matters (gold first).
  """
  try:
    guess_parsed = parse(guess_boxed, (ExprExtractionConfig(), LatexExtractionConfig()))
    golds_parsed = list(itertools.chain.from_iterable(
        parse(g, (ExprExtractionConfig(), LatexExtractionConfig())) for g in gold_boxed_list))
    if not guess_parsed or not golds_parsed:
      return False
    return bool(verify(golds_parsed, guess_parsed))
  except Exception:
    return False


def _format_score(completion: str) -> float:
  """match_format_exactly."""
  return REWARD_EXACT_FORMAT_MATCH if MATCH_FORMAT.search(completion) else 0.0


def _answer_score(completion: str, ground_truth_json: str) -> float:
  """check_numbers (single-completion form)."""
  try:
    acceptable = list(dict.fromkeys(json.loads(ground_truth_json)))
  except (json.JSONDecodeError, TypeError):
    acceptable = [str(ground_truth_json)]

  guess = extract_answer(completion)
  if guess == FALLBACK_ANSWER:
    return PENALTY_INCORRECT_ANSWER  # 0.0

  score = PENALTY_INCORRECT_FORMAT  # 0.0 default
  for true_answer in acceptable:
    if guess == true_answer:
      return max(score, REWARD_EXACT_ANSWER)
    if guess.strip() == true_answer.strip():
      score = max(score, REWARD_WHITE_SPACE_FORMAT_MATCH)
  if score > 0:
    return score

  norm_guess = boxed(preprocess_math_string(guess))
  norm_answers = [boxed(preprocess_math_string(a)) for a in acceptable]
  if _math_verify_equal(norm_answers, norm_guess):
    return REWARD_EXACT_ANSWER
  return 0.0


# --------------------------- lendist / reward dump --------------------------
# Parity instrumentation mirroring the TPU side's patch_lendist.py. Headline
# cross-check numbers (osl_mean, cap-hit %) come from tensorboard for free
# (response_length/mean|max|min|clip_ratio, full-batch, every step). This dump
# adds a small qualitative sample: the FIRST N reward calls in each worker
# process per step-window, giving a handful of concrete completions per step
# (reward split + tail text) without per-call RNG or volume.
#   REWARD_DUMP_DIR=<dir>       enable (default off)
#   REWARD_DUMP_PER_STEP=2      samples per worker per step-window
# Step detection is call-count based: batch_size x rollout_n / num_workers
# calls arrive per step per worker; we reset the quota every REWARD_DUMP_EVERY
# calls (default 512 = 2048 completions / 4 reward workers).
import json as _json, os as _os

_DUMP_DIR = _os.environ.get("REWARD_DUMP_DIR", "")
_DUMP_PER_STEP = int(_os.environ.get("REWARD_DUMP_PER_STEP", "2"))
_DUMP_EVERY = int(_os.environ.get("REWARD_DUMP_EVERY", "512"))
_call_count = 0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
  """verl custom reward entry point. Total = format(0.1) + answer(1.0)."""
  del data_source, kwargs
  completion = solution_str if isinstance(solution_str, str) else str(solution_str)
  fmt = _format_score(completion)
  ans = _answer_score(completion, ground_truth)
  global _call_count
  if _DUMP_DIR and (_call_count % _DUMP_EVERY) < _DUMP_PER_STEP:
    try:  # fail-open: a logging bug must never affect the run
      _os.makedirs(_DUMP_DIR, exist_ok=True)
      n_tok = None
      if isinstance(extra_info, dict):
        n_tok = extra_info.get("num_response_tokens") or extra_info.get("response_length")
      rec = {"call": _call_count, "fmt": fmt, "ans": ans,
             "n_chars": len(completion), "n_tokens": n_tok,
             "gt": str(ground_truth)[:80], "tail": completion[-160:]}
      # Per-run SUBDIRECTORY: <dump_dir>/<run_tag>/pid*.jsonl -- one folder
      # per run, so runs never mix and archival needs no mv. run_tag prefers
      # EXPERIMENT_NAME (usually absent in reward workers, which see only
      # container env) and falls back to this process's first-call date+hour;
      # all workers of one run boot within the same minute, so they agree on
      # the folder and it matches the tee'd log's timestamp.
      _run_tag = _os.environ.get("EXPERIMENT_NAME")
      if not _run_tag:
        if not hasattr(compute_score, "_boot_tag"):
          import datetime as _dt
          compute_score._boot_tag = _dt.datetime.now().strftime("run%m%d_%H%M")
        _run_tag = compute_score._boot_tag
      _run_dir = _os.path.join(_DUMP_DIR, _run_tag)
      _os.makedirs(_run_dir, exist_ok=True)
      fn = _os.path.join(_run_dir, f"pid{_os.getpid()}.jsonl")
      with open(fn, "a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
      pass
  _call_count += 1
  return fmt + ans


# --------------------------- self-test -------------------------------------
if __name__ == "__main__":
  gt = json.dumps(["72", "72"])
  cases = [
      # (completion, expected, note)
      ("<reasoning>2*36</reasoning><answer>72</answer>", 1.1, "exact + format"),
      ("<reasoning>2*36</reasoning><answer>\\boxed{72}</answer>", 1.1, "boxed inside tags"),
      ("blah <answer> 72 </answer>", 1.0, "whitespace match, no format"),
      ("<answer>36*2</answer>", 1.0, "math_verify equivalence, no format"),
      ("<reasoning>hmm</reasoning><answer>71</answer>", 0.1, "format only, wrong"),
      ("no tags at all 72", 0.0, "fallback -> FALLBACK_ANSWER -> 0"),
      ("<answer>7.2e1</answer>", 1.0, "math_verify numeric forms"),
  ]
  for completion, expected, note in cases:
    got = compute_score("x", completion, gt)
    flag = "OK " if abs(got - expected) < 1e-9 else "FAIL"
    print(f"{flag} {note}: got={got} expected={expected}")
