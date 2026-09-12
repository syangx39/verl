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
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict
  - solution_str : the decoded completion (response only)  == MaxText `completion`
  - ground_truth : json.dumps([answer, answer]) written by our preprocess script
                   == MaxText `answer` element

PHASE-0 CHANGE (logging only, reward semantics untouched):
  * compute_score returns a dict. verl's reward manager uses ["score"] as the
    training reward (identical to the old scalar: fmt + ans) and forwards the
    other keys as reward_extra_info:
        score = fmt + ans          (what the optimizer sees; unchanged)
        acc   = 1.0 if ans >= 1.0  (answer correct)            <- Y axis of the main plot
        fmt   = 1.0 if fmt  > 0    (<reasoning>/<answer> tags present)
    Validation logs these per data_source (val-core/<ds>/acc/mean@1, val-aux/...).
    For the per-step TRAINING acc/fmt curve use trainer.rollout_data_dir: verl then
    writes <dir>/<step>.jsonl with input/output/gts/score/acc/fmt for every sample
    of every step, in a background thread. plot_phase0.py reads that. (An earlier
    version aggregated inside this function; verl calls compute_score from several
    threads per AgentLoopWorker and also for validation rows, so call-count windows
    are neither thread-safe nor step-aligned. Dropped.)

Porting notes / deltas:
  * normalize chain (SUBSTITUTIONS/UNITS/REMOVED_EXPRESSIONS, fix_latex_escaping,
    normalize_final_answer, extract_answer) copied verbatim -- these define which
    answers count as correct; any drift breaks curve overlay.
  * math_verify: MaxText runs it in a kill-able spawn pool (hung sympy). verl calls
    compute_score from THREADS, where math_verify's signal timeouts raise, so this
    file runs equivalence checks in its own fork-context pool with a get(timeout)
    and pool reset on hang -- see the "killable process pool" section. Pin the
    SAME math-verify version and the same timeout as the TPU image.
  * debug logging / MCQ path / gsm8k hash path dropped (not exercised by
    OpenMathInstruct-2 default question_type).
"""

import datetime as _dt
import itertools
import json
import os
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


# ---- math_verify in killable worker processes ---------------------------------
# Two constraints collide: (1) math_verify's own timeouts use signal.alarm(), which
# only works in a MAIN thread, and verl calls compute_score from worker threads;
# (2) sympy can hang on pathological input, and one hung call would stall a
# synchronous RL step. MaxText solves both with a kill-able spawn pool
# (math_verify_num_procs). Same idea here: N forked worker processes per reward
# process, talking over Pipes with plain data (no function pickling, so it works
# however verl loaded this file). A call = send -> poll(timeout) -> recv; on
# timeout that ONE worker is killed and replaced. Inside a worker (its main
# thread) math_verify's signal timeouts are legal again and act as a 2nd guard.
#
# Failure policy (v4, per review): there is NO silent fallback. If the workers
# cannot be started, or a replacement cannot be forked, the reward RAISES -- a
# reward whose timeout semantics silently changed is worse than a dead run.
# The only way to run without workers is the explicit REWARD_MV_POOL=0 (meant
# for offline re-scoring in a main thread), and that is reported as mv_mode.
# Workers return a structured status so exceptions inside math_verify are
# counted (mv_exc) instead of being folded into "wrong answer".
#   REWARD_MV_POOL=1        use workers (0 -> in-process, no hang protection)
#   REWARD_MV_PROCS=4       workers per reward process
#   REWARD_MV_TIMEOUT=5     seconds per equivalence check (match MaxText's value)
#   REWARD_MATH_VERIFY_MAX_CHARS=400   length guard (counted as lenrej, scored 0)
# NOTE: these are read from the environment of the process that runs the reward
# (a Ray actor) -- the launcher forwards them via ray runtime_env.
# Per-call flags returned in the reward dict: mv_used (math_verify was needed),
# mv_timeout, mv_exc, mv_lenrej. They land in val-aux metrics and in the rollout dump.
import multiprocessing as _mp
import queue as _queue
import threading as _threading
from itertools import chain as _chain

_MV_POOL_ENABLED = os.environ.get("REWARD_MV_POOL", "1") == "1"
_MV_PROCS = int(os.environ.get("REWARD_MV_PROCS", "4"))
_MV_TIMEOUT = float(os.environ.get("REWARD_MV_TIMEOUT", "5"))
_MV_MAX_CHARS = int(os.environ.get("REWARD_MATH_VERIFY_MAX_CHARS", "400"))
_MV_CFG = (ExprExtractionConfig(), LatexExtractionConfig())
_mv_ctx = _mp.get_context("fork")
_mv_idle = _queue.Queue()      # idle _MVWorker objects; bounded by _MV_PROCS
_mv_stats = {"calls": 0, "timeout": 0, "exc": 0, "lenrej": 0, "replaced": 0}
_mv_stats_lock = _threading.Lock()


class MathVerifyPoolError(RuntimeError):
  """Raised when the killable worker pool cannot be (re)built. Never swallowed."""


def _mv_check(gold_boxed_list, guess_boxed, t):
  """The actual math_verify call. Returns ("ok", bool) or ("exc", <ExceptionName>).

  t = per-call timeout in seconds for math_verify's own signal-based guard
  (only legal in a main thread); t=None disables it (in-process mode).
  """
  try:
    guess = parse(guess_boxed, _MV_CFG, parsing_timeout=t)
    golds = list(_chain.from_iterable(parse(g, _MV_CFG, parsing_timeout=t) for g in gold_boxed_list))
    if not guess or not golds:
      return ("ok", False)
    return ("ok", bool(verify(golds, guess, timeout_seconds=t)))
  except Exception as e:  # noqa: BLE001 -- reported as status, not swallowed
    return ("exc", type(e).__name__)


def _mv_server(conn):
  """Worker process loop: recv (golds, guess) -> send status tuple."""
  t = max(1, int(_MV_TIMEOUT))
  while True:
    try:
      golds, guess = conn.recv()
    except (EOFError, OSError):
      return
    try:
      conn.send(_mv_check(golds, guess, t))
    except (EOFError, OSError):
      return


class _MVWorker:
  def __init__(self):
    self.conn, child = _mv_ctx.Pipe()
    self.proc = _mv_ctx.Process(target=_mv_server, args=(child,), daemon=True)
    self.proc.start()
    child.close()

  def kill(self):
    try:
      self.proc.kill()
      self.proc.join(1)
    except Exception:
      pass
    try:
      self.conn.close()
    except Exception:
      pass


def _mv_spawn_worker():
  try:
    return _MVWorker()
  except Exception as e:
    raise MathVerifyPoolError(f"cannot fork math_verify worker: {type(e).__name__}: {e}") from e


def _mv_init():
  """Start the workers at import. Raises if REWARD_MV_POOL=1 and they cannot start."""
  if not _MV_POOL_ENABLED:
    return
  for _ in range(_MV_PROCS):
    _mv_idle.put(_mv_spawn_worker())


def _mv_bump(key):
  with _mv_stats_lock:
    _mv_stats[key] += 1


def _math_verify_equal(gold_boxed_list, guess_boxed: str):
  """Returns (is_equal, flags). flags keys: mv_used, mv_timeout, mv_exc, mv_lenrej (0/1)."""
  flags = {"mv_used": 1.0, "mv_timeout": 0.0, "mv_exc": 0.0, "mv_lenrej": 0.0}
  _mv_bump("calls")
  if len(guess_boxed) > _MV_MAX_CHARS or any(len(g) > _MV_MAX_CHARS for g in gold_boxed_list):
    flags["mv_lenrej"] = 1.0
    _mv_bump("lenrej")
    return False, flags
  if not _MV_POOL_ENABLED:            # explicit opt-out only (REWARD_MV_POOL=0)
    status, val = _mv_check(gold_boxed_list, guess_boxed, None)
    if status == "exc":
      flags["mv_exc"] = 1.0
      _mv_bump("exc")
      return False, flags
    return bool(val), flags
  try:
    w = _mv_idle.get(timeout=_MV_TIMEOUT * 4)
  except _queue.Empty:
    raise MathVerifyPoolError("no idle math_verify worker within 4x timeout -- pool starved or dead")
  status = None
  try:
    w.conn.send((list(gold_boxed_list), guess_boxed))
    if w.conn.poll(_MV_TIMEOUT + 1.0):
      status, val = w.conn.recv()
      _mv_idle.put(w)
      if status == "exc":
        flags["mv_exc"] = 1.0
        _mv_bump("exc")
        return False, flags
      return bool(val), flags
    flags["mv_timeout"] = 1.0
    _mv_bump("timeout")
  except MathVerifyPoolError:
    raise
  except Exception:                    # broken pipe / dead worker: counted, worker replaced
    flags["mv_exc"] = 1.0
    _mv_bump("exc")
  w.kill()
  _mv_idle.put(_mv_spawn_worker())     # raises MathVerifyPoolError if it cannot
  _mv_bump("replaced")
  return False, flags


def mv_stats():
  """Snapshot of per-process counters (for tests / the launcher pre-flight)."""
  with _mv_stats_lock:
    return dict(_mv_stats, pool=_MV_POOL_ENABLED, cfg_procs=_MV_PROCS, cfg_timeout_s=_MV_TIMEOUT,
                cfg_max_chars=_MV_MAX_CHARS, idle=_mv_idle.qsize())


_mv_init()   # fork early, at import, before verl's reward threads exist; raises on failure


def _format_score(completion: str) -> float:
  """match_format_exactly."""
  return REWARD_EXACT_FORMAT_MATCH if MATCH_FORMAT.search(completion) else 0.0


_NOFLAGS = {"mv_used": 0.0, "mv_timeout": 0.0, "mv_exc": 0.0, "mv_lenrej": 0.0}


def _answer_score(completion: str, ground_truth_json: str):
  """check_numbers (single-completion form). Returns (score, mv_flags)."""
  try:
    acceptable = list(dict.fromkeys(json.loads(ground_truth_json)))
  except (json.JSONDecodeError, TypeError):
    acceptable = [str(ground_truth_json)]

  guess = extract_answer(completion)
  if guess == FALLBACK_ANSWER:
    return PENALTY_INCORRECT_ANSWER, dict(_NOFLAGS)  # 0.0

  score = PENALTY_INCORRECT_FORMAT  # 0.0 default
  for true_answer in acceptable:
    if guess == true_answer:
      return max(score, REWARD_EXACT_ANSWER), dict(_NOFLAGS)
    if guess.strip() == true_answer.strip():
      score = max(score, REWARD_WHITE_SPACE_FORMAT_MATCH)
  if score > 0:
    return score, dict(_NOFLAGS)

  norm_guess = boxed(preprocess_math_string(guess))
  norm_answers = [boxed(preprocess_math_string(a)) for a in acceptable]
  ok, flags = _math_verify_equal(norm_answers, norm_guess)
  return (REWARD_EXACT_ANSWER if ok else 0.0), flags


# --------------------------- sample dump -----------------------------------
# A small qualitative sample of concrete completions (tail text + reward split),
# the FIRST N reward calls per process per window of REWARD_DUMP_EVERY calls.
#   REWARD_DUMP_DIR=<dir>       enable (default off)
#   REWARD_DUMP_PER_STEP=2      samples per process per window
#   REWARD_DUMP_EVERY=512       calls per window
# verl calls compute_score from several threads per AgentLoopWorker process and
# from many nodes (pids collide across nodes), so: one lock per process, and the
# file name carries the hostname. Validation rows also pass through here; the
# record carries data_source so they can be told apart.
import socket as _socket
import threading as _threading

_DUMP_DIR = os.environ.get("REWARD_DUMP_DIR", "")
_DUMP_PER_STEP = int(os.environ.get("REWARD_DUMP_PER_STEP", "2"))
_DUMP_EVERY = int(os.environ.get("REWARD_DUMP_EVERY", "512"))
_call_count = 0
_lock = _threading.Lock()
_boot_tag = None
_host = _socket.gethostname().split(".")[0]


def _run_dir() -> str:
  """<dump_dir>/<run_tag>/ -- run_tag = EXPERIMENT_NAME (pushed via ray runtime_env
  by the Phase-0 launcher) or this process's first-call date+hour as fallback."""
  global _boot_tag
  tag = os.environ.get("EXPERIMENT_NAME")
  if not tag:
    if _boot_tag is None:
      _boot_tag = _dt.datetime.now().strftime("run%m%d_%H%M")
    tag = _boot_tag
  d = os.path.join(_DUMP_DIR, tag)
  os.makedirs(d, exist_ok=True)
  return d


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
  """verl custom reward entry point.

  Returns {"score": fmt + ans, "acc", "fmt", "qid", "mv_used", "mv_timeout", "mv_exc", "mv_lenrej"}.
  "score" is the training reward (identical to the original scalar return).
  """
  del kwargs
  completion = solution_str if isinstance(solution_str, str) else str(solution_str)
  fmt = _format_score(completion)
  ans, mvf = _answer_score(completion, ground_truth)
  acc_flag = 1.0 if ans >= REWARD_EXACT_ANSWER else 0.0
  fmt_flag = 1.0 if fmt > 0 else 0.0
  qid = float(extra_info.get("index", -1)) if isinstance(extra_info, dict) else -1.0

  if _DUMP_DIR:
    global _call_count
    try:  # fail-open: a logging bug must never affect the run
      with _lock:
        c = _call_count
        _call_count += 1
        if (c % _DUMP_EVERY) < _DUMP_PER_STEP:
          n_tok = None
          if isinstance(extra_info, dict):
            n_tok = extra_info.get("num_response_tokens") or extra_info.get("response_length")
          rec = {"call": c, "data_source": str(data_source), "fmt": fmt, "ans": ans,
                 "n_chars": len(completion), "n_tokens": n_tok,
                 "gt": str(ground_truth)[:80], "tail": completion[-160:]}
          fn = os.path.join(_run_dir(), f"samples_{_host}_pid{os.getpid()}.jsonl")
          with open(fn, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except (OSError, AttributeError):
      pass
  # qid = extra_info["index"] (verl repeats it for the n=8 completions of a prompt)
  # so the rollout dump can be grouped exactly; mv_* are per-call outcome flags.
  return {"score": fmt + ans, "acc": acc_flag, "fmt": fmt_flag, "qid": qid, **mvf}


# --------------------------- self-test -------------------------------------
if __name__ == "__main__":
  gt = json.dumps(["72", "72"])
  cases = [
      # (completion, expected score, expected acc, note)
      ("<reasoning>2*36</reasoning><answer>72</answer>", 1.1, 1.0, "exact + format"),
      ("<reasoning>2*36</reasoning><answer>\\boxed{72}</answer>", 1.1, 1.0, "boxed inside tags"),
      ("blah <answer> 72 </answer>", 1.0, 1.0, "whitespace match, no format"),
      ("<answer>36*2</answer>", 1.0, 1.0, "math_verify equivalence, no format"),
      ("<reasoning>hmm</reasoning><answer>71</answer>", 0.1, 0.0, "format only, wrong"),
      ("no tags at all 72", 0.0, 0.0, "fallback -> FALLBACK_ANSWER -> 0"),
      ("<answer>7.2e1</answer>", 1.0, 1.0, "math_verify numeric forms"),
  ]
  for completion, exp_score, exp_acc, note in cases:
    got = compute_score("x", completion, gt, extra_info={"index": 0})
    ok = abs(got["score"] - exp_score) < 1e-9 and abs(got["acc"] - exp_acc) < 1e-9
    print(f"{'OK ' if ok else 'FAIL'} {note}: got={got} expected score={exp_score} acc={exp_acc}")