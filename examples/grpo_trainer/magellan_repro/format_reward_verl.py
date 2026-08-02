#!/usr/bin/env python3
"""Format-only <think>/<answer> reward — verl port of Meta's reward_format.py.

Regex and semantics copied VERBATIM (anchored full-match on stripped text,
1.0 iff ^<think>.*?</think>\\s*<answer>.*?</answer>$, else 0.0). Only the
calling convention changes: TRL passes a batch (completions list), verl calls
per-completion with (data_source, solution_str, ground_truth, extra_info).

Expected behavior under this reward (per Meta's own notes): completions tend
to run to the generation cap (clipped_ratio ~ 1.0) since nothing incentivizes
EOS — infrastructure-scaling regime, not representative learning.
"""

import re

FORMAT_RE = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)


def format_reward(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs):
  """verl custom_reward_function entry point."""
  text = solution_str if isinstance(solution_str, str) else str(solution_str or "")
  return 1.0 if FORMAT_RE.match(text.strip()) else 0.0


if __name__ == "__main__":
  good = "<think> reasoning </think><answer> 42 </answer>"
  bad = "the answer is 42"
  multiline = "<think>\nstep 1\nstep 2\n</think>\n<answer>7</answer>"
  prefixed = "Sure! <think>x</think><answer>y</answer>"  # leading prose -> 0
  for name, s, expect in [("good", good, 1.0), ("bad", bad, 0.0),
                          ("multiline", multiline, 1.0), ("prefixed", prefixed, 0.0)]:
    got = format_reward("x", s, "")
    print(f"{'OK ' if got == expect else 'FAIL'} {name}: {got}")
