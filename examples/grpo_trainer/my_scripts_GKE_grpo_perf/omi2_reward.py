"""Reward after the TPU reference (MaxText gsm8k_rl): per-completion reward = check_numbers + match_format_exactly.
check_numbers (1.0): the answer in the LAST <answer>...</answer> (or last \\boxed{}) equals the gold answer; exact match after light
normalization, else math_verify when available. match_format_exactly (0.1): <reasoning>...</reasoning> ... <answer>...</answer> present.
Reward in {0, 0.1, 1.0, 1.1}. Ground truth may be a JSON list of acceptable strings (OMI2 MaxText parquet) or a plain string."""
import json, re
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except Exception:  # math_verify is optional
    _mv_parse = _mv_verify = None

_ANS = re.compile(r"<answer>(.*?)</answer>", re.S)
_FMT = re.compile(r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>", re.S)

def _last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    j, depth = i + len("\\boxed{"), 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
    return s[i + len("\\boxed{"): j - 1] if depth == 0 else None

def _norm(x):
    x = x.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\,", "").replace("dfrac", "frac").replace("tfrac", "frac")
    x = re.sub(r"\\text\{(.*?)\}", r"\1", x).rstrip(".")
    if x.startswith("\\boxed{") and x.endswith("}"): x = x[7:-1]
    return x

def extract_answer(text):
    m = _ANS.findall(text)
    if m: return m[-1].strip()
    return _last_boxed(text)

def check_numbers(pred, golds):
    if pred is None: return 0.0
    p = _norm(pred)
    for g in golds:
        if p == _norm(g): return 1.0
        try:
            if float(p.replace(",", "")) == float(_norm(g).replace(",", "")): return 1.0
        except ValueError:
            pass
    if _mv_parse is not None:
        for g in golds:
            try:
                if _mv_verify(_mv_parse(f"${g}$"), _mv_parse(f"${pred}$")): return 1.0
            except Exception:
                pass
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw):
    golds = ground_truth
    if isinstance(golds, str):
        try:
            golds = json.loads(golds) if golds.lstrip().startswith("[") else [golds]
        except Exception:
            golds = [golds]
    golds = [str(g) for g in golds]
    acc = check_numbers(extract_answer(solution_str), golds)
    fmt = 0.1 if _FMT.search(solution_str) else 0.0
    return {"score": acc + fmt, "acc": acc, "fmt": fmt}

if __name__ == "__main__":
    tests = [("<reasoning>x</reasoning><answer>18</answer>", '["18","18"]', 1.1), ("blah \\boxed{\\frac{8\\sqrt{3}}{3}}", '["\\\\frac{8\\\\sqrt{3}}{3}"]', 1.0),
             ("<reasoning>..</reasoning><answer>17</answer>", '["18"]', 0.1), ("no answer", '["18"]', 0.0), ("<answer>1,000</answer>", '["1000"]', 1.0)]
    for s, g, exp in tests:
        r = compute_score("x", s, g)["score"]; assert abs(r - exp) < 1e-9, (s, r, exp)
    print("omi2_reward self-test OK; math_verify", "available" if _mv_parse else "not installed (exact/numeric match only)")
