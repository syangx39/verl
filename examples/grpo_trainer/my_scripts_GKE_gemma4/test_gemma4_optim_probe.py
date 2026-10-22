#!/usr/bin/env python3
"""CPU self-test of gemma4_optim_probe: bf16 vs fp32 parameters under torch.optim.AdamW(lr=1e-6).

Each dtype runs in its own subprocess (the probe counts calls per process). Checks:
  * fp32 params: nearly every element changes, ||applied|| / ||ideal|| ~ 1 and cos ~ 1 (positive control);
  * bf16 params: the changed set equals the round-to-nearest prediction (|ideal| > half ulp) to within ties.
Prints a JSON line per dtype and exits non-zero if a check fails.
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r"""
import json, os, sys, torch
sys.path.insert(0, os.environ["PROBE_PARENT"])
import gemma4_optim_probe as P
P.install()
dtype = getattr(torch, os.environ["PROBE_DTYPE"])
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(512, 512, bias=False), torch.nn.Linear(512, 512, bias=False))
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0.0, 0.02)
model = model.to(dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
for _ in range(2):
    x = torch.randn(64, 512, dtype=dtype)
    loss = model(x).float().pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
"""


def main() -> int:
    ok = True
    for dt in ("float32", "bfloat16"):
        d = tempfile.mkdtemp(prefix=f"optim_probe_test_{dt}_")
        env = dict(os.environ, VERL_OPTIM_PROBE_DIR=d, VERL_OPTIM_PROBE_STEPS="2", PROBE_DTYPE=dt,
                   PROBE_PARENT=HERE, RANK="0", WORLD_SIZE="1")
        subprocess.run([sys.executable, "-c", CHILD], env=env, check=True)
        sys.path.insert(0, HERE)
        import gemma4_optim_probe_report as R

        s = R.summarize(d, 1)
        for k, c in sorted(s["calls"].items()):
            x = c["counterfactual_2x2"]
            res = {"dtype": dt, "call": k, "param_dtype": c["dtypes"]["param"], "exp_avg": c["dtypes"]["exp_avg"],
                   "changed_frac": round(c["changed_frac"], 5), "expect_frac": round(c["expect_survive_frac"], 5),
                   "pred_precision": round(c["changed_and_expected_over_changed"], 4),
                   "pred_recall": round(c["changed_and_expected_over_expected"], 4),
                   "applied_over_ideal": round(c["applied_over_ideal_norm"], 4), "cos": round(c["cos_applied_ideal"], 4),
                   "fb_cos_ideal": round(x["fb_cos_ideal"], 4), "bf_cos_ideal": round(x["bf_cos_ideal"], 4),
                   "applied_cos_fb": round(x["applied_cos_fb"], 4),
                   "bf16_visible": round(c["rollout_visible"]["cast_changed_frac"], 5)}
            print(json.dumps(res))
            if dt == "float32":
                ok &= c["changed_frac"] > 0.99 and abs(c["applied_over_ideal_norm"] - 1) < 0.02 and c["cos_applied_ideal"] > 0.99
                ok &= x["fb_cos_ideal"] > 0.99 and x["bf_cos_ideal"] > 0.99
            else:
                ok &= c["changed_and_expected_over_changed"] > 0.97 and c["changed_and_expected_over_expected"] > 0.97
                ok &= x["applied_cos_fb"] > 0.95 and x["bf_cos_ideal"] > 0.99
        ok &= not s["errors"] and s["init"]["complete"]
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
