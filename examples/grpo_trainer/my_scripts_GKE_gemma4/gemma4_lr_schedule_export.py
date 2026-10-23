#!/usr/bin/env python3
"""Export the learning rates of PR #12 arm A4 (native half-cosine) and check them (PR #12 A4 preflight).

verl FSDP (pin ace775e8) builds the schedule in `FSDPEngine._build_lr_scheduler` with
`verl.utils.torch_functional.get_cosine_schedule_with_warmup` (a torch `LambdaLR`) and advances it once per training
step after the optimizer update (`engine.lr_scheduler_step()`, `update_lr_scheduler` on the last mini-batch). The
logged `actor/lr` is `lr_scheduler.get_last_lr()[0]` read *after* that advance, i.e. the LR of the next update:

    update k (1-based) uses   lr(s = k - 1)
    actor/lr logged at step k = lr(s = k)

With num_cycles=0.25, min_lr_ratio=0, no warmup, total_training_steps=100 and zero_indexed_step=True:

    lr(s) = peak * 0.5 * (1 + cos(pi * s / 200))

which equals MaxText RL's optax `warmup_cosine_decay_schedule(init 0, peak, warmup_steps=0, decay_steps=200, end 0)`
evaluated at optimizer count s.

Modes:
  default      pure-python re-implementation of the verl lambda (runs anywhere);
  --use-verl   instantiate verl's own get_cosine_schedule_with_warmup on a dummy torch optimizer and step it exactly
               like the trainer (optimizer.step, then scheduler.step), recording the param-group LR before each update
               (used) and get_last_lr() after the advance (logged). Run this on the training image.

Usage:
    python3 gemma4_lr_schedule_export.py [--use-verl] --out a4_used_lr.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys

PEAK = 2e-6
TOTAL = 100
NUM_CYCLES = 0.25
MIN_LR_RATIO = 0.0
WARMUP = 0
EXPECTED = {0: 2e-6, 40: 1.809016994e-6, 99: 1.015707317e-6, 100: 1e-6}
EXPECTED_MEAN_USED = 1.641606682e-6


def verl_lambda(step: int) -> float:
    """Copy of verl get_cosine_schedule_with_warmup.lr_lambda (zero_indexed_step=True, init_lr_ratio=0)."""
    coef = (1 - MIN_LR_RATIO) * 0.5
    intercept = (1 + MIN_LR_RATIO) * 0.5
    if step < WARMUP:
        return float(step) / float(max(1, WARMUP))
    progress = float(step - WARMUP) / float(max(1, TOTAL - WARMUP))
    x = math.cos(math.pi * float(NUM_CYCLES) * 2.0 * progress)
    return max(MIN_LR_RATIO, x * coef + intercept)


def sequence_pure():
    used = [PEAK * verl_lambda(k - 1) for k in range(1, TOTAL + 1)]
    logged = [PEAK * verl_lambda(k) for k in range(1, TOTAL + 1)]
    return used, logged


def sequence_verl():
    import torch

    from verl.utils.torch_functional import get_cosine_schedule_with_warmup

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=PEAK)
    sched = get_cosine_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=WARMUP,
        num_training_steps=TOTAL,
        min_lr_ratio=MIN_LR_RATIO,
        num_cycles=NUM_CYCLES,
        zero_indexed_step=True,
    )
    used, logged = [], []
    for _ in range(TOTAL):
        used.append(opt.param_groups[0]["lr"])
        p.grad = torch.zeros_like(p)
        opt.step()
        sched.step()
        logged.append(sched.get_last_lr()[0])
    return used, logged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-verl", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    used, logged = sequence_verl() if args.use_verl else sequence_pure()
    checks = {}
    fails = []

    def check(name, got, want, tol=1e-15):
        ok = abs(got - want) <= tol
        checks[name] = {"got": got, "want": want, "ok": ok}
        if not ok:
            fails.append(name)

    check("update_1_uses_lr(0)", used[0], EXPECTED[0])
    check("update_41_uses_lr(40)", used[40], EXPECTED[40], 1e-15)
    check("update_100_uses_lr(99)", used[99], EXPECTED[99], 1e-15)
    check("logged_after_update_100_is_lr(100)", logged[99], EXPECTED[100], 1e-15)
    check("mean_used_lr_s0_99", sum(used) / len(used), EXPECTED_MEAN_USED, 1e-15)
    check("monotone_non_increasing", float(all(a >= b for a, b in zip(used, used[1:], strict=False))), 1.0, 0.0)
    out = {
        "schedule": {
            "peak": PEAK,
            "total_training_steps": TOTAL,
            "num_cycles": NUM_CYCLES,
            "min_lr_ratio": MIN_LR_RATIO,
            "warmup_steps": WARMUP,
            "zero_indexed_step": True,
            "formula": (
                "lr(s) = 2e-6 * 0.5 * (1 + cos(pi * s / 200)); update k uses s = k - 1; "
                "logged actor/lr at step k = lr(k)"
            ),
        },
        "source": "verl get_cosine_schedule_with_warmup on torch"
        if args.use_verl
        else "pure-python copy of the verl lambda",
        "checks": checks,
        "update_used_lr": {str(k): used[k - 1] for k in range(1, TOTAL + 1)},
        "logged_actor_lr": {str(k): logged[k - 1] for k in range(1, TOTAL + 1)},
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    for k, v in checks.items():
        print(f"{'OK  ' if v['ok'] else 'FAIL'} {k}: got {v['got']:.12g} want {v['want']:.12g}")
    print(f"{'PASS' if not fails else 'FAIL'} ({out['source']})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
