"""Check fresh V1 updates; actor/lr is logged AFTER the scheduler advances."""
import argparse
import json
import math
import statistics
from pathlib import Path


def expected_lr(optim, scheduler_step):
    """Pinned FSDP cosine LambdaLR, including short smoke runs inside warmup."""
    assert optim["lr_scheduler_type"] == "cosine"
    total = int(optim["total_training_steps"])
    warmup = int(optim["lr_warmup_steps"])
    if warmup <= 0:
        warmup = int(float(optim.get("lr_warmup_steps_ratio", 0)) * total)
    current = scheduler_step + (0 if optim["zero_indexed_step"] else 1)
    if current < warmup:
        factor = current / max(1, warmup)
    else:
        floor = float(optim.get("min_lr_ratio") or 0)
        progress = (current - warmup) / max(1, total - warmup)
        wave = math.cos(math.pi * float(optim.get("num_cycles", 0.5)) * 2 * progress)
        factor = max(floor, wave * (1 - floor) / 2 + (1 + floor) / 2)
    return float(optim["lr"]) * factor


def check_metrics(scalars, config, steps, max_worst_lag=None, require_no_drops=False):
    assert steps > 0, "--steps must be positive"
    trainer = config["trainer"]
    optim = config["actor_rollout_ref"]["actor"]["optim"]
    assert trainer["resume_mode"] == "disable", "Checker expects a fresh run"
    assert trainer["v1"]["separate_async"]["parameter_sync_step"] == 1
    assert optim["total_training_steps"] == trainer["total_training_steps"], "LR horizon mismatch"
    assert steps <= trainer["total_training_steps"] and optim["zero_indexed_step"] is True
    prefix = "training/off_policy/"
    tis = "actor/rollout_corr/"
    required = ["actor/grad_norm", "actor/pg_loss", "actor/entropy_loss",
                "actor/lr", tis + "rollout_is_mean", tis + "rollout_is_ratio_fraction_high"]
    required += [prefix + metric + "/" + stat
                 for metric in ("trajectory_staleness", "trajectory_staleness_worst", "trajectory_spans")
                 for stat in ("min", "mean", "max")]
    checked = {}
    for tag in required:
        values = scalars.get(tag, {})
        assert all(s in values for s in range(1, steps + 1)), f"Missing steps/tag: {tag}"
        checked[tag] = [float(values[s]) for s in range(1, steps + 1)]
        assert all(math.isfinite(v) for v in checked[tag]), f"Non-finite metric: {tag}"
    assert all(v > 0 for v in checked["actor/grad_norm"]), "Zero gradient in smoke"
    beta = float(config["algorithm"]["rollout_correction"]["rollout_is_threshold"])
    assert all(0 < v <= beta + 1e-5 for v in checked[tis + "rollout_is_mean"]), "Invalid TIS mean"
    assert all(0 <= v <= 1 for v in checked[tis + "rollout_is_ratio_fraction_high"]), "Invalid truncation fraction"
    lr = []
    for step, logged in enumerate(checked["actor/lr"], 1):
        want = expected_lr(optim, step)
        assert math.isclose(logged, want, rel_tol=2e-5, abs_tol=1e-12), \
            f"step {step}: post-scheduler actor/lr={logged:.9g}, expected {want:.9g}"
        lr.append({"step": step, "update_lr_expected": expected_lr(optim, step - 1),
                   "logged_next_update_lr": logged})

    # Pinned replay_buffer emits these counters ONLY when groups are evicted.
    # Units are prompt groups; reason sets can overlap, so do not add their totals.
    evictions = {}
    for tag in (prefix + "evicted_samples", "training/filter_groups/evicted_samples",
                "training/rollout_failure/evicted_samples"):
        values = [float(scalars.get(tag, {}).get(s, 0)) for s in range(1, steps + 1)]
        assert all(math.isfinite(v) and v >= 0 and v.is_integer() for v in values), f"Invalid count: {tag}"
        evictions[tag] = {"per_step": values, "total_prompt_groups": int(sum(values))}
    has_drops = any(v["total_prompt_groups"] for v in evictions.values())
    worst = max(checked[prefix + "trajectory_staleness_worst/max"])
    span = max(checked[prefix + "trajectory_spans/max"])
    assert min(checked[prefix + "trajectory_spans/min"]) >= 1, "Invalid span"
    assert min(checked[prefix + "trajectory_staleness/min"]) >= 0, "Invalid staleness"
    if max_worst_lag is not None:
        assert worst <= max_worst_lag, f"Worst lag {worst} exceeds requested gate {max_worst_lag}"
    if require_no_drops:
        assert not has_drops, f"Nonzero eviction counts: {evictions}"
    warnings = []
    if worst > 1:
        warnings.append(f"Oldest segment lag reaches {worst:g}; inspect before formal training.")
    if has_drops:
        warnings.append("Groups were evicted; inspect counts/reasons and wasted generation before formal training.")
    if any(abs(v - 1) > 0.1 for v in checked[tis + "rollout_is_mean"]):
        warnings.append("TIS mean differs from 1 by >0.1 (diagnostic warning, not an acceptance gate).")
    timings = {}
    for tag in ("timing_s/gen", "timing_s/update_actor", "timing_s/update_weights", "timing_s/save_checkpoint"):
        values = {s: float(scalars[tag][s]) for s in range(1, steps + 1) if s in scalars.get(tag, {})}
        if values:
            assert all(math.isfinite(v) and v >= 0 for v in values.values()), f"Invalid timing: {tag}"
            timings[tag] = {"by_step": values, "mean": statistics.mean(values.values())}
    return {"status": "passed", "checked_steps": steps, "metrics": checked, "lr": lr,
            "evictions": evictions, "timings_seconds": timings, "warnings": warnings,
            "max_worst_lag": worst, "max_trajectory_span": span,
            "notes": ["actor/lr is the next-update LR; update_lr_expected is derived, not separately measured.",
                      "ratio_fraction_high is the pre-cap ratio > beta fraction; zero is allowed.",
                      "Missing eviction counters mean zero in the pinned implementation; counts are prompt groups.",
                      "Eviction reasons can overlap. Mixed-version partial rollouts are allowed.",
                      "Span=max_version-min_version+1, not an exact histogram of token versions.",
                      "timing_s/gen is trainer sampling wait, not total sampler generation time."]}


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from omegaconf import OmegaConf
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max-worst-lag", type=float, help="Optional strict oldest-segment lag bound")
    p.add_argument("--require-no-drops", action="store_true", help="Optional stricter zero-eviction gate")
    a = p.parse_args()
    root = Path(a.run_dir)
    assert a.max_worst_lag is None or a.max_worst_lag >= 0
    assert (root / "exit_code.txt").read_text().strip() == "0", "Training driver did not exit successfully"
    config = OmegaConf.to_container(OmegaConf.load(root / "resolved_config.yaml"), resolve=True)
    scalars = {}
    for event_file in sorted((root / "tensorboard").rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            target = scalars.setdefault(tag, {})
            target.update({e.step: e.value for e in accumulator.Scalars(tag)})
    try:
        report = check_metrics(scalars, config, a.steps, a.max_worst_lag, a.require_no_drops)
    except Exception as exc:
        (root / "check_smoke.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n")
        raise
    (root / "check_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    for tag, values in report["metrics"].items():
        print(tag, [round(v, 8) for v in values])
    print("LR (update-used expected / post-scheduler logged):", report["lr"])
    print("Evictions (prompt groups; absent counters=0):", report["evictions"])
    print("Available timings (seconds):", report["timings_seconds"])
    for warning in report["warnings"]:
        print("WARNING:", warning)
    print("SMOKE OK: update/LR/TIS/staleness checks passed; not a convergence test. Saved check_smoke.json")


if __name__ == "__main__":
    main()
