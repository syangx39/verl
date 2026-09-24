"""Verify finite update metrics and active TIS after the 3-step cluster smoke."""
import argparse
import math
from pathlib import Path


def main():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args()
    root = Path(a.run_dir)
    assert (root / "exit_code.txt").read_text().strip() == "0", "Training driver did not exit successfully"
    scalars = {}
    for event_file in sorted((root / "tensorboard").rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            target = scalars.setdefault(tag, {})
            target.update({e.step: e.value for e in accumulator.Scalars(tag)})
    required = ["actor/grad_norm", "actor/pg_loss", "actor/entropy_loss",
                "actor/rollout_corr/rollout_is_mean",
                "training/off_policy/trajectory_staleness/mean"]
    for tag in required:
        values = scalars.get(tag, {})
        assert all(s in values for s in range(1, a.steps + 1)), f"Missing steps/tag: {tag}"
        assert all(math.isfinite(v) for v in values.values()), f"Non-finite metric: {tag}"
        print(tag, [round(values[s], 7) for s in range(1, a.steps + 1)])
    assert all(v > 0 for v in scalars["actor/grad_norm"].values()), "Zero gradient in smoke"
    print("SMOKE OK: finite gradients/loss/entropy, active TIS, staleness recorded; not a convergence test")


if __name__ == "__main__":
    main()
