"""Copy node-local TensorBoard evidence from the existing Ray nodes after a run."""
import argparse
import os
from pathlib import Path


def main():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)

    @ray.remote(num_cpus=0)
    def read_events(source):
        root = Path(source)
        return [(str(f.relative_to(root)), f.read_bytes())
                for f in root.rglob("events.out.tfevents.*") if f.is_file()]

    try:
        nodes = [n for n in ray.nodes() if n["Alive"]]
        refs = [read_events.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
                .remote(args.source) for n in nodes]
        payloads = ray.get(refs, timeout=120)
        count = 0
        for node, files in zip(nodes, payloads):
            for relative, data in files:
                dest = Path(args.out) / node["NodeID"] / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                count += 1
        if count == 0:
            raise RuntimeError(f"No TensorBoard events found at {args.source} on any live node")
        print(f"Copied {count} event files to {args.out}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
