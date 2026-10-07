#!/usr/bin/env python3
"""Read every weight shard of a model on every GPU node, with R concurrent readers per node (like the R trainer ranks
that each load the full model when embeddings are tied). Reports per-node seconds; exits 1 if any node errors or does
not finish within --timeout. Also warms the gcsfuse cache on each node.
usage: prefetch_model.py /workspace/meta-RL/models/Qwen3-4B [--readers 4] [--timeout 900]"""
import argparse, sys, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--readers", type=int, default=4); ap.add_argument("--timeout", type=int, default=900)
a = ap.parse_args()
ray.init(address="auto", logging_level="ERROR")

@ray.remote(num_cpus=1)
def read_all(d):
    import json, os, time
    t0 = time.time(); idx = f"{d}/model.safetensors.index.json"
    shards = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    n = 0
    for s in shards:
        with open(f"{d}/{s}", "rb") as f:
            while (b := f.read(64 << 20)): n += len(b)
    return round(n / 1e9, 2), round(time.time() - t0, 1)

nodes = sorted((n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0), key=lambda n: n["NodeManagerAddress"])
refs = {read_all.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(a.model): n["NodeManagerAddress"]
        for n in nodes for _ in range(a.readers)}
done, pending = ray.wait(list(refs), num_returns=len(refs), timeout=a.timeout)
res = {}
for r in done:
    try: gb, sec = ray.get(r); res.setdefault(refs[r], []).append(f"{sec}s")
    except Exception as e: res.setdefault(refs[r], []).append(f"ERR {type(e).__name__}")
slow = {refs[r] for r in pending}
for r in pending: ray.cancel(r, force=True)
bad = False
for ip in sorted(set(refs.values())):
    v = res.get(ip, []); flag = "TIMEOUT" if ip in slow else ("ERROR" if any(x.startswith("ERR") for x in v) else "")
    bad |= bool(flag); print(f"{ip:15s} {' '.join(v):60s} {flag}")
print(f"{a.model}: {'FAILED' if bad else 'OK'} on {len(nodes)} nodes x {a.readers} readers")
sys.exit(1 if bad else 0)
