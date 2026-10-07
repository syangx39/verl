#!/usr/bin/env python3
"""Copy a directory that exists on one Ray node's local disk (e.g. TensorBoard events written to /tmp by the trainer
process, which Ray may place on any node) to a shared path (gcsfuse), by checking every alive node.
usage: collect_dir.py SRC DEST     (safe to run during a run: it copies whatever has been written so far)"""
import sys, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

src, dest = sys.argv[1], sys.argv[2]
ray.init(address="auto", logging_level="ERROR")

@ray.remote(num_cpus=1)
def collect(src, dest):
    import os, shutil, socket
    if not os.path.isdir(src):
        return None
    files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs]
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return socket.gethostname(), len(files), sum(os.path.getsize(f) for f in files)

nodes = [n for n in ray.nodes() if n["Alive"]]
res = ray.get([collect.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(src, dest) for n in nodes])
found = [r for r in res if r]
for host, n, size in found:
    print(f"[collect] {host}: {n} files, {size / 1e6:.1f} MB  {src} -> {dest}")
if not found:
    print(f"[collect] {src} not found on any of {len(nodes)} nodes"); sys.exit(1)
