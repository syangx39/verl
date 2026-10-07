#!/usr/bin/env python3
"""Copy a model directory from the shared gcsfuse mount to node-local storage on every GPU node, with large sequential
reads (which gcsfuse serves fast) instead of the small random mmap reads that safetensors loading issues through FUSE.
Skips files already present with the right size; verifies sizes; prints per-node free space and copy time.
usage:  stage_model.py SRC DEST --check      # only report free space on /tmp and /dev/shm per node
        stage_model.py SRC DEST [--threads 4]"""
import argparse, sys, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dest")
ap.add_argument("--check", action="store_true"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
ray.init(address="auto", logging_level="ERROR")

@ray.remote(num_cpus=4)
def stage(src, dest, check, threads):
    import os, shutil, socket, time
    from concurrent.futures import ThreadPoolExecutor
    def free_gb(p):
        os.makedirs(p, exist_ok=True); st = os.statvfs(p); return round(st.f_bavail * st.f_frsize / 1e9, 1)
    files = sorted(f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)))
    need = sum(os.path.getsize(os.path.join(src, f)) for f in files)
    info = {"host": socket.gethostname(), "need_GB": round(need / 1e9, 1), "free_tmp_GB": free_gb("/tmp"), "free_shm_GB": free_gb("/dev/shm")}
    if check:
        return info
    parent = os.path.dirname(dest.rstrip("/")) or "/"
    have = sum(os.path.getsize(os.path.join(dest, f)) for f in files if os.path.exists(os.path.join(dest, f)))
    if free_gb(parent) * 1e9 < (need - have) * 1.05:
        raise RuntimeError(f"not enough space under {parent}: {info}")
    os.makedirs(dest, exist_ok=True); t0 = time.time()
    def cp(f):
        s, d = os.path.join(src, f), os.path.join(dest, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            return 0
        with open(s, "rb") as fi, open(d + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo, length=64 << 20)
        os.replace(d + ".part", d); return os.path.getsize(d)
    with ThreadPoolExecutor(threads) as ex:
        copied = sum(ex.map(cp, files))
    bad = [f for f in files if os.path.getsize(os.path.join(dest, f)) != os.path.getsize(os.path.join(src, f))]
    if bad:
        raise RuntimeError(f"size mismatch after copy: {bad}")
    info.update(copied_GB=round(copied / 1e9, 1), sec=round(time.time() - t0, 1))
    return info

nodes = sorted((n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0), key=lambda n: n["NodeManagerAddress"])
refs = [(n["NodeManagerAddress"], stage.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)).remote(a.src, a.dest, a.check, a.threads)) for n in nodes]
ok = True
for ip, r in refs:
    try:
        print(f"{ip:15s} {ray.get(r)}")
    except Exception as e:
        ok = False; print(f"{ip:15s} FAILED {str(e)[:200]}")
print(("CHECK" if a.check else "STAGED") + (" OK" if ok else " FAILED") + f" on {len(nodes)} nodes")
sys.exit(0 if ok else 1)
