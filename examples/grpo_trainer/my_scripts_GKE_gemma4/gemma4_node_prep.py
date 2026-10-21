#!/usr/bin/env python3
"""Prepare every Ray node for the Gemma-4 recipe and record per-node provenance (run on the Ray head before main_ppo).

One task per alive Ray node, pinned with a hard NodeAffinitySchedulingStrategy (soft=False). Each task:
  1. Pinned packages: math-verify and its two runtime deps at the versions validated for this recipe
     (PINNED_PACKAGES). Installed with `uv pip install` only if the installed versions differ.
  2. Managed verl files, rebuilt from the pin (`git show HEAD:<file>` in the node's verl checkout) plus the managed
     patchers, and written only if the live file differs:
       verl/models/transformers/monkey_patch.py      <- patch_verl_gemma4_flex_attn.py, patch_verl_gemma4_rope_buffers.py
       verl/workers/engine/fsdp/transformer_impl.py  <- patch_verl_gemma4_rope_buffers.py (init-checkpoint hook)
     Any other tracked file that differs from the pin (e.g. a diagnostic patch) is restored with `git checkout`, so
     every node runs exactly "pin + managed patches" and the managed files are byte-identical across nodes, whatever
     state (unpatched, v1, diagnostics) the pod was in.
  3. Import check in a fresh interpreter of the Ray worker's python: module file paths, [GEMMA4-ROPE] patch version
     and block sha256, hook present in FSDPEngine._build_model_optimizer, flex patch present, package versions.
The patcher scripts are read once here on the head (sha256 recorded) and shipped to the tasks by value, so worker pods
never read them from the FUSE mount. The driver requires a result from every alive node (and from the node it was
pinned to), ok on all, and identical python / verl HEAD / managed-file sha256 / module paths / package versions
across nodes. It writes a JSON report and exits 0 only on PASS.

Usage (the launcher does this): python3 gemma4_node_prep.py --report <guard_dir>/node_prep.json [--min-gpu-nodes N]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

MANAGED = ("verl/models/transformers/monkey_patch.py", "verl/workers/engine/fsdp/transformer_impl.py")
FLEX_PATCHER = "patch_verl_gemma4_flex_attn.py"
ROPE_PATCHER = "patch_verl_gemma4_rope_buffers.py"
# math-verify as validated for this recipe (README section 7) plus its runtime deps at the versions on the validated pods
PINNED_PACKAGES = {"math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0", "antlr4-python3-runtime": "4.9.3"}
REPORT_PACKAGES = ["torch", "transformers", "vllm", "ray", "flash-attn"] + list(PINNED_PACKAGES)
CONSISTENT = ("python", "verl_root", "verl_head", "managed_sha256", "modified_files", "import.python",
              "import.verl_file", "import.monkey_patch_file", "import.transformer_impl_file",
              "import.rope_patch_version", "import.rope_block_sha256", "import.packages")

IMPORT_CHECK = r'''
import hashlib, importlib, importlib.metadata as md, inspect, json, sys
pk_names = json.loads(sys.argv[1])
out = {"python": sys.executable, "python_version": sys.version.split()[0]}
import verl
out["verl_file"] = verl.__file__
mp = importlib.import_module("verl.models.transformers.monkey_patch")
out["monkey_patch_file"] = mp.__file__
out["rope_patch_version"] = getattr(mp, "GEMMA4_ROPE_PATCH_VERSION", None)
out["has_flex_patch"] = callable(getattr(mp, "_verl_patch_flex_kernel_options", None))
out["has_init_checkpoint"] = callable(getattr(mp, "_verl_gemma4_rope_init_checkpoint", None))
out["apply_monkey_patch_wrapped"] = getattr(mp, "_verl_orig_apply_monkey_patch", None) is not None and mp.apply_monkey_patch is not mp._verl_orig_apply_monkey_patch
src = open(mp.__file__, encoding="utf-8").read()
lines = src.split("\n")
b = [i for i, l in enumerate(lines) if l.startswith("# ===== [GEMMA4-ROPE] begin (version ")]
e = [i for i, l in enumerate(lines) if l == "# ===== [GEMMA4-ROPE] end ====="]
out["rope_blocks"] = len(b)
out["rope_block_sha256"] = hashlib.sha256(("\n".join(lines[b[0]:e[0] + 1]) + "\n").encode()).hexdigest() if len(b) == 1 and len(e) == 1 else None
ti = importlib.import_module("verl.workers.engine.fsdp.transformer_impl")
out["transformer_impl_file"] = ti.__file__
out["hook_in_build_model_optimizer"] = "_g4_ckpt(module, forward_only=self.engine_config.forward_only)" in inspect.getsource(ti.FSDPEngine._build_model_optimizer)
out["engine_uses_wrapped_apply_monkey_patch"] = ti.apply_monkey_patch is mp.apply_monkey_patch
pk = {}
for p in pk_names:
    try:
        pk[p] = md.version(p)
    except md.PackageNotFoundError:
        pk[p] = None
out["packages"] = pk
print("NODEPREP-IMPORT " + json.dumps(out, sort_keys=True))
'''


def prep_node(payload):
    """Runs inside a Ray task on one node (shipped by value). Returns a JSON-serializable per-node record."""
    import hashlib
    import importlib.metadata as md
    import importlib.util
    import json
    import os
    import shutil
    import socket
    import subprocess
    import sys
    import tempfile
    import time

    import ray

    t0 = time.time()
    res = {"hostname": socket.gethostname(), "python": sys.executable, "pid": os.getpid(), "actions": [], "errors": []}

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def run(cmd, binary=False, timeout=600):
        p = subprocess.run(cmd, capture_output=True, text=not binary, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def versions(names):
        v = {}
        for n in names:
            try:
                v[n] = md.version(n)
            except md.PackageNotFoundError:
                v[n] = None
        return v

    try:
        res["node_id"] = ray.get_runtime_context().get_node_id()
        res["node_ip"] = ray.util.get_node_ip_address()
        spec = importlib.util.find_spec("verl")
        if spec is None or not spec.origin:
            raise RuntimeError("verl is not importable by the Ray worker python")
        root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
        res["verl_root"] = root
        rc, out, err = run(["git", "-C", root, "rev-parse", "HEAD"])
        if rc:
            raise RuntimeError(f"git rev-parse failed: {err.strip()}")
        res["verl_head"] = out.strip()
        pin_file = "/workspace/VERL_PIN"
        res["image_verl_pin"] = open(pin_file).read().strip() if os.path.exists(pin_file) else None

        # 1) pinned packages (install only if not already satisfied)
        pinned = payload["pinned"]
        before = versions(pinned)
        res["pinned_before"] = before
        if before != pinned:
            uv = shutil.which("uv")
            if not uv:
                raise RuntimeError("uv not found; cannot install the pinned packages")
            specs = [f"{k}=={v}" for k, v in pinned.items()]
            rc, out, err = run([uv, "pip", "install", "--python", sys.executable, "--quiet"] + specs)
            res["actions"].append({"action": "pip_install", "specs": specs, "rc": rc, "stderr_tail": err[-600:]})
            if rc:
                raise RuntimeError(f"uv pip install failed (rc={rc}): {err[-400:]}")
        else:
            res["actions"].append({"action": "pinned_packages_already_satisfied"})

        # 2) managed files = pin + managed patchers, written only if different
        with tempfile.TemporaryDirectory(prefix="g4prep_") as td:
            pdir = os.path.join(td, "_patchers")
            os.makedirs(pdir)
            for name, text in payload["patchers"].items():
                with open(os.path.join(pdir, name), "w", encoding="utf-8") as f:
                    f.write(text)
            res["pristine_sha256"] = {}
            for rel in payload["managed"]:
                rc, blob, err = run(["git", "-C", root, "show", f"HEAD:{rel}"], binary=True)
                if rc:
                    raise RuntimeError(f"git show HEAD:{rel} failed: {err.decode(errors='replace').strip()}")
                dst = os.path.join(td, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(blob)
                res["pristine_sha256"][rel] = sha(blob)
            res["patcher_output"] = []
            for cmd in ([sys.executable, os.path.join(pdir, payload["flex_patcher"]), os.path.join(td, payload["managed"][0])],
                        [sys.executable, os.path.join(pdir, payload["rope_patcher"]), "--verl-root", td]):
                rc, out, err = run(cmd)
                res["patcher_output"].append([l for l in out.splitlines() if "RESULT" not in l][-3:])
                if rc:
                    raise RuntimeError(f"{os.path.basename(cmd[1])} failed (rc={rc}): {out[-300:]} {err[-300:]}")
            for rel in payload["managed"]:
                with open(os.path.join(td, rel), "rb") as f:
                    want = f.read()
                live_path = os.path.join(root, rel)
                with open(live_path, "rb") as f:
                    live = f.read()
                entry = {"file": rel, "sha256_before": sha(live), "sha256_expected": sha(want)}
                if live != want:
                    tmp = live_path + ".g4prep.tmp"
                    with open(tmp, "wb") as f:
                        f.write(want)
                    os.chmod(tmp, os.stat(live_path).st_mode & 0o7777)
                    os.replace(tmp, live_path)
                    cache = os.path.join(os.path.dirname(live_path), "__pycache__")
                    stem = os.path.splitext(os.path.basename(live_path))[0] + "."
                    if os.path.isdir(cache):
                        for fn in os.listdir(cache):
                            if fn.startswith(stem) and fn.endswith(".pyc"):
                                os.remove(os.path.join(cache, fn))
                    entry["action"] = "rebuilt"
                else:
                    entry["action"] = "unchanged"
                res["actions"].append(entry)

        # 3) restore any other tracked file that differs from the pin
        rc, out, err = run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
        if rc:
            raise RuntimeError(f"git status failed: {err.strip()}")
        for line in out.splitlines():
            path = line[3:].strip()
            if path and path not in payload["managed"]:
                rc2, _, err2 = run(["git", "-C", root, "checkout", "--", path])
                res["actions"].append({"file": path, "action": "restored_to_pin", "rc": rc2})
                if rc2:
                    raise RuntimeError(f"git checkout -- {path} failed: {err2.strip()}")

        # 4) final tree state (tracked files that differ from the pin, with sha256)
        rc, out, err = run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"])
        mods = {}
        for line in out.splitlines():
            path = line[3:].strip()
            with open(os.path.join(root, path), "rb") as f:
                mods[path] = sha(f.read())
        res["modified_files"] = mods
        res["managed_sha256"] = {rel: mods.get(rel) for rel in payload["managed"]}
        if sorted(mods) != sorted(payload["managed"]):
            res["errors"].append(f"tracked files differing from the pin: {sorted(mods)} (expected exactly the managed set)")

        # 5) import check in a fresh interpreter (same python as the trainer workers)
        rc, out, err = run([sys.executable, "-c", payload["import_check"], json.dumps(payload["report_packages"])],
                           timeout=600)
        lines = [l for l in out.splitlines() if l.startswith("NODEPREP-IMPORT ")]
        if rc or not lines:
            raise RuntimeError(f"import check failed (rc={rc}): {err[-800:]}")
        imp = json.loads(lines[-1][len("NODEPREP-IMPORT "):])
        res["import"] = imp
        expect = {
            "rope_patch_version": payload["rope_version"],
            "rope_block_sha256": payload["rope_block_sha256"],
            "rope_blocks": 1,
            "has_flex_patch": True,
            "has_init_checkpoint": True,
            "apply_monkey_patch_wrapped": True,
            "hook_in_build_model_optimizer": True,
            "engine_uses_wrapped_apply_monkey_patch": True,
            "monkey_patch_file": os.path.join(root, payload["managed"][0]),
            "transformer_impl_file": os.path.join(root, payload["managed"][1]),
        }
        for k, v in expect.items():
            if imp.get(k) != v:
                res["errors"].append(f"import check: {k}={imp.get(k)!r}, expected {v!r}")
        for k, v in pinned.items():
            if imp["packages"].get(k) != v:
                res["errors"].append(f"package {k}=={imp['packages'].get(k)}, expected {v}")
    except Exception as e:
        res["errors"].append(f"{type(e).__name__}: {e}")
    res["ok"] = not res["errors"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--report", required=True, help="output JSON report")
    ap.add_argument("--min-gpu-nodes", type=int, default=1, help="fail unless at least this many alive nodes have GPUs")
    ap.add_argument("--launcher-sha256", default=None, help="recorded in the report")
    ap.add_argument("--timeout", type=int, default=900, help="seconds for all node tasks")
    args = ap.parse_args()

    me = os.path.abspath(__file__)
    with open(me, "rb") as f:
        self_sha = hashlib.sha256(f.read()).hexdigest()
    patchers = {}
    for name in (FLEX_PATCHER, ROPE_PATCHER):
        with open(os.path.join(args.patch_dir, name), encoding="utf-8") as f:
            patchers[name] = f.read()
    patcher_sha = {n: hashlib.sha256(t.encode("utf-8")).hexdigest() for n, t in patchers.items()}
    ver = json.loads(subprocess.check_output([sys.executable, os.path.join(args.patch_dir, ROPE_PATCHER),
                                              "--print-version"], text=True))
    payload = {"patchers": patchers, "managed": list(MANAGED), "flex_patcher": FLEX_PATCHER,
               "rope_patcher": ROPE_PATCHER, "pinned": PINNED_PACKAGES, "report_packages": REPORT_PACKAGES,
               "rope_version": ver["version"], "rope_block_sha256": ver["block_sha256"], "import_check": IMPORT_CHECK}

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    t0 = time.time()
    ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", namespace="gemma4_node_prep", log_to_driver=True)
    nodes = sorted((n for n in ray.nodes() if n.get("Alive")), key=lambda n: n["NodeManagerAddress"])
    gpu_nodes = [n for n in nodes if n.get("Resources", {}).get("GPU", 0) > 0]
    task = ray.remote(num_cpus=0, max_retries=0)(prep_node)
    refs = {}
    for n in nodes:
        strategy = NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)
        refs[n["NodeID"]] = task.options(scheduling_strategy=strategy, name=f"g4prep_{n['NodeManagerAddress']}").remote(payload)
    results, deadline = {}, time.time() + args.timeout
    for nid, ref in refs.items():
        try:
            results[nid] = ray.get(ref, timeout=max(1.0, deadline - time.time()))
        except Exception as e:
            results[nid] = {"node_id": nid, "ok": False, "errors": [f"task failed: {type(e).__name__}: {e}"]}
    ray.shutdown()

    errors = []
    by_ip = {n["NodeID"]: n["NodeManagerAddress"] for n in nodes}
    for nid, r in results.items():
        if r.get("node_id") != nid:
            r["ok"] = False
            r.setdefault("errors", []).append(f"ran on node {r.get('node_id')} instead of {nid}")
        if r.get("image_verl_pin") and r.get("verl_head") != r.get("image_verl_pin"):
            r["ok"] = False
            r["errors"].append(f"verl HEAD {r.get('verl_head')} != image pin {r.get('image_verl_pin')}")
    bad = {by_ip[nid]: r.get("errors") for nid, r in results.items() if not r.get("ok")}
    if bad:
        errors.append(f"{len(bad)} node(s) failed: {bad}")
    if len(gpu_nodes) < args.min_gpu_nodes:
        errors.append(f"only {len(gpu_nodes)} alive GPU node(s), need {args.min_gpu_nodes}")
    consistency = {}
    for key in CONSISTENT:
        vals = {json.dumps(_get(r, key), sort_keys=True) for r in results.values() if r.get("ok")}
        consistency[key] = {"n_distinct": len(vals), "value": json.loads(next(iter(vals))) if len(vals) == 1 else None}
        if len(vals) > 1:
            errors.append(f"{key} differs across nodes: {sorted(vals)[:4]}")
    verdict = "PASS" if not errors and results and len(results) == len(nodes) else "FAIL"
    report = {
        "kind": "gemma4_node_prep", "verdict": verdict, "errors": errors, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1), "node_prep_sha256": self_sha, "launcher_sha256": args.launcher_sha256,
        "patch_dir": os.path.abspath(args.patch_dir), "patcher_sha256": patcher_sha, "rope_patch": ver,
        "pinned_packages": PINNED_PACKAGES, "n_nodes": len(nodes), "n_gpu_nodes": len(gpu_nodes),
        "node_ips": [n["NodeManagerAddress"] for n in nodes], "consistency": consistency,
        "nodes": [dict(results[n["NodeID"]], ray_node_ip=n["NodeManagerAddress"]) for n in nodes],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    print(f"{'node_ip':<14} {'hostname':<52} {'monkey_patch':<22} {'transformer_impl':<22} {'restored':>8} {'pip':<10} ok")
    for n in nodes:
        r = results[n["NodeID"]]
        acts = {a.get("file"): a for a in r.get("actions", []) if a.get("file")}
        mp, ti = acts.get(MANAGED[0], {}), acts.get(MANAGED[1], {})
        restored = sum(1 for a in r.get("actions", []) if a.get("action") == "restored_to_pin")
        pip = "installed" if any(a.get("action") == "pip_install" for a in r.get("actions", [])) else "satisfied"
        print(f"{n['NodeManagerAddress']:<14} {r.get('hostname', '?'):<52} "
              f"{mp.get('action', '?') + ' ' + (mp.get('sha256_expected') or '')[:10]:<22} "
              f"{ti.get('action', '?') + ' ' + (ti.get('sha256_expected') or '')[:10]:<22} {restored:>8} {pip:<10} "
              f"{'ok' if r.get('ok') else 'FAIL'}")
    sha_mp = (consistency.get("managed_sha256", {}).get("value") or {}).get(MANAGED[0]) or "?"
    sha_ti = (consistency.get("managed_sha256", {}).get("value") or {}).get(MANAGED[1]) or "?"
    print(f"GEMMA4-NODE-PREP: {verdict} nodes={len(nodes)} gpu_nodes={len(gpu_nodes)} rope=v{ver['version']} "
          f"block={ver['block_sha256'][:12]} monkey_patch={sha_mp[:12]} transformer_impl={sha_ti[:12]} "
          f"seconds={report['seconds']} report={args.report}")
    for e in errors:
        print(f"GEMMA4-NODE-PREP ERROR: {e}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
