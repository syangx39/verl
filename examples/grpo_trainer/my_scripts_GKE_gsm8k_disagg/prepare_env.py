#!/usr/bin/env python3
"""Prepare node-local, isolated environments without changing the running Ray cluster.

Run with the existing Ray image's Python, not with the new recipe environment.
The checkout and recipe bundle must be mounted at the same paths on every node.
"""

import argparse
import datetime as dt
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


COMMIT = "ace775e87d8765bcdd114aac734ab71da5367a0f"
LOCKED_RAY = "2.55.1"


def ambient_profile():
    import ray
    from packaging.version import Version

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Existing Ray Python must be 3.12; found {sys.version.split()[0]}")
    if Version(ray.__version__) < Version("2.41.0"):
        raise RuntimeError(f"verl requires Ray >= 2.41; found {ray.__version__}")
    if importlib.util.find_spec("ray._private.runtime_env.py_executable") is None:
        raise RuntimeError("This Ray build lacks the py_executable runtime-env plugin")
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "ray": ray.__version__,
        "ray_commit": getattr(ray, "__commit__", None),
    }


PROBE = r'''
import importlib, json, sys
from importlib.metadata import version
from pathlib import Path
import ray, torch, verl
import transfer_queue
import cupy
import flash_attn
import vllm
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.workers.rollout.vllm_rollout import vllm_async_server
profile = {
    "python": sys.version.split()[0], "python_executable": sys.executable,
    "ray": ray.__version__, "ray_commit": getattr(ray, "__commit__", None),
    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__, "verl_file": str(Path(verl.__file__).resolve()),
    "transferqueue": version("transferqueue"),
    "transformers": version("transformers"),
    "flash_attn": version("flash-attn"), "cupy": cupy.__version__,
}
print("ENV_PROFILE_JSON=" + json.dumps(profile, sort_keys=True))
'''


def prepare_node(node_id, node_ip, options, expected):
    """Runs on one specific Ray node, using its existing ambient interpreter."""
    result = {"node_id": node_id, "node_ip": node_ip, "ok": False}
    log_path = Path(options["log_dir"]) / f"{node_id}.log"
    result["log"] = str(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", buffering=1) as log:
            def run(args, *, env=None, capture=False):
                print("COMMAND " + json.dumps([str(x) for x in args]), file=log)
                if capture:
                    completed = subprocess.run(
                        args, cwd=options["repo"], env=env, text=True,
                        stdout=subprocess.PIPE, stderr=log, check=True,
                    )
                    print(completed.stdout, file=log)
                    return completed.stdout.strip()
                subprocess.run(
                    args, cwd=options["repo"], env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=True,
                )

            ambient = ambient_profile()
            result["ambient"] = ambient
            if ambient["python"] != expected["python"]:
                raise RuntimeError(f"Python differs from driver: {ambient} vs {expected}")
            if (ambient["ray"], ambient["ray_commit"]) != (expected["ray"], expected["ray_commit"]):
                raise RuntimeError(f"Ray differs from driver: {ambient} vs {expected}")
            repo = Path(options["repo"])
            if not Path(options["bundle"]).is_dir():
                raise RuntimeError(f"Shared recipe bundle is missing: {options['bundle']}")
            # Node-local source checkout. A git checkout on a gcsfuse mount is not usable from many nodes at once
            # (stale caches -> spurious diffs, SIGBUS on mmap'd index/pack files, symlinks unrepresentable), and
            # importing verl from gcsfuse on 64 workers is slow. Each node clones the pinned commit into the SAME
            # local path instead; the shared bundle (recipe, reward) stays on the shared mount.
            if options["clone_url"] and not options["check"]:      # --check only verifies; it never clones or moves HEAD
                if not (repo / ".git").is_dir():
                    repo.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["git", "clone", "--depth", "1", "--branch", options["clone_branch"], options["clone_url"], str(repo)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                head_now = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip()
                if head_now != COMMIT:
                    subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", COMMIT], stdout=log, stderr=subprocess.STDOUT, check=True)
                    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", COMMIT], stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (repo / ".git").is_dir():
                raise RuntimeError(f"verl checkout missing on this node: {repo} (pass --clone-url to clone it node-locally)")
            env_root = Path(options["venv"]).resolve()
            if env_root == repo.resolve() or env_root in repo.resolve().parents or repo.resolve() in env_root.parents:
                raise RuntimeError(f"source checkout {repo} and venv {env_root} must be separate directories (uv sync refuses a non-env dir)")
            sha = run(["git", "rev-parse", "HEAD"], capture=True)
            if sha != COMMIT:
                raise RuntimeError(f"Expected checkout {COMMIT}, found {sha}")
            run(["git", "diff", "--quiet", "HEAD", "--"])
            result["repo_commit"] = sha
            env_dir = Path(options["venv"])
            python = env_dir / "bin/python"
            env = dict(os.environ)
            # The image's PYTHONPATH can point at the older colocated verl checkout.
            env["PYTHONPATH"] = os.pathsep.join((str(repo), options["bundle"]))
            env["PYTHONNOUSERSITE"] = "1"
            env["VERL_PLATFORM"] = "nvidia"
            env["VLLM_USE_V1"] = "1"
            env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
            env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
            env["UV_CACHE_DIR"] = options["venv"] + "-uv-cache"
            env["UV_LINK_MODE"] = "copy"
            env.pop("VIRTUAL_ENV", None)
            env.pop("CONDA_PREFIX", None)

            if not options["check"]:
                uv = shutil.which("uv")
                if uv is None:
                    tools_dir = Path(options["venv"] + "-tools")
                    tools_python = tools_dir / "bin/python"
                    if not tools_python.exists():
                        run([sys.executable, "-m", "venv", str(tools_dir)])
                    run([str(tools_python), "-m", "pip", "install", "uv"])
                    uv = str(tools_dir / "bin/uv")
                result["uv"] = run([uv, "--version"], capture=True)
                run([
                    uv, "sync", "--project", str(repo), "--python", sys.executable,
                    "--frozen", "--all-packages", "--extra", "fsdp", "--extra", "vllm",
                    "--no-install-package", "ray", "--inexact",
                ], env=env)
                ray_probe = subprocess.run(
                    [str(python), "-c", "import ray; print(ray.__version__); print(ray.__commit__)"],
                    cwd=str(repo), env=env, text=True, capture_output=True,
                )
                installed = ray_probe.stdout.strip().splitlines()
                wanted = [expected["ray"], expected["ray_commit"]]
                if ray_probe.returncode or installed != wanted:
                    ray_source = options["ray_wheel"] or f"ray=={expected['ray']}"
                    run([uv, "pip", "install", "--python", str(python), "--no-deps", "--reinstall", ray_source], env=env)
                run([uv, "pip", "freeze", "--python", str(python)], env=env)
            if not python.exists():
                raise RuntimeError(f"Missing environment {python}; rerun without --check")
            output = run([str(python), "-c", PROBE], env=env, capture=True)
            profiles = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("ENV_PROFILE_JSON=")]
            if len(profiles) != 1:
                raise RuntimeError("Environment probe did not emit exactly one profile")
            profile = json.loads(profiles[0])
            if (profile["ray"], profile["ray_commit"]) != (expected["ray"], expected["ray_commit"]):
                raise RuntimeError("New environment's Ray build differs from running cluster; use --ray-wheel for a custom build")
            if profile["python"] != expected["python"]:
                raise RuntimeError("New environment's Python differs from running cluster")
            if not Path(profile["verl_file"]).is_relative_to(repo):
                raise RuntimeError(f"Imported wrong verl checkout: {profile['verl_file']}")
            result.update(ok=True, profile=profile, ray_lock_override=expected["ray"] != LOCKED_RAY)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            with log_path.open("a") as log:
                traceback.print_exc(file=log)
        except OSError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/workspace/verl-pin")
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--venv", default="/workspace/verl-pin/.venv")
    parser.add_argument("--address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="Import-check every node; install nothing")
    parser.add_argument("--ray-wheel", help="Shared path to wheel for an existing custom Ray build")
    parser.add_argument("--clone-url", default="https://github.com/jialei777/verl-upstream.git",
                        help="clone the pinned commit into --repo on every node when it is not already there (node-local source); "
                             "pass an empty string to require a pre-existing checkout")
    parser.add_argument("--clone-branch", default="tpu-main")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    args.repo = str(Path(args.repo).resolve())
    args.bundle = str(Path(args.bundle).resolve())
    args.venv = str(Path(args.venv).absolute())
    if args.ray_wheel:
        args.ray_wheel = str(Path(args.ray_wheel).resolve(strict=True))
    expected = ambient_profile()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.address, ignore_reinit_error=True)
    nodes = sorted((n for n in ray.nodes() if n.get("Alive")), key=lambda n: n["NodeID"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.bundle) / "logs/prepare" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    options = vars(args) | {"log_dir": str(log_dir)}
    # Some KubeRay head pods intentionally advertise zero schedulable CPUs.
    # This bounded administrative task must still prepare that node's interpreter.
    worker = ray.remote(num_cpus=0, num_gpus=0, max_retries=0)(prepare_node)
    pending, results = {}, []
    queue = list(nodes)
    print(f"Preparing/checking {len(nodes)} live nodes; {args.parallel} in parallel. Logs: {log_dir}", flush=True)
    while queue or pending:
        while queue and len(pending) < args.parallel:
            node = queue.pop(0)
            ref = worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(
                node["NodeID"], node["NodeManagerAddress"], options, expected,
            )
            pending[ref] = node
            print(f"START {node['NodeManagerAddress']} {node['NodeID'][:12]}", flush=True)
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=30)
        if not ready:
            print(f"WAIT {len(pending)} running, {len(queue)} queued; tail files in {log_dir}", flush=True)
        for ref in ready:
            node = pending.pop(ref)
            try:
                result = ray.get(ref)
            except Exception as exc:
                result = {"node_id": node["NodeID"], "node_ip": node["NodeManagerAddress"], "ok": False, "error": str(exc)}
            results.append(result)
            print(f"{'OK' if result['ok'] else 'FAILED'} {result['node_ip']} {result.get('error', '')}", flush=True)
    manifest = {
        "created_utc": stamp, "repo": args.repo, "repo_commit": COMMIT,
        "bundle": args.bundle, "venv": args.venv, "check_only": args.check,
        "locked_ray": LOCKED_RAY, "cluster_ambient": expected,
        "ray_policy": "Use the running cluster's exact Ray version/commit; never change raylets",
        "nodes": sorted(results, key=lambda r: r["node_id"]),
        "ok": bool(results) and all(r["ok"] for r in results),
    }
    profiles = [r["profile"] for r in results if r.get("ok")]
    if profiles and any(p != profiles[0] for p in profiles[1:]):
        manifest["ok"] = False
        manifest["error"] = "Prepared environment profiles differ across nodes; inspect manifest"
    manifest_path = log_dir / "environment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ENV_MANIFEST={manifest_path}", flush=True)
    (Path(args.bundle) / "ENV_MANIFEST_PATH").write_text(str(manifest_path) + "\n")
    ray.shutdown()
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
