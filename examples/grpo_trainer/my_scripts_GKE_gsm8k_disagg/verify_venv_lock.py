#!/usr/bin/env python3
"""Verify that the active Python environment matches a pinned uv.lock for the extras actually enabled.

Usage: verify_venv_lock.py <repo_dir> [--require-async] [--extras fsdp,vllm]
`uv export` expands uv.lock for the given extras into requirement lines with environment markers; the line whose
marker evaluates true on this machine is the expected version (this is how torch 2.11.0+cpu vs +cu130 and
transformers 5.3.0 vs 5.5.3 are told apart -- the lock holds several entries per package). Checks ray, torch, vllm,
transformers, transferqueue, flash-attn; imports flash_attn (a version string does not prove the CUDA extension
loads); asserts verl is imported from <repo_dir>. Exit 0 with a one-line summary, exit 1 with the mismatches.
"""
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys

CHECK = ("ray", "torch", "vllm", "transformers", "transferqueue", "flash-attn")


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_versions(repo, extras):
    uv = shutil.which("uv") or "/usr/local/bin/uv"
    cmd = [uv, "export", "--frozen", "--all-packages", "--format", "requirements-txt", "--no-hashes", "--no-header",
           "--no-emit-project", "--no-annotate"]
    for e in extras:
        cmd += ["--extra", e]
    out = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, check=True).stdout
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    want = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        name = canon(req.name)
        if name not in {canon(c) for c in CHECK}:
            continue
        if req.marker is not None and not Marker(str(req.marker)).evaluate():
            continue
        pins = [s.version for s in req.specifier if s.operator == "=="]
        if not pins:
            continue
        if name in want and want[name] != pins[0]:
            raise RuntimeError(f"lock resolves {name} to both {want[name]} and {pins[0]} on this platform")
        want[name] = pins[0]
    return want


def main():
    repo = sys.argv[1]
    extras = ["fsdp", "vllm"]
    if "--extras" in sys.argv:
        extras = sys.argv[sys.argv.index("--extras") + 1].split(",")
    want = expected_versions(repo, extras)
    bad, lines = [], []
    for name in CHECK:
        w = want.get(canon(name))
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        lines.append(f"{name:14s} installed {got!s:16s} locked {w!s}")
        if w is None or got != w:
            bad.append(name)
    print("\n".join(lines))
    import verl  # noqa: E402
    print("verl from", verl.__file__)
    if not verl.__file__.startswith(os.path.abspath(repo).rstrip("/") + "/"):
        bad.append(f"verl-import-path:{verl.__file__}")
    import flash_attn  # a matching version string does not prove the CUDA extension loads; importing does
    print("flash_attn", flash_attn.__version__, "imports OK")
    if "--require-async" in sys.argv:
        from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync  # noqa: F401
    if bad:
        print("MISMATCH:", bad)
        sys.exit(1)
    import ray, torch, vllm, transformers  # noqa: E401
    print(f"VENV OK: ray {ray.__version__} torch {torch.__version__} cuda {torch.version.cuda} vllm {vllm.__version__} transformers {transformers.__version__}")


if __name__ == "__main__":
    main()
