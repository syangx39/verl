#!/usr/bin/env python3
"""Verify that the active Python environment matches a pinned uv.lock (CUDA-13 variants where the lock has several).

Usage: verify_venv_lock.py <repo_dir> [--require-async]
Checks ray, torch, vllm, transformers, transferqueue, flash-attn against uv.lock; when a package has several entries
(torch: 2.11.0+cpu and 2.11.0+cu130), the entry whose version or source mentions cu13/cu130 is the expected one.
Also asserts verl is imported from <repo_dir>. Exit 0 with a one-line summary, exit 1 with the mismatches.
"""
import importlib.metadata as md
import sys
import tomllib


def expected_versions(lock_path, names, cuda_tag="cu13"):
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)
    out = {}
    for name in names:
        entries = [p for p in lock.get("package", []) if p.get("name") == name]
        if not entries:
            out[name] = None
            continue
        def is_cuda(p):
            src = p.get("source", {}); txt = " ".join(str(v) for v in src.values()) + " " + p.get("version", "")
            return cuda_tag in txt
        cuda = [p for p in entries if is_cuda(p)]
        chosen = (cuda or entries)[0]
        out[name] = chosen["version"]
        if len(entries) > 1 and not cuda:
            out[name] = None  # ambiguous and no CUDA marker: do not guess
    return out


def main():
    repo = sys.argv[1]
    want = expected_versions(f"{repo}/uv.lock", ("ray", "torch", "vllm", "transformers", "transferqueue", "flash-attn"))
    bad, lines = [], []
    for name, w in want.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        lines.append(f"{name:14s} installed {got!s:16s} locked {w!s}")
        if w is None or got != w:
            bad.append(name)
    import verl  # noqa: E402
    print("\n".join(lines)); print("verl from", verl.__file__)
    if not verl.__file__.startswith(repo.rstrip("/") + "/"):
        bad.append(f"verl-import-path:{verl.__file__}")
    import flash_attn  # a matching version string does not prove the CUDA extension loads; importing does
    print("flash_attn", flash_attn.__version__, "imports OK")
    if "--require-async" in sys.argv:
        from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync  # noqa: F401
    if bad:
        print("MISMATCH:", bad); sys.exit(1)
    import ray, torch, vllm, transformers
    print(f"VENV OK: ray {ray.__version__} torch {torch.__version__} cuda {torch.version.cuda} vllm {vllm.__version__} transformers {transformers.__version__}")


if __name__ == "__main__":
    main()
