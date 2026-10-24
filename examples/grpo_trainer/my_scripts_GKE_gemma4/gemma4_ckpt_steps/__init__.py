"""[GEMMA4-CKPT-STEPS] Save full checkpoints only at an explicit list of global steps (env VERL_CKPT_STEPS="100,250").

At the runtime pin (ace775e8) RayPPOTrainer.fit calls _save_checkpoint() when save_freq > 0 and the step is a
multiple of save_freq, the last step, or an ESI expiration is near. There is no allowlist option: save_freq=100 over
250 steps saves 100, 200 and 250. This hook wraps RayPPOTrainer._save_checkpoint so that it returns without
saving unless the current global step is listed. Pair it with a save_freq that reaches every listed step, e.g.
save_freq=50 for 100 and 250 (the launcher checks this). Everything a listed step saves is unchanged: model,
optimizer and extra state per checkpoint.save_contents, data.pt (dataloader state) and
latest_checkpointed_iteration.txt.

It is installed as a Ray worker_process_setup_hook through the job runtime_env; py_modules ships this directory.
It needs no change to verl or to the node-prep-managed files:

    '++ray_kwargs.ray_init.runtime_env.py_modules=["<abs path>/gemma4_ckpt_steps"]'
    ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gemma4_ckpt_steps.install
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_CKPT_STEPS=100,250

Ray accepts one worker_process_setup_hook per job, so this cannot be combined with gemma4_optim_probe (the launcher
refuses both). The hook is a no-op unless VERL_CKPT_STEPS is set. The trainer module is patched when it is first
imported, through a sys.meta_path finder, so worker processes that never import it are untouched.

Every decision is printed with the [GEMMA4-CKPT-STEPS] prefix: patched, skip and save, with the step and the
allowlist. It is also appended to VERL_CKPT_STEPS_LOG when that is set. An install error is printed and logged,
never raised: a failing hook would kill the worker process. Without the "patched" line, verl's own save_freq
behaviour applies.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import time
import traceback

ENV_STEPS = "VERL_CKPT_STEPS"
ENV_LOG = "VERL_CKPT_STEPS_LOG"
TARGET = "verl.trainer.ppo.ray_trainer"
TAG = "[GEMMA4-CKPT-STEPS]"


def parse_steps(text: str) -> frozenset[int]:
    steps = frozenset(int(p) for p in text.replace(" ", "").split(",") if p)
    if not steps or min(steps) < 1:
        raise ValueError(f"{ENV_STEPS}={text!r}: need a comma list of positive global steps")
    return steps


def _log(msg: str) -> None:
    line = f"{TAG} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid={os.getpid()} {msg}"
    print(line, flush=True)
    path = os.environ.get(ENV_LOG)
    if path:
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def patch_trainer_class(cls, steps: frozenset[int]) -> None:
    if getattr(cls, "_gemma4_ckpt_steps", None) is not None:
        return
    orig = cls._save_checkpoint
    allowed = sorted(steps)

    def _save_checkpoint(self, *args, **kwargs):
        step = int(self.global_steps)
        if step not in steps:
            _log(f"skip save at global_step={step} (allowed {allowed})")
            return None
        _log(f"save at global_step={step} (allowed {allowed})")
        t0 = time.time()
        out = orig(self, *args, **kwargs)
        _log(f"saved global_step={step} in {time.time() - t0:.1f}s")
        return out

    cls._save_checkpoint = _save_checkpoint
    cls._gemma4_ckpt_steps = steps
    _log(f"patched {cls.__module__}.{cls.__qualname__}._save_checkpoint allowlist={allowed}")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Finds TARGET with the standard path finder and patches RayPPOTrainer right after the module executes."""

    def __init__(self, steps: frozenset[int]):
        self.steps = steps

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            _log(f"WARNING: could not locate {TARGET} (path={path}); checkpoint allowlist NOT installed")
            return None
        orig_exec, steps = spec.loader.exec_module, self.steps

        def exec_module(module):
            orig_exec(module)
            try:
                patch_trainer_class(module.RayPPOTrainer, steps)
            except Exception:  # noqa: BLE001
                _log(f"patch FAILED: {traceback.format_exc()[-1500:]}")

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Ray worker_process_setup_hook. Never raises."""
    text = os.environ.get(ENV_STEPS, "")
    if not text:
        return
    try:
        steps = parse_steps(text)
        if TARGET in sys.modules:
            patch_trainer_class(sys.modules[TARGET].RayPPOTrainer, steps)
        elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
            sys.meta_path.insert(0, _PatchOnImport(steps))
    except Exception:  # noqa: BLE001
        _log(f"install FAILED: {traceback.format_exc()[-1500:]}")
