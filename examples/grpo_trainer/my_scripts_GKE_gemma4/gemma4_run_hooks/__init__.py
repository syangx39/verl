"""[GEMMA4-RUN-HOOKS] One Ray worker_process_setup_hook for the GSM8K runs: coordinator step clock + checkpoint allowlist.

Ray accepts one worker_process_setup_hook per job, so this package is the single entry point. It installs:

* the step clock, when VERL_STEP_CLOCK=1. verl's Tracking.log is wrapped so that every metrics record carries the
  coordinator's own wall clock, taken in the TaskRunner process at the moment it logs that step:
    wallclock/coordinator_unix   time.time() at the log call (seconds since the epoch)
    wallclock/interval_s         time since the previous log call in this process (absent on the first record)
  Records are logged once per update, after the update, validation and metric computation. The interval of update
  k therefore covers the whole cycle of k: data loading, the timed step, validation when it runs, metric
  computation, plus the logging of update k-1 (the clock is read before the backends write). The values reach
  every backend (console line, file logger JSONL). They come from the coordinator itself, not from file
  timestamps or the driver's log forwarding;
* the checkpoint step allowlist, when VERL_CKPT_STEPS is set: gemma4_ckpt_steps.install(); that package ships next
  to this one via py_modules.

Usage in the job runtime_env:
    '++ray_kwargs.ray_init.runtime_env.py_modules=["<dir>/gemma4_ckpt_steps","<dir>/gemma4_run_hooks"]'
    ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gemma4_run_hooks.install
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_STEP_CLOCK="1"

verl.utils.tracking is patched when it is first imported, through a sys.meta_path finder, or at once if it is already
imported. The hook never raises. A failing hook would kill the worker, so failures are printed with the
[GEMMA4-RUN-HOOKS] prefix instead.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import time
import traceback

TAG = "[GEMMA4-RUN-HOOKS]"
ENV_CLOCK = "VERL_STEP_CLOCK"
TRACKING = "verl.utils.tracking"
KEY_UNIX = "wallclock/coordinator_unix"
KEY_INTERVAL = "wallclock/interval_s"


def _log(msg: str) -> None:
    print(f"{TAG} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} pid={os.getpid()} {msg}", flush=True)


def patch_tracking_class(cls) -> None:
    if getattr(cls, "_gemma4_step_clock", False):
        return
    orig = cls.log
    state = {"prev": None}

    def log(self, data, step, backend=None, **kwargs):
        now = time.time()
        try:
            data = dict(data)
            data[KEY_UNIX] = now
            if state["prev"] is not None:
                data[KEY_INTERVAL] = now - state["prev"]
            state["prev"] = now
        except Exception:  # noqa: BLE001  (never break logging)
            pass
        return orig(self, data, step, backend=backend, **kwargs)

    cls.log = log
    cls._gemma4_step_clock = True
    _log(f"step clock installed on {cls.__module__}.{cls.__qualname__}.log ({KEY_UNIX}, {KEY_INTERVAL})")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TRACKING:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            _log(f"WARNING: could not locate {TRACKING} (path={path}); step clock NOT installed")
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                patch_tracking_class(module.Tracking)
            except Exception:  # noqa: BLE001
                _log(f"step clock patch FAILED: {traceback.format_exc()[-1500:]}")

        spec.loader.exec_module = exec_module
        return spec


def install_step_clock() -> None:
    if os.environ.get(ENV_CLOCK, "") != "1":
        return
    try:
        if TRACKING in sys.modules:
            patch_tracking_class(sys.modules[TRACKING].Tracking)
        elif not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
            sys.meta_path.insert(0, _PatchOnImport())
    except Exception:  # noqa: BLE001
        _log(f"step clock install FAILED: {traceback.format_exc()[-1500:]}")


def install() -> None:
    """Ray worker_process_setup_hook. Never raises."""
    install_step_clock()
    if os.environ.get("VERL_CKPT_STEPS", ""):
        try:
            import gemma4_ckpt_steps

            gemma4_ckpt_steps.install()
        except Exception:  # noqa: BLE001
            _log(f"checkpoint allowlist install FAILED: {traceback.format_exc()[-1500:]}")
