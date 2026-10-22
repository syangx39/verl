"""[GEMMA4-OPTIM-PROBE] Env-gated diagnostic: what does torch.optim.AdamW actually write back to the weights?

Installed as a Ray ``worker_process_setup_hook`` through the job runtime_env (``py_modules`` ships this directory),
so it needs no change to verl or to the node-prep-managed files and runs through the unchanged launcher:

    bash run_gemma4_e2b_rl05_parity.sh \\
      '++ray_kwargs.ray_init.runtime_env.py_modules=["<abs path>/gemma4_optim_probe"]' \\
      ++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gemma4_optim_probe.install \\
      ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OPTIM_PROBE_DIR=<out dir>

It is a no-op unless ``VERL_OPTIM_PROBE_DIR`` is set, and it never modifies parameters, gradients or optimizer state
(read-only snapshots), so the training math is unchanged. For the first ``VERL_OPTIM_PROBE_STEPS`` (default 2) calls
of ``AdamW.step()`` in a process it records, per parameter tensor held by the optimizer (an FSDP1 flat-parameter
shard on this rank) and per original parameter inside it (via ``_fqns`` / ``_shard_param_infos``):

* the dtypes the optimizer actually holds: param, grad, ``exp_avg``, ``exp_avg_sq``, ``step``;
* an optimizer-only counterfactual on the *same* gradients, from the same (param-dtype) starting values cast to fp32
  and fresh moments, as a 2x2 of moment precision x write-back precision:

  ======== ===================================== ==================== ======================================
  name     moments                               write-back           how
  ======== ===================================== ==================== ======================================
  ideal    shadow fp32 ``m``, ``v``              fp32 (no rounding)   AdamW math in fp32
  fb       shadow fp32 ``m``, ``v``              param dtype          ``round(old + ideal) - old``
  bf       the optimizer's own ``exp_avg``/sq    fp32 (no rounding)   AdamW math on the optimizer's moments
  applied  the optimizer's own ``exp_avg``/sq    param dtype          measured ``param_after - param_before``
  ======== ===================================== ==================== ======================================

* element counts (total, nonzero gradient, changed, expected to survive = ``|ideal|`` above half an ulp of the stored
  value, the round-to-nearest prediction), norms and cosines of each variant against ``ideal``;
* rollout visibility: elements whose bf16 cast changed in this step and since the first probed call (the rollout
  engine receives bf16 weights, so small fp32 updates may accumulate before they become visible).

Output: ``<dir>/init_rank<RRRR>_pid<P>.json`` per optimizer construction and ``<dir>/rank<RRRR>_call<k>.json`` per
probed step. Summarize with ``gemma4_optim_probe_report.py``. Errors go to ``<dir>/error_*.json``; the probe never
raises into the trainer and always returns the original ``step`` result.
"""

from __future__ import annotations

import json
import os
import socket
import time
import traceback

ENV_DIR = "VERL_OPTIM_PROBE_DIR"
ENV_STEPS = "VERL_OPTIM_PROBE_STEPS"
PROBE_VERSION = "2"
SMALL = 2.0**-11  # |w| below this: a 1e-6 update survives bf16 round-to-nearest


def _rank_world() -> tuple[int, int]:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:  # noqa: BLE001
        pass
    return int(os.environ.get("RANK", "-1")), int(os.environ.get("WORLD_SIZE", "-1"))


def _write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _write_error(out_dir: str, where: str) -> None:
    try:
        os.makedirs(out_dir, exist_ok=True)
        rank, _ = _rank_world()
        _write_json(
            os.path.join(out_dir, f"error_{where}_rank{rank:04d}_{socket.gethostname()}_{os.getpid()}.json"),
            {"where": where, "traceback": traceback.format_exc()[-4000:]},
        )
    except Exception:  # noqa: BLE001
        pass


def half_ulp(x):
    """Half the spacing of ``x``'s own dtype at ``|x|`` (fp32 tensor). Zeros get the smallest normal spacing.

    For a normal number |x| = m * 2**e with m in [0.5, 1) (``torch.frexp``), the spacing of a format with ``p``
    significand bits (implicit bit included) is 2**(e - p): p = 8 for bf16, 11 for fp16, 24 for fp32. Under
    round-to-nearest an update ``d`` applied to ``x`` survives the write-back iff |d| > half_ulp(x) (ties and the
    smaller spacing just below a power of two are ignored).
    """
    import torch

    bits = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}[x.dtype]
    xf = x.detach().float().abs()
    _, e = torch.frexp(xf)
    ulp = torch.ldexp(torch.ones_like(xf), e - bits)
    ulp = torch.where(xf == 0, torch.full_like(xf, torch.finfo(x.dtype).tiny), ulp)
    return 0.5 * ulp


def _flat_names(p) -> dict | None:
    fqns = getattr(p, "_fqns", None)
    if not fqns:
        return None
    fqns = list(fqns)
    return {"n": len(fqns), "first": fqns[0], "last": fqns[-1]}


def _segments(p):
    """(fqn, slice) of each original parameter inside this rank's FSDP1 flat-parameter shard; None otherwise."""
    fqns, infos = getattr(p, "_fqns", None), getattr(p, "_shard_param_infos", None)
    if not fqns or infos is None or p.dim() != 1:
        return None
    out = []
    for fqn, info in zip(fqns, infos):
        if info.in_shard and info.numel_in_shard:
            out.append((fqn, slice(info.offset_in_shard, info.offset_in_shard + info.numel_in_shard)))
    return out


def install() -> None:
    """Ray worker_process_setup_hook. Never raises: a failing hook would kill the worker process."""
    out_dir = os.environ.get(ENV_DIR, "")
    if not out_dir:
        return
    try:
        _install(out_dir, int(os.environ.get(ENV_STEPS, "2")))
    except Exception:  # noqa: BLE001
        _write_error(out_dir, "install")


def _install(out_dir: str, max_calls: int) -> None:
    import torch

    cls = torch.optim.AdamW
    if getattr(cls, "_gemma4_optim_probe", False):
        return
    orig_init = cls.__init__
    orig_step = cls.step  # inherited Adam.step in torch >= 2.x; patching AdamW leaves Adam untouched
    shadow: dict[int, tuple] = {}  # id(param) -> (m_fp32, v_fp32)
    init_cast: dict[int, object] = {}  # id(param) -> bf16 cast of the value before the first probed step
    state = {"calls": 0}

    def probed_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            os.makedirs(out_dir, exist_ok=True)
            rank, world = _rank_world()
            params = [p for g in self.param_groups for p in g["params"]]
            _write_json(
                os.path.join(out_dir, f"init_rank{rank:04d}_pid{os.getpid()}.json"),
                {
                    "probe_version": PROBE_VERSION,
                    "rank": rank,
                    "world_size": world,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "torch": torch.__version__,
                    "optimizer_class": type(self).__name__,
                    "n_param_tensors": len(params),
                    "numel": int(sum(p.numel() for p in params)),
                    "param_dtypes": sorted({str(p.dtype) for p in params}),
                    "defaults": {
                        k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in self.defaults.items()
                        if isinstance(v, (int, float, bool, str, tuple, type(None)))
                    },
                },
            )
        except Exception:  # noqa: BLE001
            _write_error(out_dir, "init")

    def probed_step(self, *args, **kwargs):
        k = state["calls"]
        if k >= max_calls:
            return orig_step(self, *args, **kwargs)
        state["calls"] += 1
        t_start = time.time()
        pre = []
        try:
            with torch.no_grad():
                for gi, group in enumerate(self.param_groups):
                    b1, b2 = group["betas"]
                    for pi, p in enumerate(group["params"]):
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        gf = g.float()
                        m, v = shadow.get(id(p), (None, None))
                        if m is None:
                            m = torch.zeros_like(p, dtype=torch.float32)
                            v = torch.zeros_like(p, dtype=torch.float32)
                        m.mul_(b1).add_(gf, alpha=1.0 - b1)
                        v.mul_(b2).addcmul_(gf, gf, value=1.0 - b2)
                        shadow[id(p)] = (m, v)
                        old = p.detach().clone()
                        if id(p) not in init_cast:
                            init_cast[id(p)] = old.to(torch.bfloat16)
                        pre.append((gi, pi, p, old, int((g != 0).sum()), float(gf.pow(2).sum()), str(g.dtype)))
        except Exception:  # noqa: BLE001
            _write_error(out_dir, f"pre_call{k}")
            pre = None

        out = orig_step(self, *args, **kwargs)

        if pre is None:
            return out
        try:
            rank, world = _rank_world()
            keys = ("numel", "grad_nonzero", "changed", "expect_survive", "changed_and_expected", "changed_not_expected",
                    "param_abs_lt_2pow-11", "fb_changed", "cast_changed", "cast_changed_cum")
            fkeys = ("grad_sq", "ideal_sq", "applied_sq", "dot", "param_sq", "ideal_abs_sum", "applied_abs_sum",
                     "fb_sq", "fb_dot_ideal", "bf_sq", "bf_dot_ideal", "applied_dot_fb")
            totals = {key: 0 for key in keys}
            totals.update({key: 0.0 for key in fkeys})
            totals["bf_available"] = 0
            dtypes = {"param": set(), "grad": set(), "exp_avg": set(), "exp_avg_sq": set(), "step": set()}
            step_ts = set()
            recs = []
            with torch.no_grad():
                for gi, pi, p, old, n_gnz, g_sq, g_dtype in pre:
                    group = self.param_groups[gi]
                    lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
                    st = self.state.get(p, {})
                    step_v = st.get("step")
                    t = int(step_v.item()) if torch.is_tensor(step_v) else int(step_v if step_v is not None else k + 1)
                    step_ts.add(t)
                    for name in ("exp_avg", "exp_avg_sq"):
                        if name in st and torch.is_tensor(st[name]):
                            dtypes[name].add(str(st[name].dtype))
                    if torch.is_tensor(step_v):
                        dtypes["step"].add(f"{step_v.dtype}@{step_v.device.type}")
                    dtypes["param"].add(str(p.dtype))
                    dtypes["grad"].add(g_dtype)

                    m, v = shadow[id(p)]
                    bc1, bc2 = 1.0 - b1**t, 1.0 - b2**t
                    oldf = old.float()
                    newp = p.detach()
                    ideal = -lr * wd * oldf - (lr / bc1) * m / (v.sqrt() / (bc2**0.5) + eps)
                    applied = newp.float() - oldf
                    fb = (oldf + ideal).to(old.dtype).float() - oldf
                    bf = None
                    if torch.is_tensor(st.get("exp_avg")) and torch.is_tensor(st.get("exp_avg_sq")):
                        mo, vo = st["exp_avg"].float(), st["exp_avg_sq"].float()
                        bf = -lr * wd * oldf - (lr / bc1) * mo / (vo.sqrt() / (bc2**0.5) + eps)
                    changed = applied != 0
                    expect = ideal.abs() > half_ulp(old)
                    nb = newp.to(torch.bfloat16)
                    cast_ch = nb != old.to(torch.bfloat16)
                    cast_cum = nb != init_cast[id(p)]
                    small = oldf.abs() < SMALL
                    rec = {
                        "group": gi,
                        "index": pi,
                        "numel": p.numel(),
                        "grad_nonzero": n_gnz,
                        "changed": int(changed.sum()),
                        "expect_survive": int(expect.sum()),
                        "changed_and_expected": int((changed & expect).sum()),
                        "changed_not_expected": int((changed & ~expect).sum()),
                        "param_abs_lt_2pow-11": int(small.sum()),
                        "fb_changed": int((fb != 0).sum()),
                        "cast_changed": int(cast_ch.sum()),
                        "cast_changed_cum": int(cast_cum.sum()),
                        "grad_sq": g_sq,
                        "ideal_sq": float(ideal.pow(2).sum()),
                        "applied_sq": float(applied.pow(2).sum()),
                        "dot": float((ideal * applied).sum()),
                        "param_sq": float(oldf.pow(2).sum()),
                        "ideal_abs_sum": float(ideal.abs().sum()),
                        "applied_abs_sum": float(applied.abs().sum()),
                        "fb_sq": float(fb.pow(2).sum()),
                        "fb_dot_ideal": float((fb * ideal).sum()),
                        "bf_sq": float(bf.pow(2).sum()) if bf is not None else 0.0,
                        "bf_dot_ideal": float((bf * ideal).sum()) if bf is not None else 0.0,
                        "applied_dot_fb": float((applied * fb).sum()),
                        "bf_available": int(bf is not None),
                        "names": _flat_names(p),
                        "dtype": str(p.dtype),
                    }
                    segs = _segments(p)
                    if segs:
                        # [fqn, numel, changed, expect, small, cast_changed, cast_cum, ideal_sq, applied_sq, dot, param_sq]
                        rec["segments"] = [
                            [fqn, sl.stop - sl.start, int(changed[sl].sum()), int(expect[sl].sum()),
                             int(small[sl].sum()), int(cast_ch[sl].sum()), int(cast_cum[sl].sum()),
                             float(ideal[sl].pow(2).sum()), float(applied[sl].pow(2).sum()),
                             float((ideal[sl] * applied[sl]).sum()), float(oldf[sl].pow(2).sum())]
                            for fqn, sl in segs
                        ]
                    recs.append(rec)
                    for key in keys:
                        totals[key] += rec[key]
                    for key in fkeys:
                        totals[key] += rec[key]
                    totals["bf_available"] += rec["bf_available"]
                    del ideal, applied, fb, bf, changed, expect, nb, cast_ch, cast_cum, small, oldf
            g0 = self.param_groups[0]
            _write_json(
                os.path.join(out_dir, f"rank{rank:04d}_call{k}.json"),
                {
                    "probe_version": PROBE_VERSION,
                    "rank": rank,
                    "world_size": world,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "call": k,
                    "optimizer_step_t": sorted(step_ts),
                    "seconds": round(time.time() - t_start, 2),
                    "torch": torch.__version__,
                    "optimizer_class": type(self).__name__,
                    "hyper": {
                        key: (list(g0[key]) if isinstance(g0.get(key), tuple) else g0.get(key))
                        for key in ("lr", "betas", "eps", "weight_decay", "foreach", "fused", "amsgrad", "maximize",
                                    "capturable", "differentiable", "decoupled_weight_decay")
                    },
                    "dtypes": {key: sorted(val) for key, val in dtypes.items()},
                    "totals": totals,
                    "params": recs,
                },
            )
        except Exception:  # noqa: BLE001
            _write_error(out_dir, f"post_call{k}")
        finally:
            del pre
            if state["calls"] >= max_calls:
                shadow.clear()
                init_cast.clear()
        return out

    cls.__init__ = probed_init
    cls.step = probed_step
    cls._gemma4_optim_probe = True
