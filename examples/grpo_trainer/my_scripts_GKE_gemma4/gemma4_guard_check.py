#!/usr/bin/env python3
"""Validate one run's node-prep and [GEMMA4-ROPE] guard evidence by unique node / unique rank (never by log-line counts:
Ray de-duplicates per-rank log lines).

  python3 gemma4_guard_check.py --log LOG --guard-dir DIR --world 16 [--steps 2] [--expect-fail-rank R] [--out X.json]

Positive run (default):
  - DIR/node_prep.json: verdict PASS, one record per alive Ray node, all ok, managed files identical on every node;
  - exactly one GEMMA4-ROPE-GUARD-REPORT line for the actor role, covering ranks 0..W-1, all PASS, no fault injected;
  - DIR/init_actor_rank*.json and DIR/first_forward_actor_rank*.json for every rank 0..W-1, all PASS, every RoPE buffer
    rebuilt bit-exact in fp32 and still fp32-exact at the first forward;
  - "[driver] main_ppo finished with exit_code=0" and >= --steps training steps logged.
Negative run (--expect-fail-rank R, a deliberately corrupted rank):
  - node prep PASS; the guard report covers ranks 0..W-1 with exactly rank R FAIL and a fault recorded on rank R only;
  - the driver exit code is non-zero, the log carries the "[GEMMA4-ROPE] guard FAILED" error, and the job stopped at the
    init checkpoint: no training step logged and no first-forward record from any rank (the optimizer is built after
    the checkpoint, so no optimizer update can have happened).
Exit 0 when every expectation holds.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

MARK = "GEMMA4-ROPE-GUARD-REPORT "


def load_reports(log_text):
    reps = []
    for line in log_text.splitlines():
        i = line.find(MARK)
        if i >= 0:
            reps.append(json.loads(line[i + len(MARK):].strip()))
    return reps


def per_rank_files(guard_dir, pattern, world, wait_s):
    """Read DIR/<pattern> files keyed by rank; retry while the FUSE listing catches up."""
    t_end = time.time() + wait_s
    while True:
        recs = {}
        for p in glob.glob(os.path.join(guard_dir, pattern)):
            try:
                with open(p) as f:
                    r = json.load(f)
                recs[int(r["rank"])] = r
            except Exception:
                pass
        if len(recs) >= world or time.time() > t_end:
            return recs
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--guard-dir", required=True)
    ap.add_argument("--world", type=int, required=True, help="expected number of trainer ranks (NNODES x GPUs/node)")
    ap.add_argument("--role", default="actor")
    ap.add_argument("--steps", type=int, default=1, help="positive run: minimum number of training steps logged")
    ap.add_argument("--expect-fail-rank", type=int, default=None)
    ap.add_argument("--wait", type=int, default=120, help="seconds to wait for per-rank files on FUSE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    negative = args.expect_fail_rank is not None
    ranks = list(range(args.world))
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    with open(args.log, errors="replace") as f:
        log = f.read()

    # node prep
    try:
        with open(os.path.join(args.guard_dir, "node_prep.json")) as f:
            npr = json.load(f)
        nodes = npr.get("nodes", [])
        check("node_prep verdict PASS", npr.get("verdict") == "PASS", npr.get("verdict"))
        check("node_prep one ok record per alive node",
              len(nodes) == npr.get("n_nodes") and len({n.get("node_id") for n in nodes}) == len(nodes)
              and all(n.get("ok") for n in nodes), f"{len(nodes)} records / {npr.get('n_nodes')} alive nodes")
        cons = npr.get("consistency", {})
        check("node_prep managed files identical on every node",
              cons.get("managed_sha256", {}).get("n_distinct") == 1, json.dumps(cons.get("managed_sha256", {}).get("value")))
        node_summary = {"n_nodes": npr.get("n_nodes"), "rope_patch": npr.get("rope_patch"),
                        "managed_sha256": cons.get("managed_sha256", {}).get("value"),
                        "python": cons.get("python", {}).get("value"), "verl_head": cons.get("verl_head", {}).get("value"),
                        "packages": cons.get("import.packages", {}).get("value"),
                        "actions": sorted({f"{a.get('file', a.get('action'))}:{a.get('action')}"
                                           for n in nodes for a in n.get("actions", [])})}
    except Exception as e:
        check("node_prep report readable", False, repr(e))
        node_summary = None

    # guard report (rank 0, one line per role)
    reps = [r for r in load_reports(log) if r.get("role") == args.role and r.get("phase") == "init_checkpoint"]
    check(f"exactly one guard report for role={args.role}", len(reps) == 1, f"found {len(reps)}")
    rep = reps[-1] if reps else {}
    per_rank = {row[0]: row for row in rep.get("per_rank", [])}
    check("guard report covers ranks 0..W-1", sorted(per_rank) == ranks and rep.get("world_size") == args.world,
          f"world_size={rep.get('world_size')} ranks={sorted(per_rank)}")
    if negative:
        failed = sorted(r for r, row in per_rank.items() if row[1] != "PASS")
        check(f"guard report: exactly rank {args.expect_fail_rank} FAIL", failed == [args.expect_fail_rank], f"failed={failed}")
        check("fault configured for that rank", (rep.get("fault") or {}).get("rank") == args.expect_fail_rank,
              json.dumps(rep.get("fault")))
    else:
        check("guard report: all ranks PASS", rep.get("n_fail") == 0 and all(row[1] == "PASS" for row in per_rank.values()),
              f"n_fail={rep.get('n_fail')}")
        check("no fault injected", rep.get("fault") is None, json.dumps(rep.get("fault")))
        check("every rank rebuilt fp32 bit-exact", all(row[5] is True for row in per_rank.values()), "")

    # per-rank init files (written by each rank before the collective)
    init = per_rank_files(args.guard_dir, f"init_{args.role}_rank*.json", args.world, args.wait)
    check("per-rank init records for ranks 0..W-1", sorted(init) == ranks, f"have {sorted(init)}")
    fault_ranks = sorted(r for r, x in init.items() if x.get("fault_applied"))
    if negative:
        bad = sorted(r for r, x in init.items() if x.get("status") != "PASS")
        check(f"per-rank init records: exactly rank {args.expect_fail_rank} FAIL", bad == [args.expect_fail_rank], f"failed={bad}")
        check("fault applied on that rank only", fault_ranks == [args.expect_fail_rank], f"fault_applied on {fault_ranks}")
        x = init.get(args.expect_fail_rank, {})
        check("failing rank errors recorded", bool(x.get("errors")), " | ".join(x.get("errors", [])[:3])[:500])
    else:
        check("per-rank init records: all PASS", all(x.get("status") == "PASS" for x in init.values()), "")
        posts = [b.get("post", {}) for x in init.values() for m in x.get("modules", []) for b in m["buffers"].values()]
        check("per-rank init records: all buffers post-rebuild fp32 bit-exact",
              posts and all(p.get("bit_exact") and p.get("state") == "fp32" for p in posts), f"{len(posts)} buffers")
        check("no fault applied", not fault_ranks, f"{fault_ranks}")

    # first-forward records
    ff = per_rank_files(args.guard_dir, f"first_forward_{args.role}_rank*.json", 0 if negative else args.world,
                        0 if negative else args.wait)
    if negative:
        check("no first-forward record (stopped before any forward)", not ff, f"have {sorted(ff)}")
    else:
        check("first-forward records for ranks 0..W-1", sorted(ff) == ranks, f"have {sorted(ff)}")
        check("first-forward records: all PASS, nothing restored",
              all(x.get("status") == "PASS" and not x.get("restored") for x in ff.values()), "")

    # driver exit code and training steps
    m = re.findall(r"\[driver\] main_ppo finished with exit_code=(\d+)", log)
    rc = int(m[-1]) if m else None
    steps = sorted({int(s) for s in re.findall(r"\bstep:(\d+) - ", log)})
    if negative:
        check("driver exit code non-zero", rc not in (None, 0), f"exit_code={rc}")
        check("guard failure in the driver log", "[GEMMA4-ROPE] guard FAILED" in log, "")
        check("no training step logged", not steps, f"steps={steps}")
    else:
        check("driver exit_code=0", rc == 0, f"exit_code={rc}")
        check(f">= {args.steps} training steps logged", len(steps) >= args.steps, f"steps={steps}")

    ok = all(c["ok"] for c in checks)
    out = {"verdict": "PASS" if ok else "FAIL", "mode": "negative" if negative else "positive", "world": args.world,
           "role": args.role, "log": os.path.abspath(args.log), "guard_dir": os.path.abspath(args.guard_dir),
           "driver_exit_code": rc, "steps_logged": steps, "guard_report": rep, "node_prep": node_summary, "checks": checks}
    for c in checks:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    print(f"GEMMA4-GUARD-CHECK: {out['verdict']} mode={out['mode']} world={args.world} exit_code={rc} steps={steps}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
