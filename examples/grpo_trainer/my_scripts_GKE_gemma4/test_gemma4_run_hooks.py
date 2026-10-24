#!/usr/bin/env python3
"""Tests for gemma4_run_hooks (coordinator step clock + checkpoint allowlist dispatch). Run: python3 test_gemma4_run_hooks.py

Each case runs in a fresh interpreter against stubs of verl.utils.tracking (Tracking.log fans out to backends, as at
the pin) and verl.trainer.ppo.ray_trainer (the pin's save condition).
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

TRACKING_STUB = '''
RECORDS = []

class _Backend:
    def log(self, data, step):
        RECORDS.append((step, dict(data)))

class Tracking:
    def __init__(self):
        self.logger = {"console": _Backend()}

    def log(self, data, step, backend=None):
        for name, inst in self.logger.items():
            if backend is None or name in backend:
                inst.log(data=data, step=step)
'''

TRAINER_STUB = '''
class RayPPOTrainer:
    def __init__(self, total, save_freq):
        self.total_training_steps, self.save_freq, self.global_steps, self.saved = total, save_freq, 0, []

    def _save_checkpoint(self):
        self.saved.append(self.global_steps)

    def fit(self):
        for step in range(1, self.total_training_steps + 1):
            self.global_steps = step
            if self.save_freq > 0 and (step >= self.total_training_steps or step % self.save_freq == 0):
                self._save_checkpoint()
        return self.saved
'''

CHILD = textwrap.dedent('''
    import json, os, sys, time
    import gemma4_run_hooks as h
    if os.environ.get("IMPORT_FIRST") == "1":
        import verl.utils.tracking  # noqa: F401
    h.install()
    h.install()
    import verl.utils.tracking as tr
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    t = tr.Tracking()
    src = {"actor/lr": 1e-6}
    for step in range(3):
        t.log(data=src, step=step)
        time.sleep(0.05)
    saved = RayPPOTrainer(250, 50).fit()
    print(json.dumps({"records": tr.RECORDS, "src_untouched": src == {"actor/lr": 1e-6}, "saved": saved,
                      "finders": sum(type(f).__name__ == "_PatchOnImport" for f in sys.meta_path)}))
''')


class RunHooks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        for d in ("verl", "verl/utils", "verl/trainer", "verl/trainer/ppo"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
            open(os.path.join(root, d, "__init__.py"), "w").close()
        with open(os.path.join(root, "verl/utils/tracking.py"), "w") as f:
            f.write(TRACKING_STUB)
        with open(os.path.join(root, "verl/trainer/ppo/ray_trainer.py"), "w") as f:
            f.write(TRAINER_STUB)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_child(self, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("VERL_")}
        e.update(PYTHONPATH=os.pathsep.join([self.tmp.name, HERE]), **env)
        p = subprocess.run([sys.executable, "-c", CHILD], env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.strip().splitlines()
        return json.loads(lines[-1]), [ln for ln in lines if ln.startswith("[GEMMA4-")]

    def test_clock_and_allowlist(self):
        out, lines = self.run_child(VERL_STEP_CLOCK="1", VERL_CKPT_STEPS="100,250")
        recs = out["records"]
        self.assertEqual([s for s, _ in recs], [0, 1, 2])
        self.assertTrue(all("wallclock/coordinator_unix" in d for _, d in recs))
        self.assertNotIn("wallclock/interval_s", recs[0][1])
        for (_, a), (_, b) in zip(recs, recs[1:]):
            self.assertAlmostEqual(b["wallclock/interval_s"], b["wallclock/coordinator_unix"] - a["wallclock/coordinator_unix"],
                                   places=6)
            self.assertGreaterEqual(b["wallclock/interval_s"], 0.05)
        self.assertTrue(out["src_untouched"])          # the caller's dict is not mutated
        self.assertEqual(out["saved"], [100, 250])     # allowlist still applied through the combined hook
        self.assertEqual(sum("step clock installed" in ln for ln in lines), 1)
        self.assertEqual(sum("patched" in ln for ln in lines), 1)

    def test_clock_only(self):
        out, _ = self.run_child(VERL_STEP_CLOCK="1")
        self.assertTrue(all("wallclock/coordinator_unix" in d for _, d in out["records"]))
        self.assertEqual(out["saved"], [50, 100, 150, 200, 250])   # no allowlist: verl's own behaviour

    def test_off_by_default(self):
        out, lines = self.run_child()
        self.assertTrue(all("wallclock/coordinator_unix" not in d for _, d in out["records"]))
        self.assertEqual(out["saved"], [50, 100, 150, 200, 250])
        self.assertEqual(lines, [])

    def test_already_imported(self):
        out, _ = self.run_child(VERL_STEP_CLOCK="1", IMPORT_FIRST="1")
        self.assertTrue(all("wallclock/coordinator_unix" in d for _, d in out["records"]))
        self.assertEqual(out["finders"], 0)


if __name__ == "__main__":
    unittest.main()
