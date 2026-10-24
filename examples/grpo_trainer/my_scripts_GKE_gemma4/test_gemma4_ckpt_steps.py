#!/usr/bin/env python3
"""Tests for gemma4_ckpt_steps (checkpoint step allowlist hook). Run: python3 test_gemma4_ckpt_steps.py [-v]

Each case runs in a fresh interpreter against a stub verl.trainer.ppo.ray_trainer whose fit loop copies the pin's save
condition (ace775e8, RayPPOTrainer.fit): save_freq > 0 and (last step or step % save_freq == 0).
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

STUB = '''
class RayPPOTrainer:
    def __init__(self, total, save_freq):
        self.total_training_steps, self.save_freq, self.global_steps, self.saved = total, save_freq, 0, []

    def _save_checkpoint(self):
        self.saved.append(self.global_steps)

    def fit(self):
        for step in range(1, self.total_training_steps + 1):
            self.global_steps = step
            is_last_step = self.global_steps >= self.total_training_steps
            if self.save_freq > 0 and (is_last_step or self.global_steps % self.save_freq == 0):
                self._save_checkpoint()
        return self.saved
'''

CHILD = textwrap.dedent('''
    import json, os, sys
    import gemma4_ckpt_steps as h
    if os.environ.get("IMPORT_FIRST") == "1":
        import verl.trainer.ppo.ray_trainer  # noqa: F401  (already imported when the hook runs)
    h.install()
    h.install()  # idempotent
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    t = RayPPOTrainer(int(os.environ["TOTAL"]), int(os.environ["SAVE_FREQ"]))
    print(json.dumps({"saved": t.fit(), "patched": getattr(RayPPOTrainer, "_gemma4_ckpt_steps", None) is not None,
                      "finders": sum(isinstance(f, h._PatchOnImport) for f in sys.meta_path)}))
''')


class CkptSteps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        pkg = os.path.join(root, "verl", "trainer", "ppo")
        os.makedirs(pkg)
        for d in ("verl", "verl/trainer", "verl/trainer/ppo"):
            open(os.path.join(root, d, "__init__.py"), "w").close()
        with open(os.path.join(pkg, "ray_trainer.py"), "w") as f:
            f.write(STUB)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_child(self, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith("VERL_CKPT")}
        e.update(PYTHONPATH=f"{self.tmp.name}{os.pathsep}{HERE}", **env)
        p = subprocess.run([sys.executable, "-c", CHILD], env=e, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.strip().splitlines()
        return json.loads(lines[-1]), [ln for ln in lines if ln.startswith("[GEMMA4-CKPT-STEPS]")]

    def test_100_and_250_only(self):
        log = os.path.join(self.tmp.name, "ckpt_steps.log")
        out, lines = self.run_child(VERL_CKPT_STEPS="100,250", VERL_CKPT_STEPS_LOG=log, TOTAL="250", SAVE_FREQ="50")
        self.assertEqual(out["saved"], [100, 250])
        self.assertTrue(out["patched"])
        self.assertEqual(out["finders"], 1)
        self.assertEqual(sum("patched" in ln for ln in lines), 1)
        self.assertEqual([ln.split("global_step=")[1].split()[0] for ln in lines if " skip save " in ln],
                         ["50", "150", "200"])
        self.assertEqual(sum(" saved global_step=" in ln for ln in lines), 2)
        with open(log) as f:
            self.assertEqual(f.read().count("[GEMMA4-CKPT-STEPS]"), len(lines))

    def test_without_hook_verl_saves_200_too(self):
        out, lines = self.run_child(TOTAL="250", SAVE_FREQ="100")
        self.assertEqual(out["saved"], [100, 200, 250])
        self.assertFalse(out["patched"])
        self.assertEqual(lines, [])

    def test_smoke_setting(self):
        out, _ = self.run_child(VERL_CKPT_STEPS="2", TOTAL="250", SAVE_FREQ="1")
        self.assertEqual(out["saved"], [2])

    def test_module_already_imported(self):
        out, _ = self.run_child(VERL_CKPT_STEPS="100,250", TOTAL="250", SAVE_FREQ="50", IMPORT_FIRST="1")
        self.assertEqual(out["saved"], [100, 250])
        self.assertEqual(out["finders"], 0)

    def test_bad_value_is_logged_not_raised(self):
        out, lines = self.run_child(VERL_CKPT_STEPS="100,abc", TOTAL="250", SAVE_FREQ="100")
        self.assertEqual(out["saved"], [100, 200, 250])
        self.assertTrue(any("install FAILED" in ln for ln in lines))


if __name__ == "__main__":
    unittest.main()
