#!/usr/bin/env python3
"""Patch verl (pin ace775e8, legacy RayPPOTrainer) to dump a pre-update log-prob parity fixture.

Env-gated, off by default. When LOGPROB_FIXTURE_DIR is set, at global step LOGPROB_FIXTURE_STEP
(default 1), right after advantages are computed and BEFORE the critic/actor update, the trainer:
  1. reruns the trainer log-prob pass on the same batch and weights (trainer repeatability), and
  2. writes fixture_step<N>.npz with the padded batch tensors
       prompts, responses, attention_mask, position_ids, response_mask,
       rollout_log_probs (vLLM sampler), old_log_probs (trainer, pre-update),
       old_log_probs_repeat (second trainer pass), advantages, token_level_scores, ...
     plus 1-D non-tensor fields (uid, ...) and a JSON sidecar with shapes, sampling params,
     package versions and hashes of the small checkpoint files.
Token IDs are saved as produced by the rollout (no decode / re-tokenize).

Adapted from my_scripts_GKE/patch_verl_logprob_fixture.py (written for the older fork) to the
anchors in /workspace/verl-pin/verl/trainer/ppo/ray_trainer.py. Normal runs (env unset) are untouched.

Idempotent. Prints: LOGPROB-FIXTURE PATCH: APPLIED | already patched | FAILED <reason>
Usage:
  python3 patch_verl_pin_logprob_fixture.py /workspace/verl-pin/verl/trainer/ppo/ray_trainer.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl-pin/verl/trainer/ppo/ray_trainer.py"
src = open(path, encoding="utf-8").read()
MARK = "_LOGPROB_FIXTURE"
if MARK in src:
    print("LOGPROB-FIXTURE PATCH: already patched")
    sys.exit(0)

CALL_ANCHOR = '''                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    # update critic
'''
METHOD_ANCHOR = '''    def _update_actor(self, batch: DataProto) -> DataProto:
'''
for name, anchor in (("call", CALL_ANCHOR), ("method", METHOD_ANCHOR)):
    if src.count(anchor) != 1:
        print(f"LOGPROB-FIXTURE PATCH: FAILED {name} anchor found {src.count(anchor)} times (expected 1)")
        sys.exit(1)

CALL_INSERT = '''                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    # _LOGPROB_FIXTURE: env-gated parity dump of the pre-update batch (see _dump_logprob_fixture)
                    import os as _os
                    if _os.environ.get("LOGPROB_FIXTURE_DIR") and self.global_steps == int(
                        _os.environ.get("LOGPROB_FIXTURE_STEP", "1")
                    ):
                        self._dump_logprob_fixture(batch)
                    # update critic
'''
METHOD_INSERT = '''    def _dump_logprob_fixture(self, batch: DataProto):  # _LOGPROB_FIXTURE
        """Parity fixture: pre-update batch tensors + a repeated trainer log-prob pass on the same weights."""
        import hashlib
        import importlib.metadata as md
        import json
        import os

        import numpy as np

        out_dir = os.environ["LOGPROB_FIXTURE_DIR"]
        os.makedirs(out_dir, exist_ok=True)
        repeat_error = None
        try:
            repeat, _ = self._compute_old_log_prob(batch)  # second trainer pass, identical inputs and weights
        except Exception as e:  # noqa: BLE001 -- the dump must not kill the run
            repeat, repeat_error = None, repr(e)
        keys = ["prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask",
                "rollout_log_probs", "old_log_probs", "token_level_scores", "token_level_rewards",
                "advantages", "returns"]
        arrays = {k: batch.batch[k].cpu().numpy() for k in keys if k in batch.batch.keys()}
        if repeat is not None:
            arrays["old_log_probs_repeat"] = repeat.batch["old_log_probs"].cpu().numpy()
        skipped = []
        for k, v in batch.non_tensor_batch.items():
            a = np.asarray(v)
            if a.dtype.kind == "O":  # verl stores uid (and other strings) as dtype=object
                try:
                    a = a.astype(str)
                except Exception:  # noqa: BLE001
                    skipped.append(k)
                    continue
            if a.dtype.kind in "biufUS" and a.ndim == 1 and a.shape[0] == len(batch):
                arrays[f"nt__{k}"] = a
            else:
                skipped.append(k)
        model_path = self.config.actor_rollout_ref.model.path

        def _sha_small(fn, limit=64 << 20):
            fp = os.path.join(model_path, fn)
            if not os.path.isfile(fp):
                return None
            if os.path.getsize(fp) > limit:  # do not read multi-GB weight files from the FUSE mount
                return f"size={os.path.getsize(fp)}"
            return hashlib.sha256(open(fp, "rb").read()).hexdigest()

        def _ver(pkg):
            try:
                return md.version(pkg)
            except md.PackageNotFoundError:
                return None

        step = self.global_steps
        np.savez_compressed(os.path.join(out_dir, f"fixture_step{step}.npz"), **arrays)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        meta = {
            "global_step": step,
            "batch_size": len(batch),
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
            "rollout": {"temperature": rollout_cfg.temperature, "top_p": rollout_cfg.top_p,
                        "top_k": rollout_cfg.top_k, "n": rollout_cfg.n,
                        "calculate_log_probs": rollout_cfg.calculate_log_probs,
                        "logprobs_mode": rollout_cfg.get("logprobs_mode", None)},
            "actor": {"model_dtype": str(self.config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", None)),
                      "strategy": self.config.actor_rollout_ref.actor.strategy,
                      "use_dynamic_bsz": self.config.actor_rollout_ref.actor.use_dynamic_bsz,
                      "attn_implementation": str(self.config.actor_rollout_ref.model.get("override_config", {})
                                                 .get("attn_implementation", None))},
            "repeat_error": repeat_error,
            "non_tensor_keys_skipped": skipped,
            "model_path": model_path,
            "files": {fn: _sha_small(fn) for fn in sorted(os.listdir(model_path))
                      if fn.endswith((".json", ".safetensors", ".model", ".jinja"))},
            "collect_env": {pkg: _ver(pkg) for pkg in ("torch", "vllm", "verl", "transformers", "flashinfer-python")},
            "note": ("old_log_probs = trainer log-probs before this step's update; old_log_probs_repeat = the same "
                     "pass run again; rollout_log_probs = sampler log-probs of the sampled tokens. Response-position "
                     "tensors are aligned with responses/response_mask (right-padded); prompts are left-padded."),
        }
        with open(os.path.join(out_dir, f"fixture_step{step}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[_LOGPROB_FIXTURE] wrote {out_dir}/fixture_step{step}.npz ({len(batch)} sequences, "
              f"keys={sorted(arrays)}, repeat_error={repeat_error})")

'''
new = src.replace(CALL_ANCHOR, CALL_INSERT, 1).replace(METHOD_ANCHOR, METHOD_INSERT + METHOD_ANCHOR, 1)
try:
    ast.parse(new)
except SyntaxError as e:
    print(f"LOGPROB-FIXTURE PATCH: FAILED patched file does not parse: {e}")
    sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("LOGPROB-FIXTURE PATCH: APPLIED")
