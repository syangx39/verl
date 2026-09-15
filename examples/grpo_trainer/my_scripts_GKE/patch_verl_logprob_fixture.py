#!/usr/bin/env python3
"""Patch the verl fork to dump a parity fixture batch (env-gated, off by default).

When LOGPROB_FIXTURE_DIR is set, at global step LOGPROB_FIXTURE_STEP (default 1),
right after advantages are computed and BEFORE the actor update, the trainer:
  1. runs the trainer log-prob pass a second time on the same batch/weights
     (repeatability error of the trainer itself), and
  2. writes fixture_step<N>.npz with the padded batch tensors:
       prompts, responses, attention_mask, position_ids, response_mask,
       rollout_log_probs (vLLM sampler, raw), old_log_probs (trainer, pre-update),
       old_log_probs_repeat (second trainer pass), token_level_scores,
       token_level_rewards, advantages, returns,
     plus every numeric/string per-sample field in non_tensor_batch (uid, qid,
     acc, fmt, length_penalty, score, mv_* ...), and a JSON sidecar with shapes,
     the rollout temperature and the resolved dtypes.
Nothing else changes; normal runs (env unset) are untouched.

Idempotent. Prints: LOGPROB-FIXTURE PATCH: APPLIED | already patched | FAILED <reason>
Usage:
  python3 patch_verl_logprob_fixture.py /workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/meta-RL/verl/verl/trainer/ppo/ray_trainer.py"
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
                    if _os.environ.get("LOGPROB_FIXTURE_DIR") and self.global_steps == int(_os.environ.get("LOGPROB_FIXTURE_STEP", "1")):
                        self._dump_logprob_fixture(batch)

                    # update critic
'''
METHOD_INSERT = '''    def _dump_logprob_fixture(self, batch: DataProto):  # _LOGPROB_FIXTURE
        """Parity fixture: pre-update batch tensors + a repeated trainer log-prob pass on the same weights."""
        import json
        import os
        import numpy as np

        out_dir = os.environ["LOGPROB_FIXTURE_DIR"]
        os.makedirs(out_dir, exist_ok=True)
        repeat, _ = self._compute_old_log_prob(batch)          # second trainer pass, identical inputs and weights
        keys = ["prompts", "responses", "attention_mask", "position_ids", "response_mask",
                "rollout_log_probs", "old_log_probs", "token_level_scores", "token_level_rewards",
                "advantages", "returns"]
        arrays = {k: batch.batch[k].cpu().numpy() for k in keys if k in batch.batch.keys()}
        arrays["old_log_probs_repeat"] = repeat.batch["old_log_probs"].cpu().numpy()
        skipped = []
        for k, v in batch.non_tensor_batch.items():
            a = np.asarray(v)
            if a.dtype.kind == "O":                      # verl stores uid (and other strings) as dtype=object
                try:
                    a = a.astype(str)
                except Exception:  # noqa: BLE001
                    skipped.append(k)
                    continue
            if a.dtype.kind in "biufUS" and a.ndim == 1 and a.shape[0] == len(batch):
                arrays[f"nt__{k}"] = a
            else:
                skipped.append(k)
        # integrity: full batch must be 256 uid groups x 8 completions, one qid per uid
        if "nt__uid" not in arrays:
            raise RuntimeError("_LOGPROB_FIXTURE: uid missing from non_tensor_batch")
        uids = arrays["nt__uid"]
        groups = {}
        for i, u in enumerate(uids):
            groups.setdefault(u, []).append(i)
        n_rep = self.config.actor_rollout_ref.rollout.n
        bad_size = [u for u, idx in groups.items() if len(idx) != n_rep]
        if bad_size:
            raise RuntimeError(f"_LOGPROB_FIXTURE: {len(bad_size)} uid groups do not have {n_rep} completions")
        if "nt__qid" in arrays:
            mixed = [u for u, idx in groups.items() if len(set(arrays["nt__qid"][idx].tolist())) != 1]
            if mixed:
                raise RuntimeError(f"_LOGPROB_FIXTURE: {len(mixed)} uid groups carry more than one qid")
        # provenance: checkpoint / tokenizer / config hashes and versions of the environment that produced the dump
        import hashlib
        import importlib.metadata as md
        model_path = self.config.actor_rollout_ref.model.path

        def _sha(fn):
            fp = os.path.join(model_path, fn)
            return hashlib.sha256(open(fp, "rb").read()).hexdigest() if os.path.exists(fp) else None

        def _ver(pkg):
            try:
                return md.version(pkg)
            except md.PackageNotFoundError:
                return None

        step = self.global_steps
        np.savez_compressed(os.path.join(out_dir, f"fixture_step{step}.npz"), **arrays)
        meta = {
            "global_step": step,
            "batch_size": len(batch),
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
            "rollout": {"temperature": self.config.actor_rollout_ref.rollout.temperature,
                        "top_p": self.config.actor_rollout_ref.rollout.top_p,
                        "top_k": self.config.actor_rollout_ref.rollout.top_k,
                        "n": self.config.actor_rollout_ref.rollout.n,
                        "calculate_log_probs": self.config.actor_rollout_ref.rollout.calculate_log_probs},
            "actor": {"model_dtype": str(self.config.actor_rollout_ref.actor.fsdp_config.get("model_dtype", None)),
                      "strategy": self.config.actor_rollout_ref.actor.strategy,
                      "use_dynamic_bsz": self.config.actor_rollout_ref.actor.use_dynamic_bsz},
            "non_tensor_keys_skipped": skipped,
            "uid_groups": len(groups),
            "model_path": model_path,
            "sha256": {"model.safetensors": _sha("model.safetensors"), "tokenizer.json": _sha("tokenizer.json"),
                       "config.json": _sha("config.json"), "generation_config.json": _sha("generation_config.json")},
            "collect_env": {pkg: _ver(pkg) for pkg in ("torch", "vllm", "verl", "transformers", "numpy", "math-verify")},
            "note": ("old_log_probs = trainer log-probs before this step's update; old_log_probs_repeat = the same pass "
                     "run again (repeatability); rollout_log_probs = sampler log-probs of the sampled tokens; all "
                     "response-position tensors are aligned with responses/response_mask (right-padded)."),
        }
        with open(os.path.join(out_dir, f"fixture_step{step}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[_LOGPROB_FIXTURE] wrote {out_dir}/fixture_step{step}.npz ({len(batch)} sequences, keys={sorted(arrays)})")

'''
new = src.replace(CALL_ANCHOR, CALL_INSERT, 1).replace(METHOD_ANCHOR, METHOD_INSERT + METHOD_ANCHOR, 1)
try:
  ast.parse(new)
except SyntaxError as e:
  print(f"LOGPROB-FIXTURE PATCH: FAILED patched file does not parse: {e}")
  sys.exit(1)
open(path, "w", encoding="utf-8").write(new)
print("LOGPROB-FIXTURE PATCH: APPLIED")
