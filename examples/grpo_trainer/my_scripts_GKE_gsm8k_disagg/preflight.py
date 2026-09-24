"""Resolved-config, data and per-node GPU checks before reserving the full pools."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    from omegaconf import OmegaConf
    import pyarrow.parquet as pq
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    c = OmegaConf.load(args.config)
    expected = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "separate_async",
        "trainer.v1.separate_async.parameter_sync_step": 1,
        "trainer.v1.separate_async.hybrid_rollout.enable_switch": False,
        "trainer.v1.sampler.max_off_policy_threshold": 2,
        "trainer.v1.sampler.max_off_policy_strategy": "drop",
        "trainer.nnodes": 4, "trainer.n_gpus_per_node": 4,
        "actor_rollout_ref.rollout.nnodes": 12,
        "actor_rollout_ref.rollout.n_gpus_per_node": 4,
        "actor_rollout_ref.hybrid_engine": False,
        "data.train_batch_size": 128, "data.gen_batch_size": 1,
        "data.max_response_length": 2048, "data.max_prompt_length": 512,
        "data.val_max_samples": -1,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.filter_groups.enable": False,
        "algorithm.use_kl_in_reward": False,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 128,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "bypass_mode",
        "algorithm.rollout_correction.bypass_mode": True,
        "algorithm.rollout_correction.loss_type": "reinforce",
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": 3.0,
        "algorithm.rollout_correction.rollout_is_batch_normalize": False,
        "algorithm.rollout_correction.rollout_rs": None,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.actor.optim.lr": 2e-6,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 10,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "cosine",
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.optim.override_optimizer_config.eps": 1e-8,
        "actor_rollout_ref.actor.optim.override_optimizer_config.fused": False,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.n": 16,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "nccl",
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "reward.reward_manager.name": "BoxedRewardManager",
        "reward.reward_manager.source": "importlib",
        "transfer_queue.enable": True,
    }
    for key, want in expected.items():
        got = OmegaConf.select(c, key, default="<missing>")
        assert got == want, f"{key}: {got!r} != {want!r}"
    assert c.actor_rollout_ref.actor.policy_loss.rollout_correction == c.algorithm.rollout_correction
    runtime_env = OmegaConf.to_container(c.ray_kwargs.ray_init.runtime_env, resolve=True)
    for k, v in {
        "REWARD_MAX_RESP_LEN": "2048", "REWARD_OVERLONG_BUFFER": "512",
        "REWARD_OVERLONG_PENALTY": "1.0", "REWARD_FORMAT_SCORE": "0.1",
        "REWARD_PENALTY_SOURCES": "gsm8k_boxed_train",
    }.items():
        assert runtime_env["env_vars"][k] == v, (k, runtime_env["env_vars"][k])

    model = Path(c.actor_rollout_ref.model.path)
    eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    assert sorted(eos) == [151643, 151645], f"Reuse the stop-set-patched model, got EOS {eos}"
    assert (model / "model.safetensors").is_file(), model
    paths = [str(model / "model.safetensors"), str(model / "tokenizer.json"),
             c.reward.reward_manager.module.path, c.reward.custom_reward_function.path]
    all_rows = []
    for fs, n, source in [(c.data.train_files, 7473, "gsm8k_boxed_train"),
                           (c.data.val_files, 1319, "gsm8k_boxed_test")]:
        assert len(fs) == 1, f"Expected one dataset, got {fs}"
        rows = pq.read_table(fs[0]).to_pylist()
        assert len(rows) == n, (fs[0], len(rows), n)
        assert all(r["data_source"] == source for r in rows), f"Wrong reward source in {fs[0]}"
        assert all(r["prompt"][0]["content"] ==
                   "You are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."
                   for r in rows), f"Unexpected prompt in {fs[0]}"
        questions = {r["prompt"][-1]["content"] for r in rows}
        assert len(questions) == n, "Repeated questions"
        all_rows.append(questions)
        paths.append(fs[0])
    assert not (all_rows[0] & all_rows[1]), "Train/eval overlap"

    # Exercise the actual dataclass/engine configuration before starting workers.
    from verl.utils.config import omega_conf_to_dataclass
    actor = omega_conf_to_dataclass(c.actor_rollout_ref.actor)
    assert actor.policy_loss.rollout_correction.get("rollout_is_threshold") == 3.0
    assert actor.fsdp_config.strategy == "fsdp2"

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    assert len(nodes) == 16 and all(n["Resources"]["GPU"] == 4 for n in nodes), \
        "This recipe requires 16 alive nodes with 4 Ray GPU resources each"
    assert math.isclose(ray.available_resources().get("GPU", 0), 64), "The 64 GPUs must be idle"

    @ray.remote(num_cpus=1, num_gpus=1)
    def probe(required_paths):
        import importlib.metadata as md
        import platform
        import subprocess
        import torch
        import vllm  # noqa: F401
        import transfer_queue  # noqa: F401
        import verl
        from pathlib import Path
        for path in required_paths:
            assert Path(path).is_file(), f"Path unavailable on this node: {path}"
        assert str(Path(verl.__file__).resolve()).startswith(os.environ["PYTHONPATH"].split(":")[0] + "/")
        # Imports alone do not prove the driver's CUDA compatibility.
        x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        assert float((x @ x).sum()) == 4096
        torch.cuda.synchronize()
        return {"host": platform.node(), "python": platform.python_version(),
                "packages": {k: md.version(k) for k in ("ray", "torch", "vllm", "transformers", "TransferQueue")},
                "device": torch.cuda.get_device_name(), "verl": verl.__file__}

    try:
        futures = [probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False),
            runtime_env=runtime_env).remote(paths) for n in nodes]
        reports = ray.get(futures, timeout=600)
        assert all(r["packages"] == reports[0]["packages"] for r in reports), "Package versions differ across nodes"
        result = {"config_fields_verified": len(expected), "train_rows": 7473,
                  "eval_rows": 1319, "trainer_gpus": 16, "rollout_gpus": 48,
                  "nodes": reports}
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"[preflight] OK: {len(expected)} fields; 7473/1319 rows; 16 nodes; CUDA BF16 probe passed", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
