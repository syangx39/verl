#!/usr/bin/env bash
#SBATCH --job-name=verl-parity-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

# =============================================================================
# Smoke test: verl GRPO parity config, 1 node x 4 GPU, shrunk workload.
# Goal: validate the full loop (data -> vLLM gen -> reward -> FSDP update ->
# refit) end to end. NOT a perf run — overrides at the bottom shrink batch,
# response length, and step count.
# =============================================================================

CONTAINER=$HOME/meta-RL/containers/verl-vllm-arm64.sqsh
MOUNTS=$HOME/meta-RL:/workspace/meta-RL
SCRIPTS=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts

srun --container-image=$CONTAINER \
     --container-mounts=$MOUNTS \
     --container-workdir=/workspace/meta-RL \
     bash -c "
  set -x

  # ---- writable HOME + caches (container HOME is read-only) -------------
  # HOME redirect is the master fix: flashinfer, triton, HF, etc. all derive
  # default cache paths from Path.home(). Must come before everything else.
  export HOME=/workspace/meta-RL/.home
  export CACHE_ROOT=/workspace/meta-RL/.cache
  mkdir -p \$HOME \$CACHE_ROOT
  export XDG_CACHE_HOME=\$CACHE_ROOT
  export FLASHINFER_WORKSPACE_BASE=\$CACHE_ROOT/flashinfer
  export HF_HOME=\$CACHE_ROOT/huggingface
  export TRITON_CACHE_DIR=\$CACHE_ROOT/triton
  export TORCHINDUCTOR_CACHE_DIR=\$CACHE_ROOT/inductor
  export HF_HUB_OFFLINE=1

  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

  # ---- verl from mounted source tree (image has deps only) --------------
  export PYTHONPATH=/workspace/meta-RL/verl:\$PYTHONPATH

  # ---- inputs consumed by the parity launch script ----------------------
  export DATA_DIR=/workspace/meta-RL/data/openmathinstruct2
  export REWARD_FN_PATH=$SCRIPTS/maxtext_math_reward.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B

  # ---- deps missing from image (installed to \$HOME/.local on Lustre) ---
  pip install --user -q math-verify
  pip show math-verify | grep Version

  # ---- launch: parity config + smoke-size overrides ---------------------
  bash $SCRIPTS/run_qwen3_0p6b_maxtext_parity.sh \
    data.train_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    data.max_response_length=1024 \
    trainer.total_training_steps=3 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"
