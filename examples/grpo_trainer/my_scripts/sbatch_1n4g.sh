#!/usr/bin/env bash
#SBATCH --job-name=verl-parity-1n4g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# =============================================================================
# Single-node parity run: 1 node x 4 GPU, frozen invariants (batch 480,
# prompt/response 8192/8192). Default 5 steps for quick validation; pass
# TOTAL_STEPS=20 in the environment for a full run.
#
# Current baseline at this scale (CUDA graphs on, free_cache off):
#   gen ~325s | update_actor ~138s | update_weights ~4s | step ~545s
#
# What to check in the log:
#   1. timing_s/* within the ranges above (regression watch)
#   2. critic/score/mean ~0.4 early steps (reward chain sanity)
#   3. response_length/clip_ratio ~0.27-0.33 (workload shape sanity)
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

  # ---- verl from mounted source tree ------------------------------------
  export PYTHONPATH=/workspace/meta-RL/verl:\$PYTHONPATH

  # ---- inputs -----------------------------------------------------------
  export DATA_DIR=/workspace/meta-RL/data/openmathinstruct2
  export REWARD_FN_PATH=$SCRIPTS/maxtext_math_reward.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B

  # ---- deps missing from image (persisted in \$HOME/.local on Lustre) ---
  pip install --user -q math-verify==0.9.0
  pip show math-verify | grep Version

  # ---- launch: full frozen-invariant config, 20 steps -------------------
  # free_cache_engine=False now lives in the parity script itself
  # (Experiment A; deviation from upstream gb200 branch documented there).
  TOTAL_STEPS=\${TOTAL_STEPS:-5} bash $SCRIPTS/run_qwen3_0p6b_maxtext_parity.sh \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"