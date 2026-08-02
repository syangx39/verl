#!/usr/bin/env bash
#SBATCH --job-name=numina-trlparity-1n4g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# =============================================================================
# NuminaMath TRL-parity single-node run: 1 node x 4 GPU (W=4 -> SPG=32, i.e.
# 32 optimizer updates per step, mirroring TRL's schedule).
# # Default 40 steps (TRL max_steps); pass TOTAL_STEPS=5 for a quick validation run.
#
# What to check in the log:
#   1. "[accounting] W=4 SPG=32 ppo_mini_batch_size=8" printed at launch
#   2. critic/score/mean — format reward: expect low early (0.6B partially
#      compliant), NOT constant 0 (constant 0 = reward/parse chain broken)
#   3. response_length/clip_ratio — expect HIGH (format reward runs to cap;
#      Meta observed clipped ~1.0); mean length ~= max_response_length
#   4. yaml-confirmed config (cap 8192, temp 1.0, no KL); default 40 steps
#      matches TRL max_steps
# =============================================================================

CONTAINER=$HOME/meta-RL/containers/verl-vllm-arm64.sqsh
MOUNTS=$HOME/meta-RL:/workspace/meta-RL
# TRL-parity assets live under data/magellan_repro (launch script + reward).
REPRO_DIR=/workspace/meta-RL/data/magellan_repro

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
  export DATA_DIR=/workspace/meta-RL/data/numina_verl
  export REWARD_FN_PATH=$REPRO_DIR/format_reward_verl.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B

  # ---- launch ---------------------------------------------
  # (math-verify install dropped: dummy reward is pure-stdlib re/json)
  TOTAL_STEPS=\${TOTAL_STEPS:-40} bash $REPRO_DIR/run_qwen3_0p6b_trlparity.sh \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"
