#!/usr/bin/env bash
#SBATCH --job-name=magellan-prefill16k-1n4g
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# =============================================================================
# Magellan prefill-heavy single-node run: 1 node x 4 GPU.
# Default 5 steps (validation); pass TOTAL_STEPS=20 for a full run.
# Workload: synthetic ranking data, ~15K-token prompts (cap 16384), 8K
# response cap, dummy diversity reward. NOT comparable to the MaxText parity
# runs — different workload line.
#
# What to check in the log:
#   1. vLLM preemption warnings — 480 x ~15K prefill wave is a new memory
#      regime; if preemption fires, raise gpu_memory_utilization or cap
#      max_num_seqs (open item #2 in run_meta_prefill_16k.sh)
#   2. timing_s/gen composition — prefill-dominated now; expect very
#      different shape from the OpenMathInstruct runs
#   3. critic/score/mean — dummy reward; 0.6B will mostly flout the format,
#      so anything in 0.0-0.3 band is fine as long as it is NOT constant 0
#      (constant 0 = parsing chain broken)
#   4. prompt_length/mean ~14.8K, clip_ratio 0 (filter guarantees fit)
# =============================================================================

CONTAINER=$HOME/meta-RL/containers/verl-vllm-arm64.sqsh
MOUNTS=$HOME/meta-RL:/workspace/meta-RL
# Magellan-line assets currently live under data/magellan_synthetic
# (launch script + reward). Move to my_scripts later if desired — then just
# update MAGELLAN_DIR.
MAGELLAN_DIR=/workspace/meta-RL/data/magellan_synthetic

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
  export DATA_DIR=/workspace/meta-RL/data/magellan_synthetic_16k
  export REWARD_FN_PATH=$MAGELLAN_DIR/dummy_ranking_reward.py
  export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B

  # ---- launch: 5-step smoke ---------------------------------------------
  # (math-verify install dropped: dummy reward is pure-stdlib re/json)
  TOTAL_STEPS=\${TOTAL_STEPS:-5} bash $MAGELLAN_DIR/run_meta_prefill_16k.sh \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"
