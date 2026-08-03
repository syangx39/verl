#!/usr/bin/env bash
#SBATCH --job-name=numina-trlparity-1n4g-pd8
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# =============================================================================
# NuminaMath TRL-parity single-node run: 1 node x 4 GPU
# (W=4 -> SPG=8 under the pd=8 / round-2 accounting, i.e. 8 optimizer
# updates per rollout, matching Meta's metaface sweep).
# Default 40 steps (TRL max_steps); pass TOTAL_STEPS=5 for a quick check.
# TP control: submit with ROLLOUT_TP=2 for the TP=2 variant, e.g.
#   ROLLOUT_TP=2 sbatch --job-name=numina-trlparity-1n4g-pd8-tp2 sbatch_numina_1n4g.sh
#   (env var must be exported to sbatch: use --export=ALL,ROLLOUT_TP=2)
#
# Direct comparison row: metaface GB200 W=4 pd=8 TRUE CYCLE = 222.8 s
# (gen 81.4 / update 40.4 / framework 100.9).
#
# What to check in the log:
#   1. "[accounting] W=4 SPG=8 ppo_mini_batch_size=32" printed at launch
#      (pd=8 accounting — NOT the old SPG=32/mini=8)
#   2. critic/score/mean — format reward: low early, NOT constant 0
#   3. response_length/mean ~3,900-4,000, clip_ratio ~0.25 (our measured
#      regime; Meta's "runs to cap" expectation is not observed here)
#   4. timing_s/step vs metaface 222.8 s; also vs our pd=2 number (250.2 s)
#      to isolate what the schedule change buys verl
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

  # ---- launch ------------------------------------------------------------
  # ROLLOUT_TP passthrough: default 1; ROLLOUT_TP=2 for the TP control run.
  ROLLOUT_TP=\${ROLLOUT_TP:-1} \
  TOTAL_STEPS=\${TOTAL_STEPS:-40} bash $REPRO_DIR/run_qwen3_0p6b_trlparity.sh \
    trainer.test_freq=-1 \
    trainer.val_before_train=False
"
