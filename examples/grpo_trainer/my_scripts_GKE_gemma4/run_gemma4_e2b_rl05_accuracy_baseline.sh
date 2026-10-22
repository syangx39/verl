#!/usr/bin/env bash
# =============================================================================
# Frozen 200-step GPU accuracy baseline launcher for google/gemma-4-E2B-it
# (PR #9, step 1). Wraps run_gemma4_e2b_rl05_parity.sh without modifying any
# training hyperparameter in the frozen rl-05 block.
#
# Frozen evaluation & baseline protocol (see EVAL_PROTOCOL.md):
#   - Recipe:         run_gemma4_e2b_rl05_parity.sh (batch 256 x n=8 = 2048,
#                     mini=256 [mu=1], lr=1e-6, clip=0.2/0.28, beta=0,
#                     max_prompt=8192, max_response=8192, T=0.8/top_k=50/top_p=0.95)
#   - Horizon:        TOTAL_STEPS=200 (steps 1..200)
#   - Held-out set:   /workspace/meta-RL/data/openmathinstruct2/gemma4_eval_5k/eval_5000.parquet
#                     (5,000 unique questions from val.parquet with ZERO overlap
#                     with train.parquet; sha256 d092a9c5e80aa2b3cf8e4e12d8367769e770f69ee6627ff1285fd5693efc662d)
#   - Prompt tokens:  eval_5000_prompt_tokens.parquet
#                     (sha256 a54fbb327d9c60545c49edfc26107b21a1e55f3ef9e673d80024c79e821b5ca4)
#   - Eval schedule:  step 0 (trainer.val_before_train=True) + every 20 steps
#                     (trainer.test_freq=20) -> 11 eval points: 0,20,40,..,200
#   - Eval decoding:  greedy (do_sample=False, temperature=0, top_k=-1, top_p=1.0, n=1)
#   - Seeds:          SEED=1 first, then SEED=2 and SEED=3 (sets data.seed,
#                     rollout.seed, actor.data_loader_seed, actor.fsdp_config.seed)
# =============================================================================

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

SEED=${SEED:-1}
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TOTAL_STEPS=${TOTAL_STEPS:-200}
TEST_FREQ=${TEST_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

W=$(( NNODES * NGPUS_PER_NODE ))
ROLLOUT_TP=${ROLLOUT_TP:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_e2b_rl05_${NNODES}n${W}g_s200_seed${SEED}_$(date -u +%m%d_%H%M)}

DATA_DIR=${GEMMA_DATA_DIR:-/workspace/meta-RL/data/openmathinstruct2}
TRAIN_FILE=${GEMMA_TRAIN_FILE:-$DATA_DIR/train.parquet}
EVAL_DIR=${GEMMA_EVAL_DIR:-$DATA_DIR/gemma4_eval_5k}
VAL_FILE=${GEMMA_VAL_FILE:-$EVAL_DIR/eval_5000.parquet}
PROMPT_TOK_FILE=${GEMMA_PROMPT_TOK_FILE:-$EVAL_DIR/eval_5000_prompt_tokens.parquet}
MODEL_PATH=${GEMMA_MODEL_PATH:-/workspace/meta-RL/models/gemma-4-E2B-it}

EXPECTED_EVAL_SHA256="d092a9c5e80aa2b3cf8e4e12d8367769e770f69ee6627ff1285fd5693efc662d"
EXPECTED_PTOK_SHA256="a54fbb327d9c60545c49edfc26107b21a1e55f3ef9e673d80024c79e821b5ca4"

if [ ! -f "$VAL_FILE" ] || [ ! -f "$PROMPT_TOK_FILE" ]; then
  echo "[eval-prep] building frozen 5,000-problem eval set in $EVAL_DIR ..."
  python3 "$SCRIPT_DIR/build_gemma4_eval_set.py" \
    --train "$TRAIN_FILE" \
    --val "$DATA_DIR/val.parquet" \
    --model "$MODEL_PATH" \
    --out-dir "$EVAL_DIR"
fi

ACTUAL_EVAL_SHA256=$(sha256sum "$VAL_FILE" | cut -d' ' -f1)
ACTUAL_PTOK_SHA256=$(sha256sum "$PROMPT_TOK_FILE" | cut -d' ' -f1)
if [ "$ACTUAL_EVAL_SHA256" != "$EXPECTED_EVAL_SHA256" ]; then
  echo "ERROR: $VAL_FILE sha256=$ACTUAL_EVAL_SHA256 != expected $EXPECTED_EVAL_SHA256" >&2
  exit 1
fi
if [ "$ACTUAL_PTOK_SHA256" != "$EXPECTED_PTOK_SHA256" ]; then
  echo "ERROR: $PROMPT_TOK_FILE sha256=$ACTUAL_PTOK_SHA256 != expected $EXPECTED_PTOK_SHA256" >&2
  exit 1
fi

EVAL_DUMP_DIR=${GEMMA4_EVAL_DUMP_DIR:-/workspace/meta-RL/logs/gemma4_eval}/${EXPERIMENT_NAME}
mkdir -p "$EVAL_DUMP_DIR/val" "$EVAL_DUMP_DIR/train"

BASELINE_LAUNCHER_SHA256=$(sha256sum "$0" | cut -d' ' -f1)
echo "[baseline-provenance] exp=${EXPERIMENT_NAME} seed=${SEED} nnodes=${NNODES} W=${W} total_steps=${TOTAL_STEPS} test_freq=${TEST_FREQ} val_file=${VAL_FILE} val_sha256=${ACTUAL_EVAL_SHA256} ptok_sha256=${ACTUAL_PTOK_SHA256} eval_dump_dir=${EVAL_DUMP_DIR} baseline_launcher_sha256=${BASELINE_LAUNCHER_SHA256}"

export EXPERIMENT_NAME NNODES NGPUS_PER_NODE TOTAL_STEPS
export GEMMA_TRAIN_FILE="$TRAIN_FILE"
export GEMMA_VAL_FILE="$VAL_FILE"
export GEMMA_MODEL_PATH="$MODEL_PATH"

exec bash "$SCRIPT_DIR/run_gemma4_e2b_rl05_parity.sh" \
  data.seed="${SEED}" \
  data.validation_shuffle=False \
  actor_rollout_ref.rollout.seed="${SEED}" \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.validation_data_dir="${EVAL_DUMP_DIR}/val" \
  trainer.rollout_data_dir="${EVAL_DUMP_DIR}/train" \
  "$@"
