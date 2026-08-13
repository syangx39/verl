#!/usr/bin/env bash
# =============================================================================
# gemma-4-2B decode-light parity: verl GRPO on GB200, NuminaMath, matched to
# Meta's metaface recalibrated sweep (TPU-baseline decode-light regime)
# =============================================================================
# THE CONTRACT (four categories):
#   1. Frozen invariant — global batch: 128 prompts x n=8 = 1024 completions
#      per rollout. Decode-light: max_completion_length=64.
#   2. Doc-confirmed values — prompt cap 512 (non-binding, ~180-tok prompts);
#      MCL 64; temperature 0.7; top_k=-1; beta=0; lr 5e-6; bf16. [META-DOC]
#   3. System knobs:
#        * Update schedule (MATCHED per row): pd=2 FROZEN on their side
#          (262k-vocab logits OOM at pd>2) => SPG = 1024/(2W) = 512/W:
#          128/64/32/16/8/4 at W=4/8/16/32/64/128.
#          verl: ppo_mini_batch_size = 128/SPG prompts (1/2/4/8/16/32).
#        * Rollout engine: theirs TP=2 mem 0.4 TRITON_ATTN (head_size=256
#          rejects FLASH/FLASHINFER on Blackwell — check whether OUR vllm
#          build accepts gemma head 256 on flashinfer; if not, export
#          VLLM_ATTENTION_BACKEND=TRITON_ATTN and record as a SHARED
#          constraint, not a free variable). ROLLOUT_TP env, default 1.
#        * Chunking: use_dynamic_bsz splits/pads freely; gradient identical.
#   4. Scale knobs — NNODES / NGPUS_PER_NODE / TOTAL_STEPS.
# Reference (metaface TRUE CYCLE): W=4 1888.0 | 8 972.8 | 16 421.1 |
#   32 234.1 | 64 122.8 | 128 64.3  (framework bucket 82-93% at every point)
# =============================================================================

set -xeuo pipefail

########################### paths ###########################################
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/numina_verl}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/data/magellan_repro/format_reward_verl.py}
MODEL_PATH=${MODEL_PATH:-$HOME/meta-RL/models/gemma-4-2b-it}

########################### scale knobs #####################################
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TOTAL_STEPS=${TOTAL_STEPS:-20}
W=$(( NNODES * NGPUS_PER_NODE ))
PROJECT_NAME=${PROJECT_NAME:-gemma4_decodelight}

########################### accounting (update schedule) ####################
train_batch_size=128           # [FROZEN] 128 prompts
rollout_n=8                    # [FROZEN] -> 1024 completions/rollout

if (( W < 4 || 512 % W != 0 )); then
  echo "ERROR: W=${W} not in the sweep {4,8,16,32,64,128}" >&2
  exit 1
fi
SPG=$(( 512 / W ))             # pd=2 frozen on meta side => SPG=512/W
ppo_mini_batch_size=$(( 128 / SPG ))
echo "[accounting] W=${W}  SPG=${SPG}  ppo_mini_batch_size=${ppo_mini_batch_size}  rollout_tp=${ROLLOUT_TP:-1} (=> ${SPG} optimizer updates/step)"

########################### doc-confirmed values ############################
max_prompt_length=${MAX_PROMPT_LENGTH:-512}       # [META-DOC] cap 512, non-binding
max_response_length=${MAX_RESPONSE_LENGTH:-64}    # [META-DOC] decode-light lever
temperature=${TEMPERATURE:-0.7}                   # [META-DOC]
top_p=${TOP_P:-1.0}                               # meta doc silent -> TRL default 1.0
top_k=${TOP_K:--1}                                # [META-DOC] unrestricted
actor_lr=${ACTOR_LR:-5e-6}                        # [META-DOC]
kl_loss_coef=0.0                                  # [META-DOC] beta=0
USE_KL=False

########################### system adaptations ##############################
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=0.30
max_num_batched_tokens=8192
# decode-light: tiny sequences; token budgets stay generous (no cost)
ppo_max_token_len_per_gpu=16384

EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_2b_dl_${NNODES}n${W}g_tp${rollout_tp}_$(date +%Y%m%d_%H%M)}

RAY_NUM_GPUS_ARG=""
if [ -z "${RAY_ADDRESS:-}" ]; then
  RAY_NUM_GPUS_ARG="+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=format_reward \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    ${RAY_NUM_GPUS_ARG} \
    "$@"

# =============================================================================
# Bring-up watch list (gemma-specific):
#   1. head_size=256 vs our FlashInfer build — if engine init rejects it,
#      export VLLM_ATTENTION_BACKEND=TRITON_ATTN (shared constraint w/ meta)
#   2. W=4: mini=1 prompt (8 completions/update) x 128 updates — verl's
#      smallest-ever mini batch; watch update_actor for per-update overhead
#   3. completion length must pin at 64.0 (cap-bound; format reward can't
#      fit <think>/<answer> in 64 tok -> reward 0.0 expected, matches meta)
#   4. gemma tokenizer/template: verify chat template renders without
#      surprises (system+user turns; no <think> injection)
# =============================================================================
