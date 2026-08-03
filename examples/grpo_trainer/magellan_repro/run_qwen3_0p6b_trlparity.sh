#!/usr/bin/env bash
# =============================================================================
# TRL-parity run: verl GRPO on GB200 (A4X), NuminaMath — Meta metaface
# strong-scaling counterpart (round-2 / pd=8 accounting)
# =============================================================================
# Same-hardware, cross-framework comparison: metaface (TRL+accelerate+FSDP+
# vLLM) vs verl (FSDP2+vLLM, colocated), Qwen3-0.6B, format-only reward.
#
# THE CONTRACT (four categories):
#   1. Frozen invariant — global batch: 256 prompts x n=8 = 2048 completions
#      per rollout (generation_batch_size counts PROMPTS, per the yaml's own
#      note). Any change here invalidates the comparison.
#   2. Yaml-confirmed values — lengths/sampling/beta/lr/steps from
#      configs/grpo_qwen3_0p6b.yaml; every such line tagged [YAML] below.
#   3. System knobs — free per stack, recorded in the env table:
#        * Update schedule (MATCHED per comparison row, not frozen):
#          Meta round-2 accounting, pd=8:
#              SPG = 256 / (8 * W), floored at 1
#              ppo_mini_batch_size = 256 / SPG
#          => mini 32/64/128/256/256/256 at W=4/8/16/32/64/128.
#          Derived below from W; resolved accounting printed at launch.
#        * Rollout engine: Meta TP=2 (mem 0.4, Triton-attn); verl TP=1
#          per-GPU engines (mem 0.30, FlashInfer). TP=2 control runs at
#          W<=16 via rollout_tp override — each side reports its own best.
#        * Training-pass chunking: use_dynamic_bsz splits large mini-batches
#          into gradient-accumulation chunks; final gradient identical.
#   4. Scale knobs — NNODES / NGPUS_PER_NODE / TOTAL_STEPS env vars.
# =============================================================================

set -xeuo pipefail

########################### paths ###########################################
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/numina_verl}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/data/magellan_repro/format_reward_verl.py}
MODEL_PATH=${MODEL_PATH:-$HOME/meta-RL/models/Qwen3-0.6B}

########################### scale knobs #####################################
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TOTAL_STEPS=${TOTAL_STEPS:-40}   # [YAML] max_steps: 40
W=$(( NNODES * NGPUS_PER_NODE ))
PROJECT_NAME=${PROJECT_NAME:-numina_trlparity}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_trlparity_${NNODES}n${W}g_$(date +%Y%m%d_%H%M)}

########################### accounting (update schedule; derived from W) ####
train_batch_size=256           # [FROZEN] generation_batch_size=256 prompts
rollout_n=8                    # [FROZEN] G=8 -> 2048 completions/rollout

# pd=8 accounting (Meta round-2): SPG = 256/(pd*W), pd=8, floored at 1.
# W in {4,8,16,32,64,128}; at W>=32 the floor engages (SPG=1, mini=256).
if (( W < 4 || 256 % W != 0 )); then
  echo "ERROR: W=${W} not in the sweep {4,8,16,32,64,128}" >&2
  exit 1
fi
SPG=$(( 256 / (8 * W) )); (( SPG < 1 )) && SPG=1
ppo_mini_batch_size=$(( 256 / SPG ))   # verl: updates/step = 256/mini = SPG
echo "[accounting] W=${W}  SPG=${SPG}  ppo_mini_batch_size=${ppo_mini_batch_size} (=> ${SPG} optimizer updates/step)"

########################### confirmed values (grpo_qwen3_0p6b.yaml) ########
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}     # [YAML] max_prompt_length: 16384
                                                  # (cap; numina prompts ~170 tok, so
                                                  # runtime filter is effectively no-op)
max_response_length=${MAX_RESPONSE_LENGTH:-8192}  # [YAML] max_completion_length: 8192.
                                                  # Format reward runs toward cap — gen
                                                  # load scales with this; MUST match for
                                                  # gen-time comparability
temperature=${TEMPERATURE:-1.0}                   # [YAML] temperature: 1.0
top_p=${TOP_P:-1.0}                               # [YAML] top_p: 1.0
top_k=${TOP_K:--1}                                # [YAML] not set -> unrestricted
kl_loss_coef=${KL_COEF:-0.0}                      # [YAML] beta: 0.0 (no KL, no ref model)
actor_lr=${ACTOR_LR:-1e-6}                        # yaml unset -> TRL default 1e-6
USE_KL=${USE_KL:-False}                           # tied to beta=0.0

########################### system adaptations (verl-native) ################
rollout_tp=${ROLLOUT_TP:-1}    # [knob] Meta: TP=2 (W/2 engines, mem 0.4,
                               # Triton-attn); verl default: TP=1 per-GPU
                               # engines. Set ROLLOUT_TP=2 for the control
                               # ladder at W<=16. Each side reports its best.
rollout_gpu_mem_util=0.30
max_num_batched_tokens=8192
ppo_max_token_len_per_gpu=16384
# NOTE: TRL forces VLLM_ATTENTION_BACKEND=TRITON_ATTN on GB200; verl keeps
# its default (flashinfer). Deliberate free variable — do not mirror.

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
# Notes:
#   1. ACCOUNTING VERSION: this script implements the pd=8 (round-2)
#      schedule. Data measured under the earlier pd=2 schedule is archived
#      and labeled; never mix accountings within a comparison row.
#   2. Known delta: verl mini-batch update order differs from TRL's SPG
#      loop in WHICH completions each update sees (verl: contiguous slices
#      of one shuffled batch; TRL: sequential micro-batches). Same count,
#      same data, same progressive off-policy structure; per-update sample
#      assignment not reproducible across frameworks.
#   3. gen-time comparability: caps matched (8192) per yaml.
# =============================================================================
