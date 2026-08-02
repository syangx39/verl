#!/usr/bin/env bash
# =============================================================================
# TRL-parity run: verl GRPO on GB200 (A4X), NuminaMath, accounting frozen to
# the Meta engineer's TRL strong-scaling sweep (qwen3_0p6b_strong_scaling.csv)
# =============================================================================
# Same-hardware, cross-framework comparison: TRL (accelerate+FSDP+vLLM) vs
# verl (FSDP2+vLLM, colocated), Qwen3-0.6B, format-only reward.
#
# THE CONTRACT (three blocks + one derived formula):
#   1. Frozen accounting  — 256 prompts x n=8 = 2048 completions/step, and
#      optimizer updates/step matched to TRL's steps_per_generation schedule:
#          SPG = 256 / (2 * W)          (TRL derivation, W = total GPUs)
#          updates/step (verl) = SPG  =>  ppo_mini_batch_size = 256 / SPG
#      computed below from W. DO NOT hand-set the mini batch.
#      Unit confirmed (yaml note): prompts. Formula final.
#   2. Values confirmed from configs/grpo_qwen3_0p6b.yaml (received) — see
#      the "confirmed values" section. per_device unit also CONFIRMED by the
#      yaml's own note: generation_batch_size counts PROMPTS (256 prompts ->
#      2048 completions/rollout, verified against trl grpo_config.py).
#   3. System adaptations — verl-native, deliberately NOT mirrored from TRL
#      (attention backend, packing, memory fractions). Recorded in env table.
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

########################### frozen accounting (derived from W) ##############
train_batch_size=256           # [TRL] generation_batch_size=256 prompts
rollout_n=8                    # [TRL] G=8 -> 2048 completions/step

if (( 256 % (2 * W) != 0 )); then
  echo "ERROR: 256 not divisible by 2*W=$((2*W)); TRL sweep only defines W in {4,8,16,32,64,128}" >&2
  exit 1
fi
SPG=$(( 256 / (2 * W) ))       # [TRL] steps_per_generation = 256/(2W)
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
rollout_tp=1                   # [free var] TRL uses vllm TP=2 (W/2 engines),
                               # mem_util 0.4; verl native: TP=1 per-GPU
                               # engines, 0.30. Recorded in env table.
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
# Open items:
#   1. RESOLVED: yaml values filled; unit confirmed (prompts) per yaml note
#   2. verl mini-batch update order differs from TRL's SPG loop in WHICH
#      completions each update sees (verl: contiguous slices of one shuffled
#      batch; TRL: sequential micro-batches). Same count, same data, same
#      staleness structure (both progressively off-policy within the step);
#      exact per-update sample assignment is not reproducible — document as
#      known delta, analogous to the data-order delta in the MaxText line.
#   3. gen-time comparability: cap now matched at 8192 (yaml-confirmed)
# =============================================================================
