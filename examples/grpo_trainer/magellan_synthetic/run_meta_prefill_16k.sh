#!/usr/bin/env bash
# =============================================================================
# Meta prefill-heavy synthetic workload: 16K input / 8K output, GB200 (A4X)
# =============================================================================
# SEPARATE from run_qwen3_0p6b_maxtext_parity.sh — different workload, different
# anchor. This one reproduces the customer's (Magellan) prompt-length profile
# with synthetic ranking data and a dummy diversity reward. NOT comparable to
# the MaxText/OpenMathInstruct parity runs.
#
# Workload definition:
#   - data: ads_synthetic_16k_typical (p50 ~15K input tokens, cap 16384)
#   - output: up to 8192 tokens
#   - reward: dummy diversity ranking reward (pipeline/perf testing only)
#   - execution: colocated sync (same as parity bring-up; customer-mode
#     disagg variant comes later)
#
# Key config differences vs the parity script, and why:
#   - max_prompt_length 16640 (=16384 cap + 256 rendering margin),
#     max_response 8192 => verl derives vLLM max_model_len = 24832.
#     (Direct rollout.max_model_len override is NOT consumed by this verl
#     version — the derivation is prompt+response, so margin goes on the
#     prompt side. See inline comment at max_prompt_length.)
#   - max_num_batched_tokens 32768 (kept, now a CRITICAL knob: prefill-heavy
#     means chunked prefill throughput dominates; sweep candidate 16K-64K)
#   - gpu_memory_utilization 0.30 -> 0.50: KV cache must hold ~24K tokens/seq;
#     at 0.6B weights are tiny, give vLLM more room for KV
#   - prefix caching ON (kept): only the shared system prompt (~150 tok) hits;
#     per-row content is random. Within-group (n=8) prompt sharing still
#     caches full prefill — inherent to GRPO, flagged to Meta.
#   - reward: dummy_ranking_reward.py (diversity score; NOTE reward is
#     RL-gameable — response distribution may drift after ~10-20 steps;
#     measure early-window steps)
# =============================================================================

set -xeuo pipefail

########################### paths ###########################################
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/magellan_synthetic_16k}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/verl/examples/grpo_trainer/my_scripts/dummy_ranking_reward.py}
MODEL_PATH=${MODEL_PATH:-$HOME/meta-RL/models/Qwen3-0.6B}

########################### scale knobs #####################################
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TOTAL_STEPS=${TOTAL_STEPS:-20}
PROJECT_NAME=${PROJECT_NAME:-magellan_prefill16k}
MODEL_TAG=$(basename ${MODEL_PATH} | tr 'A-Z' 'a-z' | tr '.-' '__')
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_TAG}_prefill16k_${NNODES}n$(( NNODES * NGPUS_PER_NODE ))g_$(date +%Y%m%d_%H%M)}

########################### workload parameters #############################
train_batch_size=${TRAIN_BATCH_SIZE:-480}
ppo_mini_batch_size=${TRAIN_BATCH_SIZE:-480}   # mu=1 preserved
max_prompt_length=16640        # Magellan spec: 16K input, +256 margin.
                               # Dataset is filtered to <=16384 by OUR
                               # converter's rendering; verl's own chat-
                               # template rendering adds up to ~72 tokens
                               # (kwargs differ), and verl derives vLLM's
                               # max_model_len as prompt+response — so the
                               # margin must live HERE for the derived
                               # budget (16640+8192=24832) to admit the
                               # worst verl-rendered prompt (~16424)+8192.
                               # Canonical dataset definition is UNCHANGED
                               # (<=16384 at converter rendering).
max_response_length=8192       # Magellan spec: 8K output
rollout_n=8
kl_loss_coef=0.05
temperature=0.8
top_p=0.95
top_k=50
actor_lr=1e-6

########################### system ##########################################
rollout_tp=1
rollout_gpu_mem_util=0.50      # KV for 24K-token seqs; 0.6B weights tiny
max_num_batched_tokens=32768   # prefill chunk budget — PRIMARY sweep knob here
ppo_max_token_len_per_gpu=49152  # train-pass packing: 2x 24K seqs

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
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    custom_reward_function.name=dummy_reward \
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
#   1. Confirm with Meta/Matt: dataset is canonical (single source for TPU+GPU)
#   2. KV budget check at batch 480: 480 prompts x ~15K tokens resident during
#      a step's prefill wave — watch preemption/recompute counters in vLLM logs;
#      if preemption is high, raise gpu_mem_util or lower max_num_seqs
#   3. Dummy reward is gameable: response-length distribution may drift after
#      ~10-20 steps; quote step times from the early window and report drift
#   4. batch 480 inherited from parity runs for comparability of scale, NOT a
#      Magellan-confirmed value — confirm their intended batch size
# =============================================================================
