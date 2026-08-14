#!/usr/bin/env bash
# =============================================================================
# rl-05 parity: verl GRPO(DAPO) on GB200/GKE, semantics matched to the TPU
# team's tianyu-rl-05 (their reproduction of Magellan's 299 s tpu7x config;
# measured 274.9 s median, 64 chips split 32+32 rollout/trainer, serial)
# =============================================================================
# Derived from run_qwen3_0p6b_maxtext_parity.sh. Deltas vs that (line-1) config:
#   1. batch 480 -> 256 prompts (x8 = 2048 completions)      [RL05]
#   2. mini 480 -> 256 (mu=1 preserved: ONE update/rollout)  [RL05]
#   3. beta 0.05 -> 0: use_kl_loss=False, NO ref model, the
#      ref/refer_inference phase disappears on both stacks   [RL05]
#   4. clip_ratio_high=0.28 (DAPO clip-higher; low stays 0.2)[RL05]
#   5. test_freq 5 -> -1 (pure timing run, matches TPU side) [RL05]
# Unchanged and already matched: response cap 8192, sampling 0.8/50/0.95,
# lr 1e-6, grad clip 1.0, reward=maxtext_math_reward (utils_rl.py port),
# dataset OpenMathInstruct-2, 20 steps.
# Known deltas (documented, not blockers): prompt-cap budget (ours 8192+margin
# vs their prefill 8192 -- non-binding either way, isl_max ~631); data split/
# order not reproducible across frameworks; their ~30 s unaccounted phase
# (likely logprob) means phase tables don't sum -- compare full-step wall.
# Comparison anchor: MaxText 274.9 s median (n=8 warm), 64 tpu7x chips
# (32 rollout + 32 trainer, serial); phases rollout ~193 / train 48.3 /
# sync 2.9 / ~30 unaccounted.
# =============================================================================

set -xeuo pipefail

########################### paths (site-specific) ###########################
# Preprocessed OpenMathInstruct-2 parquet: MUST be built with the same chat
# template / prompt format as the MaxText data template (see preprocess
# script; token-identical prompts are a precondition for curve overlay).
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/openmathinstruct2}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}

# Reward: port of MaxText's default stack (match_format_exactly +
# match_format_approximately + check_numbers), single compute_score entry.
REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/reward/maxtext_math_reward.py}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}   # [MAXTEXT] model_name=qwen3-0.6b
                                            # [TODO] confirm exact HF revision
                                            # matches TPU checkpoint source

########################### scale knobs (only these vary between runs) ######
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}         # [GB200] A4X: 4 GPUs/node
TOTAL_STEPS=${TOTAL_STEPS:-20}              # [MAXTEXT] num_batches=20 for
                                            # smoke; raise for convergence runs

PROJECT_NAME=${PROJECT_NAME:-rl05_parity}
W=$(( NNODES * NGPUS_PER_NODE ))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_rl05_${NNODES}n${W}g_tp${ROLLOUT_TP:-1}_$(date +%Y%m%d_%H%M)}

########################### frozen invariants — do not tune ################
train_batch_size=256          # [RL05] batch_size=256
ppo_mini_batch_size=256       # [RL05] mu=1: one optimizer update per rollout
                              # (mini == batch is the explicit on-policy
                              # setting; their tmbs=16 is grad accumulation
                              # inside that single update, as is our
                              # dynamic-bsz chunking).
max_prompt_length=8192        # [MAXTEXT] max_prefill_predict_length=8192
max_response_length=8192      # [MAXTEXT] max_target_length(16384) - prefill(8192)
                              # NOTE: Meta A100 script uses 16384/16384 — we
                              # anchor to MaxText, not Meta (planning doc §1.2)
rollout_n=8                   # [MAXTEXT] rl.num_generations=8
kl_loss_coef=0.0              # [RL05] rl.grpo_beta=0.0 -> NO KL, NO ref
clip_ratio_low=0.2            # [RL05] rl.grpo_epsilon=0.2
clip_ratio_high=0.28          # [RL05] rl.epsilon_high=0.28 (DAPO clip-higher)
temperature=0.8               # [MAXTEXT] decode_sampling_temperature=0.8
top_p=0.95                    # [MAXTEXT] decode_sampling_nucleus_p=0.95
top_k=50                      # [MAXTEXT] decode_sampling_top_k=50
max_num_batched_tokens=32768  # [MAXTEXT] max_num_batched_tokens=32768
actor_lr=1e-6                 # [MAXTEXT] learning_rate=1e-6

########################### system adaptations ############################
rollout_tp=${ROLLOUT_TP:-1}   # [SYS] free variable. Qwen numina line measured
                              # TP=2 optimal at W=8-32 on GB200; run both at
                              # W=16 and report each side's best.
rollout_gpu_mem_util=0.30     # [SYS] colocated HBM split. MaxText uses 0.22 on
                              # v7x; exact fraction is hardware-dependent, not
                              # semantic. 0.30 leaves ample room for 0.6B FSDP.
ppo_max_token_len_per_gpu=32768  # [SYS] dynamic-bsz packing budget for the
                              # training pass (prompt+response=16384 -> holds
                              # 2 full-length seqs). Tune freely; throughput
                              # only, no semantics.

########################### launch ########################################
# [GB200] block: adapted from upstream MACHINE=gb200 (PR #5596), with two
# measured deviations:
#   - enforce_eager=False: DEVIATION from upstream (which forced eager on
#     SM100). Verified on this image's vLLM: CUDA graphs work on Blackwell.
#     Result: 2.1x gen speedup (1n4g: 700s -> 325s), numerics identical to
#     eager over matched steps (score & length distributions, same data
#     order). [SYS] change, no semantics impact.
#   - free_cache_engine=False: DEVIATION from upstream (which sets True to
#     release vLLM KV between steps for large models). At 0.6B HBM is
#     abundant; the sleep/wake cycle caused the 29s update_weights seen in
#     the smoke run (fix: 29s -> 4s). [SYS] change, no semantics impact.
#   - model_dtype=bfloat16: FSDP master/compute dtype pinned. [MAXTEXT] is
#     also bf16 -> parity precision.
#   - ray_init.num_gpus pinned (single-node only, see below): privileged/
#     enroot containers break Ray GPU autodetect.

# ray_init.num_gpus workaround is only valid when the driver starts its own
# local Ray (single-node; privileged/enroot containers break GPU autodetect).
# When attaching to an existing cluster (RAY_ADDRESS set), Ray forbids
# num_cpus/num_gpus at ray.init() -- resources are reported by each node's
# `ray start --num-gpus`. Inject the flag only in the single-node case.
echo "[accounting] W=${W}  batch=256x8=2048  updates/rollout=1 (mini=256)  beta=0  clip=0.2/0.28  rollout_tp=${ROLLOUT_TP:-1}"

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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
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
    custom_reward_function.name=compute_score \
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
#   1. loss_agg_mode: verl default token-mean; Tunix default unconfirmed.
#      Affects curve overlay, not step time.
#   2. Data split/order: same source dataset, split implementations differ
#      (make_tpu_split.py cross-check still pending). OSL distribution
#      statistics (mean/cap%) are the cross-check: theirs 3,750/22-25%%,
#      ours ~3,950/25%% on the numina regime -- expect similar here.
#   3. Their ~30 s unaccounted phase (global - rollout - train - sync):
#      likely old-logprob + advantage. We itemize old_log_prob; when
#      comparing phase tables, compare full-step wall first.
# =============================================================================
