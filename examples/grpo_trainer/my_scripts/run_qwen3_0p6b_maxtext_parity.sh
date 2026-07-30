#!/usr/bin/env bash
# =============================================================================
# GRPO parity run: verl on GB200 (A4X), semantics frozen to the TPU MaxText run
# =============================================================================
# Forked from examples/grpo_trainer/run_qwen3_8b_fsdp.sh (verl upstream).
# Source of truth for every algorithm-semantic value: the TPU team's MaxText
# JobSet (rl-qwen3-21624). This script IS the executable form of the planning
# doc's "Frozen Invariants" table (§3.1) — review by diffing against the
# MaxText command line.
#
# Legend for annotations below:
#   [MAXTEXT]  value copied from the MaxText script (frozen invariant)
#   [GB200]    hardware adaptation inherited from upstream MACHINE=gb200 branch
#   [SYS]      system-side choice, no algorithm-semantics impact (documented)
#   [TODO]     pending confirmation (Tunix source / TPU team)
#
# Deltas vs upstream example (run_qwen3_8b_fsdp.sh), summary:
#   1. Model: Qwen3-8B -> Qwen3-0.6B                       [MAXTEXT]
#   2. Data: gsm8k+math -> OpenMathInstruct-2 (preprocessed) [MAXTEXT]
#   3. batch 1024->480, mini_batch 256->480 (mu=1, on-policy) [MAXTEXT]
#   4. lengths 1024/2048 -> 8192/8192                       [MAXTEXT]
#   5. rollout_n 5->8, TP 2->1                              [MAXTEXT]
#   6. KL coef 0.001->0.05                                  [MAXTEXT]
#   7. sampling: verl defaults (1.0/1.0/-1) -> 0.8/0.95/50  [MAXTEXT]
#      (upstream example never sets these — silent killer for parity)
#   8. max_num_batched_tokens: verl default 8192 -> 32768   [MAXTEXT]
#   9. prefix caching explicitly ON (MaxText hardcodes it)  [MAXTEXT]
#  10. custom reward fn replacing built-in gsm8k routing     [MAXTEXT]
#  11. epochs-driven loop -> step-driven (total_training_steps) [SYS]
#  12. NPU branch, INFER_BACKEND switch, MACHINE switch removed —
#      this script targets exactly one configuration          [SYS]
#
# NOT changed (verl defaults that already align, recorded for the doc):
#   - hybrid_engine=True (default)  -> colocated sync loop, matches MaxText
#   - clip_ratio=0.2 (default)      -> matches rl.grpo_epsilon=0.2, symmetric
#   - norm_adv_by_std_in_grpo=True (default) -> matches Tunix GRPO advantage
#   - actor lr=1e-6                 -> matches learning_rate=1e-6
#   - entropy_coeff=0               -> MaxText has no entropy bonus
#   - grad clip 1.0 (verl default clip_grad=1.0) -> matches
#     gradient_clipping_threshold=1.0
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

PROJECT_NAME=${PROJECT_NAME:-maxtext_parity_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_parity_${NNODES}n$(( NNODES * NGPUS_PER_NODE ))g_$(date +%Y%m%d_%H%M)}

########################### frozen invariants — do not tune ################
train_batch_size=480          # [MAXTEXT] batch_size=480
ppo_mini_batch_size=480       # [MAXTEXT] mu=1: one optimizer update per step.
                              # Upstream example used 256 (-> 4 updates/step =
                              # off-policy); that would silently change the
                              # algorithm. 480 == train_batch_size is the
                              # explicit on-policy setting.
max_prompt_length=8192        # [MAXTEXT] max_prefill_predict_length=8192
max_response_length=8192      # [MAXTEXT] max_target_length(16384) - prefill(8192)
                              # NOTE: Meta A100 script uses 16384/16384 — we
                              # anchor to MaxText, not Meta (planning doc §1.2)
rollout_n=8                   # [MAXTEXT] rl.num_generations=8
kl_loss_coef=0.05             # [MAXTEXT] rl.grpo_beta=0.05
temperature=0.8               # [MAXTEXT] decode_sampling_temperature=0.8
top_p=0.95                    # [MAXTEXT] decode_sampling_nucleus_p=0.95
top_k=50                      # [MAXTEXT] decode_sampling_top_k=50
max_num_batched_tokens=32768  # [MAXTEXT] max_num_batched_tokens=32768
actor_lr=1e-6                 # [MAXTEXT] learning_rate=1e-6

########################### system adaptations ############################
rollout_tp=1                  # [MAXTEXT] rollout_tensor_parallelism... is 8 on
                              # TPU, but that reflects v7x per-chip HBM; 0.6B
                              # needs no TP on GB200 (192GB HBM). TP is a
                              # [SYS] free variable per doc §1.2 — each side
                              # uses its natural parallelism. Meta also runs TP=1.
rollout_gpu_mem_util=0.30     # [SYS] colocated HBM split. MaxText uses 0.22 on
                              # v7x; exact fraction is hardware-dependent, not
                              # semantic. 0.30 leaves ample room for 0.6B FSDP.
ppo_max_token_len_per_gpu=32768  # [SYS] dynamic-bsz packing budget for the
                              # training pass (prompt+response=16384 -> holds
                              # 2 full-length seqs). Tune freely; throughput
                              # only, no semantics.

########################### launch ########################################
# [GB200] block inherited from upstream MACHINE=gb200 (PR #5596):
#   - enforce_eager=True: vLLM CUDA graphs off on SM100. KNOWN PERF TAX on
#     rollout; keep for bring-up, file follow-up to re-enable on newer vLLM
#     (planning doc §4 risk item — do NOT quote parity numbers as final
#     until this is either removed or measured).
#   - free_cache_engine=True: release vLLM KV between steps (colocated HBM
#     hygiene).
#   - model_dtype=bfloat16: FSDP master/compute dtype pinned. [MAXTEXT] is
#     also bf16 -> parity precision.
#   - ray_init.num_gpus pinned: privileged/enroot containers break Ray GPU
#     autodetect.

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
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE} \
    "$@"

# =============================================================================
# Open items pinned in this script (grep TODO):
#   1. kl_loss_type=low_var_kl kept from upstream; verify against Tunix's
#      grpo_beta KL estimator (k1/k3/low-var) and change if mismatched.
#      -> affects curve overlay, not step time.
#   2. loss_agg_mode: verl default token-mean; confirm Tunix rl.loss_agg_mode
#      default. If Tunix differs, add ++actor_rollout_ref.actor.loss_agg_mode.
#   3. MaxText applies its own prompt-length filter (tokenize<=8192) at data
#      prep; our preprocess script must replicate it so both sides train on
#      the identical prompt set (same filter, same shuffle seed semantics
#      are NOT reproducible across frameworks — document as known delta).
#   4. save_freq=-1 (checkpointing off) for perf runs; MaxText smoke run has
#      checkpoint_period=20 i.e. effectively once at end. For measured runs
#      both sides must exclude checkpoint steps from the timing window (§4.1).
#   5. Option name drift across verl versions (e.g. engine_kwargs path,
#      total_training_steps): validated against main (0.9.0.dev) tree; re-check
#      after pinning the release tag.
# =============================================================================
