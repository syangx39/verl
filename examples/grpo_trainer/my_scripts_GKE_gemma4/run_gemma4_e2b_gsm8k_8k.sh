#!/usr/bin/env bash
# =============================================================================
# google/gemma-4-E2B-it on GSM8K, colocated, 8K response cap, no length penalty (PR #14, GSM8K_8K_PLAN.md).
#
# Recipe = the customer-delivered Qwen3-0.6B GSM8K 8K recipe `gsm8k_8k_v1`
# (../my_scripts_GKE_gsm8k/TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md, Tier 1), transferred to Gemma:
#   data      GSM8K boxed train 7,473 / full test 1,319 (handoff package, same question IDs and gold answers);
#             system + user messages rendered ONCE by Gemma's own chat template (it accepts the system role)
#   batch     128 prompts x 16 completions = 2,048; mini == batch -> one optimizer update per rollout, 1 epoch
#   loss      single-forward REINFORCE with detached token TIS: -A * w * log pi, w = min(exp(clamp(logp_train.detach()
#             - logp_sampler, +-20)), 3.0); no PPO/dual clip, no IS batch normalization, no rejection; token-mean over
#             all valid response tokens of the global batch; GRPO advantage (sample std + 1e-6); no KL, no entropy bonus
#   optimizer AdamW lr 2e-6, betas (0.9, 0.999), eps 1e-8 (torch default), weight decay 0.0, grad clip 1.0;
#             linear warmup 10 (update 1 at LR 0, update 2 at 2e-7), cosine to 0 at step 250
#   sampling  T=1, top_p=1, top_k=-1 (no truncation), response cap 8,192 tokens, prompt cap 512 (all fit: max 249)
#   reward    boxed_math_reward_v2.py (scorer v2, PR #14 option B; train and eval use the same version):
#             correct 1 / parseable wrong 0.1 / unparseable 0; NO length penalty (REWARD_PENALTY_SOURCES empty).
#             v2 = Meta boxed_math (v1, boxed_math_reward.py, unchanged; the Qwen runs trained with it) + one explicit
#             rule crediting decorated numbers with the exact gold value ($18, 18%, 540 \text{ meters}, 28.00).
#             The v1 accuracy of every response is logged next to it as acc_v1 (diagnostic only)
#   eval      full 1,319, greedy, 8K cap, steps 0, 20, ..., 240, 250
#   ckpt      full checkpoints (model, optimizer, extra, dataloader) only at the steps in CKPT_STEPS (e.g. 100,250):
#             save_freq must reach every listed step and the gemma4_ckpt_steps hook skips the others
# Gemma system settings (validated on the OMI2 line, run_gemma4_e2b_rl05_parity.sh): FP32 persistent weights and
# Adam moments, BF16 compute / rollout, flex_attention, RoPE fix + per-rank guard via node prep every launch,
# dynamic micro-batching 16,384 tokens / GPU (memory only; batch, update count and denominator unchanged).
#
# Usage (Ray head pod): SEED=1 bash run_gemma4_e2b_gsm8k_8k.sh [extra hydra overrides]
#   TOTAL_STEPS stays 250 even for smokes: the trainer sets the LR schedule horizon to the run length, so a short
#   smoke is a 250-step run stopped early (STOP_AFTER_STEPS in g4run_gsm8k.sh), never TOTAL_STEPS=<small>.
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Deliberately NOT ${DATA_DIR:-...} / ${MODEL_PATH:-...}: setup_env.sh exports both for the Qwen line.
DATA_DIR=${GSM8K_DATA_DIR:-/workspace/meta-RL/handoff/gsm8k_8k/data}
TRAIN_FILE=$DATA_DIR/gsm8k_boxed_train.parquet
VAL_FILE=$DATA_DIR/gsm8k_boxed_test.parquet
MODEL_PATH=${GEMMA_MODEL_PATH:-/workspace/meta-RL/models/gemma-4-E2B-it}
GSM8K_SCRIPTS=$(cd "$SCRIPT_DIR/../my_scripts_GKE_gsm8k" && pwd)
REWARD_FN_PATH=${GSM8K_REWARD_FN:-$GSM8K_SCRIPTS/boxed_math_reward_v2.py}
EXPECTED_REWARD_SHA256=933abb3e7d0b58dce812e1b4a01d0f15bb3ae7054b4c9956d985434a4d01938b  # scorer v2
REWARD_V1_PATH=$GSM8K_SCRIPTS/boxed_math_reward.py      # imported by v2 (it re-checks this hash at import)
EXPECTED_REWARD_V1_SHA256=a743cf71e2094b0ebee67506460028b5a27bb9b1d86928f6c19b152561c7015b  # == handoff code/ copy
LOG_ROOT=${LOG_ROOT:-/workspace/meta-RL/logs}
CKPT_ROOT=${CKPT_ROOT:-/workspace/meta-RL/ckpt}

########################### run identity ####################################
SEED=${SEED:-1}
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
W=$(( NNODES * NGPUS_PER_NODE ))
TOTAL_STEPS=${TOTAL_STEPS:-250}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
CKPT_STEPS=${CKPT_STEPS:-}       # e.g. 100,250: save only these steps (empty = verl's save_freq behaviour)
CKPT_CONTENTS=${CKPT_CONTENTS:-'["model","optimizer","extra"]'}
PROJECT_NAME=gemma4_gsm8k_8k
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_e2b_gsm8k8k_${NNODES}n${W}g_s${TOTAL_STEPS}_seed${SEED}_$(date -u +%m%d_%H%M)}
DUMP_DIR=${LOG_ROOT}/gemma4_gsm8k/${EXPERIMENT_NAME}
GUARD_DIR=${LOG_ROOT}/gemma4_guard/${EXPERIMENT_NAME}
mkdir -p "${DUMP_DIR}/train" "${DUMP_DIR}/val" "${GUARD_DIR}"

########################### frozen recipe (gsm8k_8k_v1) ######################
train_batch_size=128
rollout_n=16
ppo_mini_batch_size=128          # x n = 2,048 = full batch -> exactly one optimizer update per rollout
max_prompt_length=512
max_response_length=8192
actor_lr=2e-6
lr_warmup_steps=10
weight_decay=0.0
grad_clip=1.0
temperature=1.0
top_p=1.0
top_k=-1
IS_THRESHOLD=3.0
# Reward knobs: always set here so OMI2 / Qwen-2K values inherited from the shell cannot leak in.
export REWARD_FORMAT_SCORE=0.1
export REWARD_PENALTY_SOURCES=""      # empty = no length penalty anywhere (train and eval)
export REWARD_MAX_RESP_LEN=${max_response_length}
export REWARD_OVERLONG_BUFFER=512     # inert while REWARD_PENALTY_SOURCES is empty
export REWARD_OVERLONG_PENALTY=1.0    # inert while REWARD_PENALTY_SOURCES is empty
unset REWARD_FMT_WEIGHT REWARD_FMT_WEIGHT_TRAIN REWARD_EVAL_QUESTIONS REWARD_DUMP_DIR REWARD_DUMP_EVERY 2>/dev/null || true

########################### system (Gemma line) ##############################
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-16384}
rollout_gpu_mem_util=${ROLLOUT_MEM:-0.30}
max_num_batched_tokens=32768

########################### sanity ##########################################
(( (train_batch_size * rollout_n) % W == 0 )) || { echo "ERROR: 2048 completions not divisible by W=${W}" >&2; exit 1; }
for f in "$TRAIN_FILE" "$VAL_FILE" "$REWARD_FN_PATH" "$REWARD_V1_PATH" "$MODEL_PATH/config.json"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
REWARD_SHA256=$(sha256sum "$REWARD_FN_PATH" | cut -d' ' -f1)
[ "$REWARD_SHA256" = "$EXPECTED_REWARD_SHA256" ] || { echo "ERROR: reward sha256 $REWARD_SHA256 != $EXPECTED_REWARD_SHA256" >&2; exit 1; }
REWARD_V1_SHA256=$(sha256sum "$REWARD_V1_PATH" | cut -d' ' -f1)
[ "$REWARD_V1_SHA256" = "$EXPECTED_REWARD_V1_SHA256" ] || { echo "ERROR: v1 reward sha256 $REWARD_V1_SHA256 != $EXPECTED_REWARD_V1_SHA256" >&2; exit 1; }
if [ -n "$CKPT_STEPS" ]; then
  [ -z "${OPTIM_PROBE_DIR:-}" ] || { echo "ERROR: CKPT_STEPS and OPTIM_PROBE_DIR both need Ray's single worker_process_setup_hook" >&2; exit 1; }
  (( SAVE_FREQ > 0 )) || { echo "ERROR: CKPT_STEPS needs SAVE_FREQ > 0 (got ${SAVE_FREQ})" >&2; exit 1; }
  for s in ${CKPT_STEPS//,/ }; do
    (( s >= 1 && s <= TOTAL_STEPS && (s % SAVE_FREQ == 0 || s == TOTAL_STEPS) )) || {
      echo "ERROR: CKPT_STEPS entry ${s} is never reached with SAVE_FREQ=${SAVE_FREQ}, TOTAL_STEPS=${TOTAL_STEPS}" >&2; exit 1; }
  done
fi

########################### preflight (no GPU) ##############################
# data / template / scorer / schedule checks; aborts on any failure. Writes $DUMP_DIR/preflight.json.
python3 "$SCRIPT_DIR/gemma4_gsm8k_preflight.py" --data-dir "$DATA_DIR" --model "$MODEL_PATH" --reward "$REWARD_FN_PATH" \
    --scorer-version 2 --max-prompt "$max_prompt_length" --cap "$max_response_length" --total-steps "$TOTAL_STEPS" \
    --warmup "$lr_warmup_steps" --peak-lr "$actor_lr" --out "$DUMP_DIR/preflight.json"

########################### node prep (every launch) #########################
LAUNCHER_SHA256=$(sha256sum "$0" | cut -d' ' -f1)
echo "[gsm8k-provenance] exp=${EXPERIMENT_NAME} seed=${SEED} W=${W} total_steps=${TOTAL_STEPS} test_freq=${TEST_FREQ}" \
     "save_freq=${SAVE_FREQ} ckpt_steps=${CKPT_STEPS:-all} launcher_sha256=${LAUNCHER_SHA256}" \
     "reward=${REWARD_FN_PATH} reward_sha256=${REWARD_SHA256} reward_v1_sha256=${REWARD_V1_SHA256}" \
     "train_sha256=$(sha256sum "$TRAIN_FILE" | cut -d' ' -f1) test_sha256=$(sha256sum "$VAL_FILE" | cut -d' ' -f1)" \
     "model_safetensors_sha256=$(sha256sum "$MODEL_PATH/model.safetensors" | cut -d' ' -f1) dump_dir=${DUMP_DIR}"
python3 "$SCRIPT_DIR/gemma4_node_prep.py" --patch-dir "$SCRIPT_DIR" --report "$GUARD_DIR/node_prep.json" \
    --min-gpu-nodes "$NNODES" --launcher-sha256 "$LAUNCHER_SHA256"

########################### optional optimizer probe (smoke only) ###########
# Ray accepts ONE worker_process_setup_hook per job: the probe and the checkpoint allowlist are mutually exclusive.
HOOK_ARGS=()
if [ -n "${OPTIM_PROBE_DIR:-}" ]; then
  mkdir -p "$OPTIM_PROBE_DIR"
  HOOK_ARGS=("++ray_kwargs.ray_init.runtime_env.py_modules=[\"$SCRIPT_DIR/gemma4_optim_probe\"]"
             "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gemma4_optim_probe.install"
             "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OPTIM_PROBE_DIR=${OPTIM_PROBE_DIR}"
             "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OPTIM_PROBE_STEPS=\"${OPTIM_PROBE_STEPS:-3}\"")
fi
########################### checkpoint step allowlist #######################
if [ -n "$CKPT_STEPS" ]; then
  HOOK_ARGS=("++ray_kwargs.ray_init.runtime_env.py_modules=[\"$SCRIPT_DIR/gemma4_ckpt_steps\"]"
             "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=gemma4_ckpt_steps.install"
             "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_CKPT_STEPS=\"${CKPT_STEPS}\""
             "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_CKPT_STEPS_LOG=${DUMP_DIR}/ckpt_steps.log")
fi

RAY_NUM_GPUS_ARG=""
[ -z "${RAY_ADDRESS:-}" ] && RAY_NUM_GPUS_ARG="+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"

set +e
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=${IS_THRESHOLD} \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=reinforce \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${VAL_FILE}']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=${SEED} \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flex_attention \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.actor.optim.zero_indexed_step=true \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}' \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.fsdp_config.seed=${SEED} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.checkpoint.save_contents="${CKPT_CONTENTS}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.seed=${SEED} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.use_v1=False \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="${CKPT_ROOT}/${EXPERIMENT_NAME}" \
    trainer.resume_mode=disable \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${DUMP_DIR}/val" \
    trainer.rollout_data_dir="${DUMP_DIR}/train" \
    trainer.total_epochs=100 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_FORMAT_SCORE=\"${REWARD_FORMAT_SCORE}\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_PENALTY_SOURCES=''" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MAX_RESP_LEN=\"${REWARD_MAX_RESP_LEN}\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_BUFFER=\"${REWARD_OVERLONG_BUFFER}\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_PENALTY=\"${REWARD_OVERLONG_PENALTY}\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_NO_USAGE_STATS=\"1\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.DO_NOT_TRACK=\"1\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_GEMMA4_ROPE_REPORT_DIR=${GUARD_DIR}" \
    ${RAY_NUM_GPUS_ARG} \
    ${HOOK_ARGS[@]+"${HOOK_ARGS[@]}"} \
    "$@"
rc=$?
set -e
echo "[driver] main_ppo finished with exit_code=${rc}"
exit "${rc}"
