#!/usr/bin/env bash
# =============================================================================
# gemma-4-E2B-it on the rl-05 parity workload: verl GRPO(DAPO) on GB200/GKE.
# CONTRACT = run_qwen3_0p6b_rl05_parity.sh (frozen block copied verbatim);
# MODEL-SPECIFIC [SYS] lines = inherited from run_gemma4_2b_decodelight.sh.
# =============================================================================
# Frozen semantics (identical to the Qwen rl-05 line -- do not tune):
#   256 prompts x n=8 = 2048 completions/step; mini == batch (mu=1, ONE
#   optimizer update per rollout); beta=0 (no KL, no ref model);
#   clip 0.2/0.28 (DAPO clip-higher); prompt cap 8192 / response cap 8192;
#   sampling 0.8/50/0.95; lr 1e-6; grad clip 1.0 (verl default);
#   reward = maxtext_math_reward.compute_score; dataset OpenMathInstruct-2.
# TPU anchor for the GEMMA line: [TBD -- fill in the tianyu run id and its
#   config once available; every [RL05]/[MAXTEXT] tag below inherits from
#   the Qwen line and must be re-verified against the Gemma TPU config].
# Model: google/gemma-4-E2B-it, FUSE-mounted at /workspace/meta-RL/models/
#   gemma-4-E2B-it (no download; HF_HUB_OFFLINE=1). [TODO] confirm HF
#   revision matches the TPU checkpoint source.
#   NOTE: the model path variable is GEMMA_MODEL_PATH, NOT MODEL_PATH --
#   setup_env.sh exports MODEL_PATH=.../Qwen3-0.6B for the Qwen line, and a
#   ${MODEL_PATH:-...} default would silently run Qwen under a Gemma name.
# =============================================================================
# What is DIFFERENT from the Qwen script, and why (all [SYS]/[GEMMA]):
#   - Data: SAME parquet only if it stores raw messages (verl applies the
#     model's chat template at load time). PREFLIGHT below renders one
#     sample through the Gemma tokenizer and fails loudly on template
#     errors (e.g. unsupported `system` role) before touching any GPU.
#   - ppo_max_token_len_per_gpu 32768 -> 16384: Gemma's ~262k vocab makes
#     the [tokens, vocab] logits tensor ~1.7x Qwen's; 16384 still holds one
#     full-length (8192+8192) sequence, which is the hard minimum. Raise it
#     back if actor/perf/max_memory_reserved_gb leaves headroom.
#   - Attention backend: nothing to set. vLLM detects Gemma4's heterogeneous
#     head dims (256 local / 512 global) and FORCES TRITON_ATTN itself
#     (log line "Forcing TRITON_ATTN backend"). Record in the doc as a
#     shared constraint by construction, not a free variable.
#   - prefix caching kept =True for config parity; the workload has no
#     shared prefixes, so if Gemma's sliding-window attention rejects it,
#     PREFIX_CACHING=False costs nothing.
# =============================================================================

set -xeuo pipefail

########################### paths (site-specific) ###########################
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/openmathinstruct2}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
PROMPT_KEY=${PROMPT_KEY:-prompt}                # parquet column read by PREFLIGHT only
                                                # (== verl's data.prompt_key default)

REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/reward/maxtext_math_reward.py}

# Deliberately NOT ${MODEL_PATH:-...}: setup_env.sh exports MODEL_PATH for Qwen.
MODEL_PATH=${GEMMA_MODEL_PATH:-/workspace/meta-RL/models/gemma-4-E2B-it}   # [GEMMA]

########################### scale knobs (only these vary between runs) ######
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}         # [GB200] A4X: 4 GPUs/node
TOTAL_STEPS=${TOTAL_STEPS:-20}

PROJECT_NAME=${PROJECT_NAME:-rl05_parity_gemma4}
W=$(( NNODES * NGPUS_PER_NODE ))
rollout_tp=${ROLLOUT_TP:-1}                 # [SYS] free variable
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_e2b_rl05_${NNODES}n${W}g_tp${rollout_tp}_$(date +%Y%m%d_%H%M)}

########################### frozen invariants — do not tune ################
train_batch_size=256          # [RL05] batch_size=256
ppo_mini_batch_size=256       # [RL05] mu=1: one optimizer update per rollout
max_prompt_length=8192        # [MAXTEXT] max_prefill_predict_length=8192
max_response_length=8192      # [MAXTEXT] max_target_length(16384) - prefill(8192)
rollout_n=8                   # [MAXTEXT] rl.num_generations=8
kl_loss_coef=0.0              # [RL05] beta=0 -> NO KL, NO ref
clip_ratio_low=0.2            # [RL05] grpo_epsilon=0.2
clip_ratio_high=0.28          # [RL05] epsilon_high=0.28 (DAPO clip-higher)
temperature=0.8               # [MAXTEXT] decode_sampling_temperature=0.8
top_p=0.95                    # [MAXTEXT] decode_sampling_nucleus_p=0.95
top_k=50                      # [MAXTEXT] decode_sampling_top_k=50
max_num_batched_tokens=32768  # [MAXTEXT] max_num_batched_tokens=32768
actor_lr=1e-6                 # [MAXTEXT] learning_rate=1e-6

########################### system adaptations ############################
rollout_gpu_mem_util=${ROLLOUT_MEM:-0.30}        # [SYS] colocated HBM split; 0.30 verified
                                                 # for this model on the decode-light line,
                                                 # but 8192-cap KV is far larger: if
                                                 # agent_loop/num_preempted > 0, raise it.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-16384}  # [SYS][GEMMA] 262k vocab; see header
PREFIX_CACHING=${PREFIX_CACHING:-True}           # [SYS] parity default; workload-neutral
REMOVE_PADDING=${REMOVE_PADDING:-True}           # [SYS] ran on gemma-4 decode-light line
PREFLIGHT=${PREFLIGHT:-1}                        # 1 = render one sample through the tokenizer

########################### sanity ########################################
if (( (train_batch_size * rollout_n) % W != 0 )); then
  echo "ERROR: 2048 completions not divisible by W=${W}" >&2; exit 1
fi
for f in "$TRAIN_FILE" "$VAL_FILE" "$REWARD_FN_PATH"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$MODEL_PATH" ] || { echo "ERROR: missing model dir $MODEL_PATH" >&2; exit 1; }

echo "[accounting] model=${MODEL_PATH}  W=${W}  batch=256x8=2048  updates/rollout=1 (mini=256)  beta=0  clip=0.2/0.28  rollout_tp=${rollout_tp}  ppo_max_tok=${ppo_max_token_len_per_gpu}"

########################### preflight: chat template + tokenizer ##########
# Fails before any GPU is touched if the parquet's message format does not
# render under the Gemma chat template (typical failure: `system` role).
if [ "${PREFLIGHT}" = "1" ]; then
python3 - "$MODEL_PATH" "$TRAIN_FILE" "$PROMPT_KEY" <<'EOF'
import sys, pyarrow.parquet as pq
from transformers import AutoTokenizer
model, path, key = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model)
row = pq.read_table(path).slice(0, 1).to_pylist()[0]
msgs = row[key]
if not isinstance(msgs, list):
    sys.exit(f"PREFLIGHT FAILED: {key!r} is not a message list (got {type(msgs).__name__}); "
             "this parquet stores pre-templated text and must be rebuilt for Gemma")
roles = [m.get("role") for m in msgs]
try:
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
except Exception as e:
    sys.exit(f"PREFLIGHT FAILED: chat template rejected roles={roles}: {e}")
n = len(tok(text, add_special_tokens=False)["input_ids"])
print(f"PREFLIGHT OK: roles={roles} rendered_tokens={n} vocab={len(tok)}")
print("----- rendered sample (first 600 chars) -----")
print(text[:600])
print("---------------------------------------------")
EOF
fi

########################### launch ########################################
# [GB200] block inherited from the Qwen rl-05 line (enforce_eager=False and
# free_cache_engine=False are measured [SYS] deviations from upstream; both
# re-verified on the gemma-4 decode-light line).
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
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING} \
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
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=${PREFIX_CACHING} \
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
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    ${RAY_NUM_GPUS_ARG} \
    "$@"

# =============================================================================
# Bring-up watch list (gemma-specific, check on the 2-step smoke):
#   1. PREFLIGHT line: roles must render; rendered_tokens ~150 like Qwen.
#   2. vLLM server log must show "Forcing TRITON_ATTN backend" (Gemma4
#      heterogeneous head dims); record as a SHARED constraint in the doc.
#   3. actor/perf/max_memory_reserved_gb after step 2: if < ~100 GB, try
#      PPO_MAX_TOKEN_LEN=32768 to restore the Qwen packing budget.
#   4. timing_s/agent_loop/num_preempted/max must stay -1/0; if > 0 the
#      KV cache is too small for 8192-cap sequences -> ROLLOUT_MEM=0.40.
#   5. response_length/clip_ratio and critic/rewards/mean: sanity-compare
#      the distribution against the Qwen line (~25-30% cap-hit) and against
#      the Gemma TPU run before reading any step time.
#   6. Head-node RAM scales with the number of vLLM replicas (W=64 hit
#      396/400 GiB on the Qwen line): raise the head pod memory limit
#      before a 64-GPU run.
# =============================================================================
