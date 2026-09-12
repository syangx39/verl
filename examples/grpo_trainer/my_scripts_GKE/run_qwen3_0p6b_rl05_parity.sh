#!/usr/bin/env bash
# =============================================================================
# Track A / Phase 0 -- PRE-CHECK: does the rl05 recipe learn on GB200 at all?
# =============================================================================
# Derived from run_qwen3_0p6b_rl05_parity.sh. The "frozen invariants" block is
# byte-identical to rl05 -- this run must train the SAME recipe, only longer and
# with eval + logging switched on. If this run learns, it IS Level-3 GB200 seed #1.
#
# Diff vs rl05 (all trainer/data/logging; nothing in the frozen block):
#   1. TOTAL_STEPS 20 -> 300 (env override)                          [PHASE0]
#   2. eval: test_freq -1 -> 50, val_before_train True (step-0 anchor),
#      greedy n=1 (val_kwargs made explicit), val generations dumped [PHASE0]
#   3. checkpoints: save_freq -1 -> 50, hf_model included so every
#      checkpoint is directly evalable by an offline vLLM harness     [PHASE0]
#   4. data.seed explicit (=SEED); EXPERIMENT_NAME carries the seed   [PHASE0]
#   5. rollout.calculate_log_probs=True -> TensorBoard gets
#      training/rollout_probs_diff_{mean,max,std} (trainer-vs-rollout
#      logp mismatch, the Miles Fig.4b panel)                          [PHASE0]
#   6. TENSORBOARD_DIR and EXPERIMENT_NAME pushed into the Ray runtime_env
#      so each run gets its own TB directory and its own reward-dump folder.
#      NOTE: verl's TensorboardLogger uses TENSORBOARD_DIR *as the log dir*
#      (no project/experiment suffix). The pod-level env in the RayCluster
#      yaml points every run at the SAME directory -- earlier runs' event
#      files are mixed together there. The per-run override below fixes it.
#   7. reward: maxtext_math_reward.py (Phase-0 version) returns
#      {score, acc, fmt}; score is unchanged, acc/fmt are logging only.
#   8. [v2] TensorBoard writes to NODE-LOCAL disk (TB_ROOT=/tmp/tb_local) and a
#      background loop mirrors it to gcsfuse every 5 min (+ once at exit).
#      Writing TB events straight onto gcsfuse cost ~40 s/step of blocking
#      (torch's async writer queue is 10 deep; gcsfuse appends are ~0.4 s each).
#   9. [v2] reward: math_verify signal-timeouts disabled (thread-safe); launcher
#      asserts from a worker thread that equivalence scoring works, and logs
#      the math-verify version for the rulebook.
#  10. [v3] PRECISION: master weights + Adam moments were bf16 (pilot checkpoint
#      showed exp_avg/exp_avg_sq bf16 and 0.4% of params moving per step at
#      lr 1e-6). Now explicit fp32 master + fp32 optimizer state, bf16 compute via
#      FSDP mixed precision (verl default). Strategy pinned to fsdp (FSDP1) --
#      that is what the resolved config showed; the perf doc's "FSDP2" was wrong.
#  11. [v3] Optimizer/schedule made explicit: betas (0.9, 0.999), wd 0.01,
#      clip_grad 1.0, lr_scheduler_type=constant, warmup ratio 0 (verl v0.8
#      key names; `warmup_style` is deprecated there). These must be mirrored on
#      the TPU side (MaxText rl.yml defaults differ) -- see rulebook. If your fork
#      rejects a key, check the resolved config it prints at startup.
#  12. [v3] Data: question-level split (train_qsplit / val_1k_qsplit, no
#      leakage) and GSM8K test are REQUIRED; the launcher aborts if missing.
#  13. [v3] reward v3: killable math_verify worker processes with timeout;
#      returns qid + mv_timeout/mv_exc/mv_lenrej flags (val-aux + rollout dump).
#
# Usage (inside the head pod, after `source /workspace/setup_env.sh`):
#   # smoke (5 steps, eval at 2 and 4, one checkpoint):
#   TOTAL_STEPS=5 TEST_FREQ=2 SAVE_FREQ=4 NNODES=4 bash $SCRIPTS_DIR/run_qwen3_0p6b_phase0_precheck.sh
#   # real run, one seed, under tmux:
#   VAL_FILE=$DATA_DIR/val_1k.parquet SEED=1 NNODES=16 bash $SCRIPTS_DIR/run_qwen3_0p6b_phase0_precheck.sh 2>&1 | tee $LOG_DIR/phase0_rfix_seed1.log
#   # TensorBoard while running (on the head pod): tensorboard --logdir /tmp/tb_local --bind_all
#   # plot:  python3 plot_phase0.py --tb $TB_DIR --rollout $ROLLOUT_DUMP_DIR --out ...
# =============================================================================

set -xeuo pipefail

########################### paths (site-specific) ###########################
DATA_DIR=${DATA_DIR:-$HOME/meta-RL/data/openmathinstruct2}
# [v3] question-level split (build_eval_data_v3.py). The old row-split train/val
# leaked 77% of val questions into train -- do not use it.
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train_qsplit.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val_1k_qsplit.parquet}       # in-distribution, held-out questions
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$DATA_DIR/gsm8k_test.parquet}   # held-out benchmark, primary
for f in "${TRAIN_FILE}" "${VAL_FILE}" "${GSM8K_TEST_FILE}"; do
  test -s "$f" || { echo "[phase0] ABORT: missing data file $f (run build_eval_data_v3.py)"; exit 2; }
done
VAL_FILES="['${VAL_FILE}','${GSM8K_TEST_FILE}']"

REWARD_FN_PATH=${REWARD_FN_PATH:-$HOME/meta-RL/reward/maxtext_math_reward.py}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}   # [MAXTEXT] model_name=qwen3-0.6b
                                            # [TODO] record exact HF revision in the rulebook

LOG_DIR=${LOG_DIR:-$HOME/meta-RL/logs}
CKPT_DIR=${CKPT_DIR:-$HOME/meta-RL/ckpt}
TB_ROOT=${TB_ROOT:-/tmp/tb_local}                     # [v2] node-local; NEVER a gcsfuse path
TB_MIRROR_ROOT=${TB_MIRROR_ROOT:-/workspace/meta-RL/.home/tensorboard_log}   # gcsfuse copy for laptop/TensorBoard
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${TB_ROOT}"

########################### scale knobs (only these vary between runs) ######
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}         # [GB200] A4X: 4 GPUs/node
TOTAL_STEPS=${TOTAL_STEPS:-300}             # [PHASE0] 20 -> 300
TEST_FREQ=${TEST_FREQ:-50}                  # [PHASE0] eval every N steps
SAVE_FREQ=${SAVE_FREQ:-50}                  # [PHASE0] checkpoint every N steps
SEED=${SEED:-1}                             # [PHASE0] data order seed (frozen in Level 0)
RUN_TAG=${RUN_TAG:-v3}                      # [v3] fp32-master + qsplit + gsm8k + reward v3; bump when the recipe changes

PROJECT_NAME=${PROJECT_NAME:-trackA_phase0}
W=$(( NNODES * NGPUS_PER_NODE ))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_phase0_${RUN_TAG}_seed${SEED}_${NNODES}n${W}g_$(date +%Y%m%d_%H%M)}
TB_DIR=${TB_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
TB_MIRROR=${TB_MIRROR_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
VAL_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/val_dump
ROLLOUT_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/rollout_dump   # per-step train samples (acc/fmt per sample)
mkdir -p "${TB_DIR}" "${TB_MIRROR}" "${VAL_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"

########################### [v2] pre-flight: reward must score equivalences from a THREAD ####
python3 - "${REWARD_FN_PATH}" <<'PYEOF'
import importlib.metadata as md, importlib.util, json, sys, threading, warnings
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("r", sys.argv[1]); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
res = {}
t = threading.Thread(target=lambda: res.__setitem__("s", r.compute_score("x", "<reasoning>x</reasoning><answer>1/2</answer>", json.dumps(["\\frac{1}{2}", "\\frac{1}{2}"]))["score"]))
t.start(); t.join()
print(f"[phase0] math-verify version = {md.version('math-verify')}; worker-thread equivalence score = {res.get('s')}")
if res.get("s") != 1.1:
  sys.exit("[phase0] ABORT: reward does not award symbolic equivalence from a worker thread -- wrong reward file?")
PYEOF

########################### [v2] TB mirror loop: local -> gcsfuse every 5 min, and once at exit ####
( while true; do sleep 300; cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true; done ) &
TB_SYNC_PID=$!
trap 'kill ${TB_SYNC_PID} 2>/dev/null; cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true' EXIT

########################### frozen invariants -- do not tune ################
# BYTE-IDENTICAL to run_qwen3_0p6b_rl05_parity.sh. Any change here = new recipe.
# [v3] Precision/optimizer are ALSO frozen semantics now (see launch args):
#   master weights fp32, Adam moments fp32, compute bf16 (FSDP mixed precision),
#   AdamW betas (0.9,0.999) eps 1e-8 wd 0.01, clip_grad 1.0, constant LR, no warmup.
train_batch_size=256          # [RL05] batch_size=256
ppo_mini_batch_size=256       # [RL05] mu=1: one optimizer update per rollout
max_prompt_length=8192        # [MAXTEXT] max_prefill_predict_length=8192
max_response_length=8192      # [MAXTEXT] max_target_length(16384) - prefill(8192)
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
rollout_tp=${ROLLOUT_TP:-1}   # [SYS] free variable; not a timing run
rollout_gpu_mem_util=0.30     # [SYS]
ppo_max_token_len_per_gpu=32768  # [SYS] dynamic-bsz packing budget

########################### launch ########################################
echo "[accounting] W=${W}  batch=256x8=2048  updates/rollout=1 (mini=256)  beta=0  clip=0.2/0.28  rollout_tp=${rollout_tp}  seed=${SEED}"
echo "[phase0] steps=${TOTAL_STEPS} test_freq=${TEST_FREQ} save_freq=${SAVE_FREQ}"
echo "[phase0] tensorboard -> ${TB_DIR}"
echo "[phase0] tb mirror   -> ${TB_MIRROR}"
echo "[phase0] checkpoints -> ${CKPT_DIR}/${EXPERIMENT_NAME}"
echo "[phase0] val dumps   -> ${VAL_DUMP_DIR}"
echo "[phase0] train dumps -> ${ROLLOUT_DUMP_DIR}"
echo "[phase0] reward dump -> ${REWARD_DUMP_DIR:-<unset>}/${EXPERIMENT_NAME}"

RAY_NUM_GPUS_ARG=""
if [ -z "${RAY_ADDRESS:-}" ]; then
  RAY_NUM_GPUS_ARG="+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=${SEED} \
    data.dataloader_num_workers=8 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]' \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
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
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="${CKPT_DIR}/${EXPERIMENT_NAME}" \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=True \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}" \
    trainer.resume_mode=disable \
    trainer.total_epochs=100 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    +ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TB_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    ${RAY_NUM_GPUS_ARG} \
    "$@"

# =============================================================================
# Notes:
#   * trainer.total_epochs=100 is a ceiling; verl stops at total_training_steps.
#   * trainer.rollout_data_dir writes <step>.jsonl (input/output/gts/score/acc/fmt for
#     all 2048 completions) in a background thread, ~35 MB/step on gcsfuse -> ~11 GB
#     for 300 steps. It is the source for the per-step train acc / solve_all /
#     solve_none panels (plot_phase0.py --rollout). Drop the flag for timing runs.
#   * After the smoke run, VERIFY the precision change took effect:
#       optim shard: python3 -c "import zipfile,glob; p=glob.glob('<ckpt>/actor/optim_*rank_0.pt')[0];
#         z=zipfile.ZipFile(p); d=z.read([n for n in z.namelist() if n.endswith('data.pkl')][0]);
#         print('bf16', d.count(b'BFloat16Storage'), 'fp32', d.count(b'FloatStorage'))"  -> fp32 >> bf16
#       and 'strategy': 'fsdp', betas/weight_decay/warmup in the resolved config printed at start.
#   * Smoke-test checklist before the real run:
#       - TB tags present: critic/score/mean, val-core/*/reward/mean@1,
#         val-aux/*/acc/mean@1, training/rollout_probs_diff_mean, actor/pg_clipfrac
#       - ${ROLLOUT_DUMP_DIR}/1.jsonl exists with 2048 lines and acc/fmt keys
#       - one checkpoint written, actor/huggingface/ present inside it, write time noted
#       - val-aux/<ds>/mv_timeout|mv_exc|mv_lenrej/mean@1 ~ 0 (reward workers healthy)
# =============================================================================