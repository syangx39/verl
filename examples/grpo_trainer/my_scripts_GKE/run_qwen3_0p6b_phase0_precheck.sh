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
#  15. [v5] GPU STABILITY ROUND. The v4 pilot (fp32, rl05 recipe) collapsed at
#      ~step 100 (entropy blow-up). Recipe knobs are now env-overridable with the
#      rl05 values as DEFAULTS, so one launcher serves the ablation ladder:
#        KL_COEF (0 = rl05, no ref model) KL_TYPE (low_var_kl)
#        ROLLOUT_TEMPERATURE / ROLLOUT_TOP_P / ROLLOUT_TOP_K (0.8/0.95/50 = rl05)
#        PPO_MINI_BATCH (256 = rl05, one update per rollout; 64 -> mu=4)
#        REWARD_FMT_WEIGHT (0.1 = parity; 0 = answer-only)
#        REWARD_OVERLONG_BUFFER / REWARD_OVERLONG_PENALTY (0/1.0 = off = parity)
#        FILTER_OVERLONG_PROMPTS (False = rl05), TEST_FREQ (now 10)
#      Every deviation from the defaults is a recipe change -> set RUN_TAG.
#      A collapse guard (collapse_guard.py) runs alongside and kills the driver
#      on the v4 signature (entropy x3, cap-hit > 0.9, score < 0.5x, grad spikes).
#  16. [v5.1] Parity-fixture support: DATA_SHUFFLE=False reads the train file in order (used with
#      train_order_seed<k>.parquet); LOGPROB_FIXTURE_DIR=<dir> (+ LOGPROB_FIXTURE_STEP, default 1)
#      makes the patched trainer (patch_verl_logprob_fixture.py) dump the pre-update batch tensors,
#      sampler/trainer log-probs and a repeated trainer pass at that step. Unset = no effect.
#  14. [v4] reward v4: no silent fallback (import raises if workers cannot start),
#      structured worker status (exceptions counted as mv_exc), REWARD_MV_* forwarded
#      to the Ray actors via runtime_env. Rollout dump carries the trainer's uid
#      (apply patch_verl_dump_uid.py to the fork ONCE). Data from build v4
#      (normalized question identity, unique extra_info.index).
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
TEST_FREQ=${TEST_FREQ:-10}                  # [v5] eval every 10 steps (was 50) to see inflection points
SAVE_FREQ=${SAVE_FREQ:-50}                  # [PHASE0] checkpoint every N steps
SEED=${SEED:-1}                             # [PHASE0] data order seed (frozen in Level 0)
RUN_TAG=${RUN_TAG:-${PRESET:-v5}}           # [v5] set per ablation; defaults to the preset name
# [v5] PRESET=stab sets the whole agreed stability-round recipe in one place
# (individual env vars still override). Without PRESET every knob defaults to rl05.
if [ "${PRESET:-}" = "stab" ]; then
  : "${ROLLOUT_TEMPERATURE:=1.0}" "${ROLLOUT_TOP_P:=1.0}" "${ROLLOUT_TOP_K:=-1}"
  : "${REWARD_FMT_WEIGHT:=0}" "${REWARD_OVERLONG_BUFFER:=1024}" "${REWARD_OVERLONG_PENALTY:=1.0}"
  : "${FILTER_OVERLONG_PROMPTS:=True}" "${KL_COEF:=0.001}" "${KL_TYPE:=low_var_kl}" "${TEST_FREQ:=10}"
  echo "[phase0] PRESET=stab: T=1 top_p=1 top_k=-1 fmt_w=0 overlong=1024/1.0 filter_overlong_prompts=True KL=0.001(low_var_kl) test_freq=10"
fi
# [v4] reward worker knobs. They must reach the Ray actors that run the reward,
# so they are forwarded through ray runtime_env below (shell exports alone do NOT
# reach them). The pre-flight below prints the values it sees.
# QUOTING: the values are passed as  "+key='${VAR}'"  -- the shell strips the outer
# double quotes and Hydra sees +key='1', i.e. a STRING. Unquoted, Hydra parses 1/4/5/400
# as ints and ray.init rejects runtime_env.env_vars with non-string values.
export REWARD_MV_POOL=${REWARD_MV_POOL:-1}
export REWARD_MV_PROCS=${REWARD_MV_PROCS:-4}
export REWARD_MV_TIMEOUT=${REWARD_MV_TIMEOUT:-5}
export REWARD_MATH_VERIFY_MAX_CHARS=${REWARD_MATH_VERIFY_MAX_CHARS:-400}
export REWARD_FMT_WEIGHT=${REWARD_FMT_WEIGHT:-0.1}            # [v5] 0.1 = parity, 0 = answer-only
export REWARD_OVERLONG_BUFFER=${REWARD_OVERLONG_BUFFER:-0}    # [v5] 0 = off (parity); e.g. 1024
export REWARD_OVERLONG_PENALTY=${REWARD_OVERLONG_PENALTY:-1.0}
export REWARD_MAX_RESP_LEN=8192                               # must equal max_response_length

PROJECT_NAME=${PROJECT_NAME:-trackA_phase0}
W=$(( NNODES * NGPUS_PER_NODE ))
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_phase0_${RUN_TAG}_seed${SEED}_${NNODES}n${W}g_$(date +%Y%m%d_%H%M)}
TB_DIR=${TB_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
TB_MIRROR=${TB_MIRROR_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
VAL_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/val_dump
ROLLOUT_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/rollout_dump   # per-step train samples (acc/fmt per sample)
mkdir -p "${TB_DIR}" "${TB_MIRROR}" "${VAL_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"

########################### [v4] pre-flight -1: cluster must be idle (no leftover vLLM/verl actors) ####
if [ "${SKIP_IDLE_CHECK:-0}" != "1" ] && command -v ray >/dev/null 2>&1; then   # SKIP_IDLE_CHECK=1 when launching a 2nd run in parallel
  GPU_USE=$(ray status 2>/dev/null | awk '/GPU/ && /\// {print $1; exit}')     # e.g. 0.0/64.0
  if [ -n "${GPU_USE}" ] && [ "${GPU_USE%%/*}" != "0.0" ]; then
    echo "[phase0] ABORT: Ray reports GPUs in use (${GPU_USE}) -- leftover actors from a previous run. Clean them first."
    exit 2
  fi
  echo "[phase0] ray GPU usage before launch: ${GPU_USE:-unknown}"
fi

########################### [v5] pre-flight 0b: length-aware reward needs the response_len patch ####
if [ "${REWARD_OVERLONG_BUFFER}" != "0" ]; then
  RM=$(python3 -c "import verl.experimental.reward_loop.reward_manager.naive as m; print(m.__file__)" 2>/dev/null | tail -1)
  grep -q "_RESP_LEN" "${RM}" || { echo "[phase0] ABORT: ${RM} lacks the response_len patch. Run: python3 ${SCRIPTS_DIR:-.}/patch_verl_reward_response_len.py ${RM}"; exit 2; }
  echo "[phase0] reward manager: ${RM} (response_len patch present)"
fi

########################### [v4] pre-flight 0: the verl fork must carry the rollout-dump uid patch ####
RT=$(python3 -c "import verl.trainer.ppo.ray_trainer as m; print(m.__file__)" 2>/dev/null | tail -1)   # import prints banner lines; keep the path only
grep -q "_DUMP_UID" "${RT}" || { echo "[phase0] ABORT: ${RT} lacks the uid dump patch. Run: python3 ${SCRIPTS_DIR:-.}/patch_verl_dump_uid.py ${RT}"; exit 2; }
echo "[phase0] verl fork: ${RT} (uid dump patch present); git head $(git -C "$(dirname "${RT}")" rev-parse --short HEAD 2>/dev/null || echo n/a)"

########################### [v5] pre-flight 1: reward workers up + knob-aware scoring from a THREAD ####
python3 - "${REWARD_FN_PATH}" <<'PYEOF'
import importlib.metadata as md, importlib.util, json, os, sys, threading, warnings
warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("r", sys.argv[1]); r = importlib.util.module_from_spec(spec)
try:
  spec.loader.exec_module(r)              # raises MathVerifyPoolError if workers cannot start
except Exception as e:
  sys.exit(f"[phase0] ABORT: reward import failed: {type(e).__name__}: {e}")
fmt_w = float(os.environ.get("REWARD_FMT_WEIGHT", "0.1")); buf = int(os.environ.get("REWARD_OVERLONG_BUFFER", "0"))
pen = float(os.environ.get("REWARD_OVERLONG_PENALTY", "1.0")); mx = int(os.environ.get("REWARD_MAX_RESP_LEN", "8192"))
gt = json.dumps(["\\frac{1}{2}", "\\frac{1}{2}"]); comp = "<reasoning>x</reasoning><answer>1/2</answer>"
res = {}
t = threading.Thread(target=lambda: res.__setitem__("s", r.compute_score("x", comp, gt, extra_info={"index": 0, "response_len": 100})))
t.start(); t.join()
o = res.get("s", {}); st = r.mv_stats()
print(f"[phase0] math-verify version = {md.version('math-verify')}; reward workers = {st['idle']}/{st['cfg_procs']} idle, pool={st['pool']}, "
      f"timeout={st['cfg_timeout_s']}s; knobs fmt_w={fmt_w} overlong={buf}/{pen}; short correct answer -> {o}")
checks = [("acc == 1", o.get("acc") == 1.0), ("fmt == 1", o.get("fmt") == 1.0), ("length_penalty == 0", o.get("length_penalty") == 0.0),
          (f"score == 1 + fmt_w ({1.0 + fmt_w})", o.get("score") is not None and abs(o["score"] - (1.0 + fmt_w)) < 1e-9)]
if buf > 0:   # penalty direction: at the cap the same correct answer must lose exactly `pen`
  o2 = r.compute_score("x", comp, gt, extra_info={"index": 0, "response_len": mx})
  checks.append((f"length_penalty at cap == -{pen}", abs(o2.get("length_penalty", 0.0) + pen) < 1e-9))
  checks.append((f"score at cap == 1 + fmt_w - pen", abs(o2["score"] - (1.0 + fmt_w - pen)) < 1e-9))
bad = [name for name, ok in checks if not ok]
if bad:
  sys.exit(f"[phase0] ABORT: reward pre-flight failed: {bad}")
if os.environ.get("REWARD_MV_POOL", "1") == "1" and (not st["pool"] or st["idle"] != st["cfg_procs"]):
  sys.exit("[phase0] ABORT: math_verify worker pool not healthy")
print("[phase0] reward pre-flight OK:", ", ".join(n for n, _ in checks))
PYEOF

########################### [v2] TB mirror loop: local -> gcsfuse every 5 min, and once at exit ####
( while true; do sleep 300; cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true; done ) &
TB_SYNC_PID=$!
# [v5] collapse guard is started AFTER the driver (it needs the driver PID) -- see the launch section.
GUARD_PID=""; DRIVER_PID=""
# [v5] Signal handling: the driver runs in the background (so the guard can target its PID), so
# INT/TERM sent to this launcher are FORWARDED to that one driver; the EXIT trap also stops it if
# still alive (e.g. `kill <launcher pid>`), then mirrors TB and prints guard verdicts.
on_signal() {
  echo "[phase0] caught signal -- stopping driver ${DRIVER_PID:-<none>}"
  [ -n "${DRIVER_PID}" ] && kill -TERM "${DRIVER_PID}" 2>/dev/null || true
}
# After forwarding, EXIT immediately so on_exit (time-bounded TERM->KILL) always runs,
# even if the driver ignores TERM and `wait` would otherwise block forever.
trap 'on_signal; exit 130' INT
trap 'on_signal; exit 143' TERM
on_exit() {
  if [ -n "${DRIVER_PID}" ] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
    echo "[phase0] exit: driver ${DRIVER_PID} still alive -- sending TERM"; kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "$(( ${DRIVER_KILL_GRACE:-30} / 2 ))"); do kill -0 "${DRIVER_PID}" 2>/dev/null || break; sleep 2; done
    kill -0 "${DRIVER_PID}" 2>/dev/null && { echo "[phase0] exit: driver did not stop -- KILL"; kill -KILL "${DRIVER_PID}" 2>/dev/null || true; }
  fi
  kill ${TB_SYNC_PID} ${GUARD_PID} 2>/dev/null || true
  cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true
  test -f "${TB_DIR}/COLLAPSE_ABORT.txt" && { echo "[phase0] RUN ABORTED BY COLLAPSE GUARD:"; cat "${TB_DIR}/COLLAPSE_ABORT.txt"; }
  test -f "${TB_DIR}/COLLAPSE_WARN.txt" && { echo "[phase0] guard warnings:"; cat "${TB_DIR}/COLLAPSE_WARN.txt"; }
  return 0
}
trap on_exit EXIT

########################### frozen invariants -- do not tune ################
# BYTE-IDENTICAL to run_qwen3_0p6b_rl05_parity.sh. Any change here = new recipe.
# [v3] Precision/optimizer are ALSO frozen semantics now (see launch args):
#   master weights fp32, Adam moments fp32, compute bf16 (FSDP mixed precision),
#   AdamW betas (0.9,0.999) eps 1e-8 wd 0.01, clip_grad 1.0, constant LR, no warmup.
train_batch_size=256                          # [RL05] batch_size=256
ppo_mini_batch_size=${PPO_MINI_BATCH:-256}    # [RL05] 256 = one update per rollout (mu=1); 64 -> mu=4
max_prompt_length=8192                        # [MAXTEXT] max_prefill_predict_length=8192
max_response_length=8192                      # [MAXTEXT] max_target_length(16384) - prefill(8192)
rollout_n=8                                   # [MAXTEXT] rl.num_generations=8
kl_loss_coef=${KL_COEF:-0.0}                  # [RL05] 0.0 = no KL, no ref model; >0 -> use_kl_loss + ref
kl_loss_type=${KL_TYPE:-low_var_kl}           # verl estimator when KL_COEF>0 (must match the TPU side later)
clip_ratio_low=0.2                            # [RL05] rl.grpo_epsilon=0.2
clip_ratio_high=0.28                          # [RL05] rl.epsilon_high=0.28 (DAPO clip-higher)
temperature=${ROLLOUT_TEMPERATURE:-0.8}       # [MAXTEXT] 0.8 ; stability round: 1.0
top_p=${ROLLOUT_TOP_P:-0.95}                  # [MAXTEXT] 0.95; stability round: 1.0
top_k=${ROLLOUT_TOP_K:-50}                    # [MAXTEXT] 50  ; stability round: -1
max_num_batched_tokens=32768                  # [MAXTEXT] max_num_batched_tokens=32768
actor_lr=${ACTOR_LR:-1e-6}                    # [MAXTEXT] learning_rate=1e-6
filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-False}
if awk "BEGIN{exit !(${kl_loss_coef} > 0)}"; then use_kl_loss=True; else use_kl_loss=False; fi

########################### system adaptations ############################
rollout_tp=${ROLLOUT_TP:-1}   # [SYS] free variable; not a timing run
rollout_gpu_mem_util=0.30     # [SYS]
ppo_max_token_len_per_gpu=32768  # [SYS] dynamic-bsz packing budget

########################### launch ########################################
echo "[accounting] W=${W}  batch=${train_batch_size}x${rollout_n}  updates/rollout=$((train_batch_size/ppo_mini_batch_size)) (mini=${ppo_mini_batch_size})  rollout_tp=${rollout_tp}  seed=${SEED}"
########################### [v5] recipe fingerprint (everything that is frozen semantics) ####
echo "[recipe] lr=${actor_lr} kl_coef=${kl_loss_coef}(${kl_loss_type},use_kl=${use_kl_loss}) mini=${ppo_mini_batch_size} (mu=$((train_batch_size/ppo_mini_batch_size))) clip=${clip_ratio_low}/${clip_ratio_high} T=${temperature} top_p=${top_p} top_k=${top_k} cap=${max_response_length} n=${rollout_n} fmt_w=${REWARD_FMT_WEIGHT} overlong=${REWARD_OVERLONG_BUFFER}/${REWARD_OVERLONG_PENALTY} filter_overlong_prompts=${filter_overlong_prompts} fp32-master"

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
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation='error' \
    data.shuffle=${DATA_SHUFFLE:-True} \
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
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_OFFLOAD:-False} \
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
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR='${TB_DIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME='${EXPERIMENT_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_POOL='${REWARD_MV_POOL}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_PROCS='${REWARD_MV_PROCS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MV_TIMEOUT='${REWARD_MV_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MATH_VERIFY_MAX_CHARS='${REWARD_MATH_VERIFY_MAX_CHARS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_FMT_WEIGHT='${REWARD_FMT_WEIGHT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_BUFFER='${REWARD_OVERLONG_BUFFER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_PENALTY='${REWARD_OVERLONG_PENALTY}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MAX_RESP_LEN='${REWARD_MAX_RESP_LEN}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_DIR='${LOGPROB_FIXTURE_DIR:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_STEP='${LOGPROB_FIXTURE_STEP:-1}'" \
    ${RAY_NUM_GPUS_ARG} \
    "$@" &
DRIVER_PID=$!
echo "[phase0] driver pid ${DRIVER_PID}"
# [v5] collapse guard: warns on single-signal anomalies, stops ONLY this driver on
# acc-decline AND (entropy OR cap-hit) -- see collapse_guard.py. COLLAPSE_GUARD=0 disables.
GUARD_LOG=${LOG_DIR}/${EXPERIMENT_NAME}/collapse_guard.log; mkdir -p "$(dirname "${GUARD_LOG}")"
if [ "${COLLAPSE_GUARD:-1}" = "1" ]; then
  python3 "${SCRIPTS_DIR:-$(dirname "$0")}/collapse_guard.py" --tb "${TB_DIR}" --rollout "${ROLLOUT_DUMP_DIR}" --pid "${DRIVER_PID}" --poll 60 > "${GUARD_LOG}" 2>&1 &
  GUARD_PID=$!
  echo "[phase0] collapse guard pid ${GUARD_PID} (watching driver ${DRIVER_PID}) -> ${GUARD_LOG}"
fi
# wait for the driver; a signal interrupts `wait` (rc>128) -- after forwarding it, wait again
set +e
wait "${DRIVER_PID}"; DRIVER_RC=$?
while kill -0 "${DRIVER_PID}" 2>/dev/null; do wait "${DRIVER_PID}"; DRIVER_RC=$?; done
set -e
echo "[phase0] driver exited with rc=${DRIVER_RC}"
exit ${DRIVER_RC}

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
#       - ${ROLLOUT_DUMP_DIR}/1.jsonl exists with 2048 lines, acc/fmt/qid/uid keys, 256 distinct uid
#       - one checkpoint written, actor/huggingface/ present inside it, write time noted
#       - val-aux/<ds>/mv_timeout|mv_exc|mv_lenrej/mean@1 ~ 0 (reward workers healthy)
#       - [v5] with REWARD_OVERLONG_BUFFER>0: val-aux/<ds>/length_penalty/mean@1 <= 0 and, per dump row,
#         score == acc + fmt_w*fmt + length_penalty; long responses (>7168 tok) carry a negative penalty
#       - [v5] with FILTER_OVERLONG_PROMPTS=True: grep "dataset len" in the log for train/val sizes,
#         and val_dump/0.jsonl must still have 2319 rows (1000 omi2 + 1319 gsm8k)
#       - [v5] collapse guard: <exp>/collapse_guard.log shows a heartbeat line per poll
# =============================================================================
