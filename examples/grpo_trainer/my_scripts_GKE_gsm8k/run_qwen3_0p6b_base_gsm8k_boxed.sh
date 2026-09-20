#!/usr/bin/env bash
# Meta GSM8K-boxed reproduction on verl -- Qwen3-0.6B-Base, GRPO, clip-higher, 8x GB200.
#
# Source of truth: Meta's configs/qwen3-0p6b-base-boxed-cliphigh.yaml (reference run
# qwen3-0p6b-base-boxed-cliphigh-0c2e52db). Every value in the FROZEN block below is copied from it;
# the mapping to verl keys is in README.md. Items Meta did not specify (top_p/top_k, weight decay,
# loss aggregation, chat-template thinking flag) are exposed as env knobs with our best-guess defaults
# and MUST be confirmed with Meta before the reference band is frozen.
#
# Infra blocks (node-local TensorBoard + mirror, idle check, fork-patch checks, signal handling,
# collapse guard, recipe fingerprint) are the same as run_qwen3_0p6b_rl05_parity.sh v5.
set -euo pipefail
set -x

########################### environment ##############################################
: "${DATA_DIR:?set DATA_DIR to the dir produced by build_gsm8k_boxed_data.py --out}"
: "${MODEL_PATH:?set MODEL_PATH to the stop-set-patched Qwen3-0.6B-Base copy (build_gsm8k_boxed_data.py --model_out)}"
# Isolation from the Track A environment: setup_env.sh exports SCRIPTS_DIR / REWARD_FN_PATH for the OLD
# recipe and shells may carry REWARD_OVERLONG_BUFFER=1024 etc. This launcher ignores those and takes
# its own directory and its own reward; overrides use META_* names only.
SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)
REWARD_FN_PATH=${META_REWARD_FN:-$SCRIPTS_DIR/boxed_math_reward.py}
for f in collapse_guard.py boxed_math_reward.py; do test -s "$SCRIPTS_DIR/$f" || { echo "[meta] ABORT: $SCRIPTS_DIR/$f missing"; exit 2; }; done
LOG_DIR=${LOG_DIR:-/workspace/meta-RL/logs}
CKPT_DIR=${CKPT_DIR:-/workspace/meta-RL/ckpt}
TB_ROOT=${TB_ROOT:-/tmp/tb_local}                                     # node-local; NEVER a gcsfuse path
TB_MIRROR_ROOT=${TB_MIRROR_ROOT:-/workspace/meta-RL/.home/tensorboard_log}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${TB_ROOT}"

########################### FROZEN -- Meta's config, do not tune ######################
train_batch_size=128                 # global_batch_size: 128 prompts per optimizer step
rollout_n=16                         # num_generations: 16  -> 2048 sequences/step
ppo_mini_batch_size=128              # ppo_epochs=1, one update per rollout (mu=1)
micro_bsz_per_gpu=${MICRO_BSZ:-8}    # micro_batch_size: 8 per GPU (fixed; dynamic bsz OFF to match)
max_prompt_length=512                # max_seq_length 2560 = 512 prompt + 2048 completion
max_response_length=2048
actor_lr=${META_ACTOR_LR:-2.0e-5}    # learning_rate (Meta: 2.0e-5). META_ACTOR_LR is for the LR-sensitivity control only;
                                     # the pre-flight, the [recipe] line and the resolved config all read this same variable
lr_scheduler=cosine                  # lr_scheduler_type: cosine, decays to 0 at max_steps
lr_warmup_steps=10                   # warmup_steps: 10
TOTAL_STEPS=${TOTAL_STEPS:-250}      # max_steps: 250
clip_ratio_low=0.2
clip_ratio_high=0.28                 # inert: Meta sets old=new.detach() so the PPO ratio is identically 1 (REPRODUCTION.md §3)
clip_ratio_c=5.0                     # clip_ratio_dual 5.0 (Meta default; also inert at ratio 1, but recorded)
kl_loss_coef=0.0                     # kl_coeff 0.0: no KL, no reference model
# Token-level importance-sampling correction (REPRODUCTION.md §4): w = min(exp(clamp(logp_actor - logp_rollout, ±20)), 3.0),
# multiplies the (inert) clipped policy loss, no batch renormalization, no lower truncation, no rejection.
# verl mapping: rollout correction in TIS mode with threshold 3.0 and NO bypass -- old_log_probs stay the trainer's
# own recompute (ratio == 1 exactly, clip inert) and the TIS weight is exp(old_trainer - rollout) truncated at 3.0,
# detached. Gradient = -adv * w * grad(log pi), the same as Meta's IF Meta's w is detached (asked, PENDING).
# verl's rollout-correction backend also clamps the log-ratio to +-20 before exponentiating (same overflow guard as Meta).
IS_THRESHOLD=${IS_THRESHOLD:-3.0}
IS_ARGS=( "algorithm.rollout_correction.rollout_is=token"                 # per-token IS (TIS), verl v0.8 rollout_correction.yaml
          "algorithm.rollout_correction.rollout_is_threshold=${IS_THRESHOLD}"   # upper truncation only
          "algorithm.rollout_correction.rollout_rs=null"                    # no rejection sampling
          "algorithm.rollout_correction.rollout_is_batch_normalize=False"   # raw weights
          "algorithm.rollout_correction.bypass_mode=False" )                # decoupled: rollout / old(trainer) / current
# the resolved-config pre-flight below aborts if the fork spells these keys differently
export REWARD_PENALTY_SOURCES=${META_PENALTY_SOURCES:-gsm8k_boxed_train}   # overlong penalty is TRAINING-only (Meta evaluates the raw reward)
temperature=1.0                      # generator.temperature
grad_clip=1.0                        # max_grad_norm
TEST_FREQ=${TEST_FREQ:-20}           # eval_steps: 20
SAVE_FREQ=${SAVE_FREQ:-50}           # save_steps: 50
# reward: boxed_math weight 1.0, format_score 0.1; overlong_buffer 512 / penalty 1.0 on cap 2048
export REWARD_FORMAT_SCORE=${META_FORMAT_SCORE:-0.1}        # always set here -> inherited REWARD_* values cannot leak in
export REWARD_OVERLONG_BUFFER=${META_OVERLONG_BUFFER:-512}
export REWARD_OVERLONG_PENALTY=${META_OVERLONG_PENALTY:-1.0}
export REWARD_MAX_RESP_LEN=2048
unset REWARD_FMT_WEIGHT REWARD_MATH_VERIFY_MAX_CHARS REWARD_MV_PROCS REWARD_MV_TIMEOUT 2>/dev/null || true
# stop set: Meta vllm_stop_token_ids=[151645] + tokenizer eos 151643 -> both are in MODEL_PATH/generation_config.json
########################### NOT specified by Meta -- confirm before freezing ###########
top_p=${ROLLOUT_TOP_P:-1.0}          # GUESS: vLLM default (Base generation_config has none)
top_k=${ROLLOUT_TOP_K:--1}           # GUESS
weight_decay=${WEIGHT_DECAY:-0.0}    # GUESS: HF TrainingArguments default is 0.0 (verl default would be 0.01)
loss_agg_mode=${LOSS_AGG_MODE:-token-mean}   # GUESS: Meta's trainer aggregation unknown
SEED=${SEED:-1}
########################### run identity ##############################################
NNODES=${NNODES:-2}; GPUS_PER_NODE=${GPUS_PER_NODE:-4}; W=$((NNODES*GPUS_PER_NODE))   # 8x GB200 = 2 GKE nodes x 4
RUN_TAG=${RUN_TAG:-meta_boxed}
PROJECT_NAME=meta_gsm8k_boxed
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_base_${RUN_TAG}_seed${SEED}_${NNODES}n${W}g_$(date +%Y%m%d_%H%M)}
TB_DIR=${TB_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
TB_MIRROR=${TB_MIRROR_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
VAL_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/val_dump
ROLLOUT_DUMP_DIR=${LOG_DIR}/${EXPERIMENT_NAME}/rollout_dump
mkdir -p "${TB_DIR}" "${TB_MIRROR}" "${VAL_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/gsm8k_boxed_train.parquet}
VAL512=${VAL512:-$DATA_DIR/gsm8k_boxed_test512.parquet}       # Meta's eval set (first 512)
VALFULL=${VALFULL:-$DATA_DIR/gsm8k_boxed_test.parquet}        # our full-set diagnostic
if [ "${EVAL_FULL:-1}" = "1" ]; then VAL_FILES="['${VAL512}','${VALFULL}']"; else VAL_FILES="['${VAL512}']"; fi   # EVAL_FULL=0 for Meta-comparable timing (512-question eval only)
for f in "${TRAIN_FILE}" "${VAL512}" "${VALFULL}" "${MODEL_PATH}/generation_config.json"; do
  test -s "$f" || { echo "[meta] ABORT: missing $f"; exit 2; }
done
python3 -c "import json,sys; e=json.load(open('${MODEL_PATH}/generation_config.json'))['eos_token_id']; sys.exit(0 if sorted(e)==[151643,151645] else 1)" \
  || { echo "[meta] ABORT: ${MODEL_PATH}/generation_config.json eos_token_id must be [151645,151643] (run build_gsm8k_boxed_data.py)"; exit 2; }
export REWARD_MV_POOL=0    # this reward has no math_verify; the pool knobs are irrelevant

########################### pre-flights ################################################
if [ "${SKIP_IDLE_CHECK:-0}" != "1" ] && command -v ray >/dev/null 2>&1; then
  GPU_USE=$(ray status 2>/dev/null | awk '/GPU/ && /\// {print $1; exit}')
  [ -n "${GPU_USE}" ] && [ "${GPU_USE%%/*}" != "0.0" ] && { echo "[meta] ABORT: GPUs in use (${GPU_USE}); clean leftovers first"; exit 2; }
  echo "[meta] ray GPU usage before launch: ${GPU_USE:-unknown}"
fi
RT=$(python3 -c "import verl.trainer.ppo.ray_trainer as m; print(m.__file__)" 2>/dev/null | tail -1)
grep -q "_DUMP_UID" "${RT}" || { echo "[meta] ABORT: ${RT} lacks the uid dump patch (patch_verl_dump_uid.py)"; exit 2; }
if [ "${REWARD_OVERLONG_BUFFER}" != "0" ]; then
  RM=$(python3 -c "import verl.experimental.reward_loop.reward_manager.naive as m; print(m.__file__)" 2>/dev/null | tail -1)
  grep -q "_RESP_LEN" "${RM}" || { echo "[meta] ABORT: ${RM} lacks the response_len patch (patch_verl_reward_response_len.py)"; exit 2; }
fi
python3 "${REWARD_FN_PATH}" > "${LOG_DIR}/reward_fixtures_check.log" 2>&1 \
  || { echo "[meta] ABORT: reward failed Meta's fixtures:"; grep -E "FAIL|RESULT" "${LOG_DIR}/reward_fixtures_check.log"; exit 2; }
grep -q "buffer': 512" "${LOG_DIR}/reward_fixtures_check.log" || { echo "[meta] ABORT: overlong buffer is not 512 (env leak?)"; grep knobs "${LOG_DIR}/reward_fixtures_check.log"; exit 2; }
echo "[meta] reward pre-flight: $(grep RESULT "${LOG_DIR}/reward_fixtures_check.log") (Meta's 15 reward + 6 overlong fixtures, penalty train-only)"

########################### full argument list (used by BOTH the --cfg job pre-flight and the launch) ####
TRAIN_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=${DATA_SHUFFLE:-True}
    data.seed=${SEED}
    data.dataloader_num_workers=2
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_scheduler_type=${lr_scheduler}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps}
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0
    actor_rollout_ref.actor.optim.num_cycles=0.5
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay}
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c}
    "${IS_ARGS[@]}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]'
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.n=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prefix_caching=True
    custom_reward_function.path="${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.default_local_dir="${CKPT_DIR}/${EXPERIMENT_NAME}"
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=True
    trainer.log_val_generations=10
    trainer.validation_data_dir="${VAL_DUMP_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}"
    trainer.resume_mode=disable
    trainer.total_epochs=100
    trainer.total_training_steps=${TOTAL_STEPS}
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR='${TB_DIR}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.EXPERIMENT_NAME='${EXPERIMENT_NAME}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_FORMAT_SCORE='${REWARD_FORMAT_SCORE}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_BUFFER='${REWARD_OVERLONG_BUFFER}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_OVERLONG_PENALTY='${REWARD_OVERLONG_PENALTY}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MAX_RESP_LEN='${REWARD_MAX_RESP_LEN}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.REWARD_PENALTY_SOURCES='${REWARD_PENALTY_SOURCES}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_DIR='${LOGPROB_FIXTURE_DIR:-}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.LOGPROB_FIXTURE_STEP='${LOGPROB_FIXTURE_STEP:-1}'"
)

########################### resolved-config pre-flight (hydra --cfg job; no Ray, no GPU) ############
CFG_LOG=${LOG_DIR}/${EXPERIMENT_NAME}/resolved_config_preflight.yaml; mkdir -p "$(dirname "${CFG_LOG}")"
CFG_ERR=${CFG_LOG%.yaml}.stderr.log
python3 -m verl.trainer.main_ppo --cfg job "${TRAIN_ARGS[@]}" "$@" > "${CFG_LOG}" 2> "${CFG_ERR}" \
  || { echo "[meta] ABORT: hydra rejected the config (unknown key?)."; echo "--- stdout (${CFG_LOG}) tail:"; tail -15 "${CFG_LOG}"; echo "--- stderr (${CFG_ERR}) tail:"; tail -30 "${CFG_ERR}";
       echo "check the rollout-correction key names: grep -rn rollout_correction \$(python3 -c 'import verl,os;print(os.path.dirname(verl.__file__))')/trainer/config"; exit 2; }
python3 - "${CFG_LOG}" "${CFG_ERR}" <<PYEOF
import sys, yaml
from omegaconf import OmegaConf
txt = open(sys.argv[1]).read()
txt = txt[txt.index("\n") + 1:] if txt.startswith("#") else txt          # hydra may prefix a comment line
try:
    cfg = OmegaConf.create(yaml.safe_load(txt))
except Exception as e:  # noqa: BLE001
    print(f"[meta] ABORT: could not parse the rendered config as YAML ({type(e).__name__}: {e})")
    print(f"--- {sys.argv[1]} head:"); print("".join(open(sys.argv[1]).readlines()[:20]))
    print(f"--- {sys.argv[2]} tail:"); print("".join(open(sys.argv[2]).readlines()[-30:]))
    sys.exit(1)
expect = {
  "algorithm.rollout_correction.rollout_is": "token", "algorithm.rollout_correction.rollout_is_threshold": ${IS_THRESHOLD},
  "algorithm.rollout_correction.rollout_rs": None, "algorithm.rollout_correction.rollout_is_batch_normalize": False,
  "algorithm.rollout_correction.bypass_mode": False, "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
  "actor_rollout_ref.actor.clip_ratio_low": ${clip_ratio_low}, "actor_rollout_ref.actor.clip_ratio_high": ${clip_ratio_high},
  "actor_rollout_ref.actor.clip_ratio_c": ${clip_ratio_c}, "actor_rollout_ref.actor.loss_agg_mode": "${loss_agg_mode}",
  "actor_rollout_ref.actor.use_dynamic_bsz": False, "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": ${micro_bsz_per_gpu},
  "actor_rollout_ref.actor.ppo_mini_batch_size": ${ppo_mini_batch_size}, "actor_rollout_ref.actor.ppo_epochs": 1,
  "actor_rollout_ref.actor.use_kl_loss": False, "actor_rollout_ref.actor.entropy_coeff": 0,
  "actor_rollout_ref.actor.optim.lr": ${actor_lr}, "actor_rollout_ref.actor.optim.lr_scheduler_type": "${lr_scheduler}",
  "actor_rollout_ref.actor.optim.lr_warmup_steps": ${lr_warmup_steps}, "actor_rollout_ref.actor.optim.min_lr_ratio": 0.0,
  "actor_rollout_ref.actor.optim.weight_decay": ${weight_decay}, "actor_rollout_ref.actor.optim.clip_grad": ${grad_clip},
  "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
  "actor_rollout_ref.rollout.n": ${rollout_n}, "actor_rollout_ref.rollout.temperature": ${temperature},
  "actor_rollout_ref.rollout.top_p": ${top_p}, "actor_rollout_ref.rollout.top_k": ${top_k}, "actor_rollout_ref.rollout.calculate_log_probs": True,
  "actor_rollout_ref.rollout.val_kwargs.do_sample": False, "actor_rollout_ref.rollout.val_kwargs.n": 1,
  "data.train_batch_size": ${train_batch_size}, "data.max_prompt_length": ${max_prompt_length}, "data.max_response_length": ${max_response_length},
  "trainer.total_training_steps": ${TOTAL_STEPS}, "trainer.test_freq": ${TEST_FREQ}, "trainer.nnodes": ${NNODES}, "trainer.n_gpus_per_node": ${GPUS_PER_NODE},
}
bad = []
for key, want in expect.items():
    got = OmegaConf.select(cfg, key, default="<MISSING>")
    if got == "<MISSING>":
        bad.append(f"{key}: missing"); continue
    if want is None:
        ok = got is None
    elif isinstance(want, bool):
        ok = bool(got) == want and isinstance(got, bool)
    elif isinstance(want, (int, float)):
        try: ok = abs(float(got) - float(want)) <= 1e-12 * max(1.0, abs(float(want)))
        except (TypeError, ValueError): ok = False
    else:
        ok = str(got) == str(want)
    if not ok:
        bad.append(f"{key}: got {got!r}, want {want!r}")
if bad:
    print("[meta] ABORT: resolved config mismatch:\n  " + "\n  ".join(bad))
    print(f"--- {sys.argv[2]} tail:"); print("".join(open(sys.argv[2]).readlines()[-15:]))
    sys.exit(1)
print(f"[meta] resolved-config pre-flight OK: {len(expect)} fields verified numerically from {sys.argv[1]}")
PYEOF

########################### TB mirror + signal handling ###################################
( while true; do sleep 300; cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true; done ) &
TB_SYNC_PID=$!; GUARD_PID=""; DRIVER_PID=""
on_signal() { echo "[meta] caught signal -- stopping driver ${DRIVER_PID:-<none>}"; [ -n "${DRIVER_PID}" ] && kill -TERM "${DRIVER_PID}" 2>/dev/null || true; }
trap 'on_signal; exit 130' INT
trap 'on_signal; exit 143' TERM
on_exit() {
  if [ -n "${DRIVER_PID}" ] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
    echo "[meta] exit: driver ${DRIVER_PID} still alive -- TERM"; kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 "$(( ${DRIVER_KILL_GRACE:-30} / 2 ))"); do kill -0 "${DRIVER_PID}" 2>/dev/null || break; sleep 2; done
    kill -0 "${DRIVER_PID}" 2>/dev/null && { echo "[meta] exit: KILL"; kill -KILL "${DRIVER_PID}" 2>/dev/null || true; }
  fi
  kill ${TB_SYNC_PID} ${GUARD_PID} 2>/dev/null || true
  cp -r "${TB_DIR}/." "${TB_MIRROR}/" 2>/dev/null || true
  test -f "${TB_DIR}/COLLAPSE_ABORT.txt" && { echo "[meta] RUN ABORTED BY COLLAPSE GUARD:"; cat "${TB_DIR}/COLLAPSE_ABORT.txt"; }
  test -f "${TB_DIR}/COLLAPSE_WARN.txt" && { echo "[meta] guard warnings:"; cat "${TB_DIR}/COLLAPSE_WARN.txt"; }
  return 0
}
trap on_exit EXIT

echo "[recipe] model=Qwen3-0.6B-Base IS=token/${IS_THRESHOLD}(no-bypass,no-norm,no-rs) dual_clip=${clip_ratio_c} eps=1e-8 fused=False lr=${actor_lr} sched=${lr_scheduler} warmup=${lr_warmup_steps} steps=${TOTAL_STEPS} batch=${train_batch_size}x${rollout_n} mini=${ppo_mini_batch_size} (mu=1) micro/gpu=${micro_bsz_per_gpu} dyn_bsz=False clip=${clip_ratio_low}/${clip_ratio_high} kl=${kl_loss_coef} T=${temperature} top_p=${top_p} top_k=${top_k} prompt=${max_prompt_length} cap=${max_response_length} fmt_score=${REWARD_FORMAT_SCORE} overlong=${REWARD_OVERLONG_BUFFER}/${REWARD_OVERLONG_PENALTY} penalty_sources=${REWARD_PENALTY_SOURCES} wd=${weight_decay} loss_agg=${loss_agg_mode} stop=[151645,151643] fp32-master eval=test512@${TEST_FREQ}"
echo "[meta] tensorboard -> ${TB_DIR}"; echo "[meta] tb mirror -> ${TB_MIRROR}"; echo "[meta] resolved config -> ${CFG_LOG}"

########################### launch ####################################################
python3 -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" "$@" &
DRIVER_PID=$!
echo "[meta] driver pid ${DRIVER_PID}"
GUARD_LOG=${LOG_DIR}/${EXPERIMENT_NAME}/collapse_guard.log; mkdir -p "$(dirname "${GUARD_LOG}")"
if [ "${COLLAPSE_GUARD:-1}" = "1" ]; then
  # guard tuned for this recipe: 2048 samples/step, 128 groups x 16; warmup 20 (fast early rise)
  python3 "${SCRIPTS_DIR}/collapse_guard.py" --tb "${TB_DIR}" --rollout "${ROLLOUT_DUMP_DIR}" --pid "${DRIVER_PID}" \
      --groups ${train_batch_size} --group_size ${rollout_n} --poll 60 --warmup 20 > "${GUARD_LOG}" 2>&1 &
  GUARD_PID=$!; sleep 5
  kill -0 "${GUARD_PID}" 2>/dev/null || { echo "[meta] ABORT: collapse guard died at start:"; cat "${GUARD_LOG}"; kill -TERM "${DRIVER_PID}" 2>/dev/null; exit 2; }
  grep -q "expecting ${train_batch_size}x${rollout_n}" "${GUARD_LOG}" || { echo "[meta] ABORT: guard not configured for ${train_batch_size}x${rollout_n}"; kill -TERM "${DRIVER_PID}" 2>/dev/null; exit 2; }
  echo "[meta] collapse guard pid ${GUARD_PID} (expecting ${train_batch_size}x${rollout_n} rows/step) -> ${GUARD_LOG}"
fi
set +e; wait "${DRIVER_PID}"; DRIVER_RC=$?
while kill -0 "${DRIVER_PID}" 2>/dev/null; do wait "${DRIVER_PID}"; DRIVER_RC=$?; done
set -e
echo "[meta] driver exited with rc=${DRIVER_RC}"
exit ${DRIVER_RC}