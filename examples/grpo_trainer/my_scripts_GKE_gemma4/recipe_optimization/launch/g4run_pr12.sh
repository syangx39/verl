#!/usr/bin/env bash
# =============================================================================
# PR #12 arm launcher (RECIPE_OPTIMIZATION_PLAN.md section 3): A4 | A5 | A6 | A6SMOKE.
#
# Every arm starts from the initial checkpoint with fresh optimizer state, seed 1, 16n64g, eval at 0/20/.../100 on the
# frozen eval_5000, and differs from the A3a control (gemma4_e2b_rl05fp32_a3lr2e6_16n64g_s100_seed1_1023_0753:
# constant LR 2e-6) only by the overrides of its case below:
#   A4       native half-cosine: peak 2e-6, num_cycles=0.25, min_lr_ratio=0, no warmup, zero-indexed; the trainer
#            injects total_training_steps=100, so update k uses 1e-6 * (1 + cos(pi * (k - 1) / 200))
#   A5       training temperature 0.8 -> 1.0 (top-p 0.95, top-k 50, n=8 and eval decoding unchanged)
#   A6       completions per prompt 8 -> 16 (256 prompts -> 4,096 completions, still one optimizer update per step:
#            mini-batch = 256 x n)
#   A6SMOKE  A6's overrides for 2 steps without eval: memory / grouping / timing / monitor-latency check before A6
#
# Frozen launcher run_gemma4_e2b_rl05_parity.sh (sha256 7d511735...) and wrapper (2a37aad5...): the launcher sets
# actor_rollout_ref.actor.optim.lr=1e-6; the overrides here are appended last via the wrapper's "$@" and win.
# Format coefficient 0.1 everywhere and the double-wrapped train.parquet: the A1 variables and GEMMA_TRAIN_FILE are unset.
# PYTHONUNBUFFERED=1 is set for the driver process only (not a Hydra key, no recipe effect): the TaskRunner's per-step
# console metric line then reaches the driver log when the step finishes instead of sitting in the driver's stdout
# buffer, which the stop monitor needs (RECIPE_OPTIMIZATION_PLAN.md section 4.1).
#
# The stop monitor (gemma4_stop_monitor.py) is started first, detached; it acts (--act) on the 100-step arms and only
# observes the smoke. START_MONITOR=0 skips it.
#
# Usage on the Ray head pod:
#   nohup setsid bash recipe_optimization/launch/g4run_pr12.sh A4 > /dev/null 2>&1 < /dev/null &
# The experiment name is printed to /tmp/g4_<arm>.current and appended to /workspace/meta-RL/logs/gemma4_pr12_launches.tsv.
# =============================================================================
set -u
ARM=${1:?usage: g4run_pr12.sh A4|A5|A6|A6SMOKE}
SCRIPT_DIR=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4
RO=${SCRIPT_DIR}/recipe_optimization
LOGS=/workspace/meta-RL/logs
source /workspace/setup_env.sh

STEPS=100; VAL_BEFORE=True; FREQ=20; MON_ARM=${ARM}; MON_ACT=--act; MON_MAX=100
COMMON=(actor_rollout_ref.actor.optim.lr=2e-6)
case "${ARM}" in
  A4)
    TAG=a4cos
    OVR=(actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
         actor_rollout_ref.actor.optim.num_cycles=0.25
         actor_rollout_ref.actor.optim.min_lr_ratio=0.0
         actor_rollout_ref.actor.optim.lr_warmup_steps=0
         actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
         actor_rollout_ref.actor.optim.zero_indexed_step=true)
    EXPECT_LR=${RO}/a4_cosine/used_lr_pure.json ;;
  A5)
    TAG=a5t1; OVR=(actor_rollout_ref.rollout.temperature=1.0); EXPECT_LR=2e-6 ;;
  A6)
    TAG=a6n16; OVR=(actor_rollout_ref.rollout.n=16); EXPECT_LR=2e-6 ;;
  A6SMOKE)
    TAG=a6n16smoke; OVR=(actor_rollout_ref.rollout.n=16); EXPECT_LR=2e-6
    STEPS=2; VAL_BEFORE=False; FREQ=-1; MON_ARM=A6; MON_ACT=""; MON_MAX=2 ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

EXP=gemma4_e2b_rl05fp32_${TAG}_16n64g_s${STEPS}_seed1_$(date -u +%m%d_%H%M)
LOG=${LOGS}/${EXP}.log
ARM_LC=$(echo "${ARM}" | tr '[:upper:]' '[:lower:]')
echo "${EXP}" > /tmp/g4_${ARM_LC}.current
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ARM}" "${EXP}" >> ${LOGS}/gemma4_pr12_launches.tsv

if [ "${START_MONITOR:-1}" = "1" ]; then
  MON_DIR=${LOGS}/gemma4_stop_monitor/${EXP}
  mkdir -p "${MON_DIR}"
  nohup setsid python3 ${SCRIPT_DIR}/gemma4_stop_monitor.py --exp ${EXP} --arm ${MON_ARM} ${MON_ACT} \
      --max-step ${MON_MAX} --expect-lr ${EXPECT_LR} > ${MON_DIR}/monitor.out 2>&1 < /dev/null &
  MON_PID=$!
else
  MON_PID=none
fi

{
  echo "[pr12-provenance] arm=${ARM} exp=${EXP} steps=${STEPS} val_before_train=${VAL_BEFORE} test_freq=${FREQ}" \
       "overrides=${COMMON[*]} ${OVR[*]} driver_env=PYTHONUNBUFFERED=1 monitor_pid=${MON_PID} monitor_act=${MON_ACT:-no}" \
       "train_sha256=$(sha256sum /workspace/meta-RL/data/openmathinstruct2/train.parquet | cut -d' ' -f1)" \
       "reward_sha256=$(sha256sum ${SCRIPT_DIR}/maxtext_math_reward.py | cut -d' ' -f1)" \
       "launcher_sha256=$(sha256sum ${SCRIPT_DIR}/run_gemma4_e2b_rl05_parity.sh | cut -d' ' -f1)" \
       "wrapper_sha256=$(sha256sum ${SCRIPT_DIR}/run_gemma4_e2b_rl05_accuracy_baseline.sh | cut -d' ' -f1)" \
       "monitor_sha256=$(sha256sum ${SCRIPT_DIR}/gemma4_stop_monitor.py | cut -d' ' -f1)" \
       "arm_launcher_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  env -u REWARD_FMT_WEIGHT_TRAIN -u REWARD_EVAL_QUESTIONS -u GEMMA_TRAIN_FILE \
    PYTHONUNBUFFERED=1 SEED=1 NNODES=16 TOTAL_STEPS=${STEPS} TEST_FREQ=${FREQ} VAL_BEFORE_TRAIN=${VAL_BEFORE} \
    EXPERIMENT_NAME=${EXP} \
    bash ${SCRIPT_DIR}/run_gemma4_e2b_rl05_accuracy_baseline.sh "${COMMON[@]}" "${OVR[@]}"
} > ${LOG} 2>&1
RC=$?
echo "LAUNCHER_RC=${RC}" >> ${LOG}
exit ${RC}
