#!/usr/bin/env bash
# =============================================================================
# PR #14 run launcher for the Gemma-4 E2B GSM8K 8K baseline: SMOKE | SEED1 | SEED2 | SEED3.
#
#   SMOKE  the production recipe and 250-step LR horizon, stopped after STOP_AFTER_STEPS (default 3) updates by a
#          watcher that SIGTERMs only this run's driver. Step-0 full eval runs (checks the eval path and gives
#          Gemma's GSM8K starting accuracy). The optimizer probe records the first 3 AdamW steps on every rank
#          (update 1 has LR 0, update 2 must apply a nonzero FP32 update). No checkpoints.
#   SEEDk  the fixed 250-step run from the initial checkpoint, full eval every 20 updates and at 250.
#
# The driver runs with PYTHONUNBUFFERED=1 (logging only: the per-step metric line reaches the driver log as the step
# finishes). The experiment name is written to /tmp/g4_gsm8k_<mode>.current and appended to
# /workspace/meta-RL/logs/gemma4_gsm8k_launches.tsv.
# Usage on the Ray head pod:
#   nohup setsid bash gsm8k_8k/launch/g4run_gsm8k.sh SMOKE > /dev/null 2>&1 < /dev/null &
# =============================================================================
set -u
MODE=${1:?usage: g4run_gsm8k.sh SMOKE|SEED1|SEED2|SEED3}
SCRIPT_DIR=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4
LOGS=/workspace/meta-RL/logs
source /workspace/setup_env.sh
TS=$(date -u +%m%d_%H%M)
EXTRA_ENV=()
case "${MODE}" in
  SMOKE)
    STOP_AFTER=${STOP_AFTER_STEPS:-3}
    SEED=1
    EXP=gemma4_e2b_gsm8k8k_smoke_16n64g_s250stop${STOP_AFTER}_seed1_${TS}
    EXTRA_ENV=(SAVE_FREQ=-1 OPTIM_PROBE_DIR=${LOGS}/gemma4_optim_probe/${EXP} OPTIM_PROBE_STEPS=3) ;;
  SEED1|SEED2|SEED3)
    STOP_AFTER=0
    SEED=${MODE#SEED}
    EXP=gemma4_e2b_gsm8k8k_16n64g_s250_seed${SEED}_${TS}
    EXTRA_ENV=(SAVE_FREQ=${SAVE_FREQ:-50}) ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
LOG=${LOGS}/${EXP}.log
MODE_LC=$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')
echo "${EXP}" > /tmp/g4_gsm8k_${MODE_LC}.current
printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${MODE}" "${EXP}" >> ${LOGS}/gemma4_gsm8k_launches.tsv

if [ "${STOP_AFTER}" -gt 0 ]; then
  # Stop watcher: after the metric line of update STOP_AFTER appears, SIGTERM only this run's driver
  # (argv contains both verl.trainer.main_ppo and trainer.experiment_name=EXP).
  (
    NOTE=${LOGS}/gemma4_gsm8k/${EXP}/smoke_stop.txt
    while ! grep -q "step:${STOP_AFTER} - " "${LOG}" 2>/dev/null; do
      grep -q "LAUNCHER_RC=" "${LOG}" 2>/dev/null && exit 0
      sleep 10
    done
    PIDS=""
    for p in $(pgrep -f "trainer.experiment_name=${EXP}"); do
      tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -qx "verl.trainer.main_ppo" && PIDS="${PIDS} $p"
    done
    mkdir -p "$(dirname "${NOTE}")"
    echo "[smoke-stop] update ${STOP_AFTER} logged; SIGTERM to driver pid(s)${PIDS} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${NOTE}"
    [ -n "${PIDS}" ] && kill -TERM ${PIDS}
  ) > /dev/null 2>&1 < /dev/null &
fi

{
  echo "[gsm8k-run] mode=${MODE} exp=${EXP} seed=${SEED} stop_after=${STOP_AFTER} extra_env=${EXTRA_ENV[*]}" \
       "run_launcher_sha256=$(sha256sum "$0" | cut -d' ' -f1)"
  env -u REWARD_FMT_WEIGHT_TRAIN -u REWARD_EVAL_QUESTIONS -u GEMMA_TRAIN_FILE -u GEMMA_VAL_FILE \
    PYTHONUNBUFFERED=1 SEED=${SEED} NNODES=16 TOTAL_STEPS=250 TEST_FREQ=20 EXPERIMENT_NAME=${EXP} "${EXTRA_ENV[@]}" \
    bash ${SCRIPT_DIR}/run_gemma4_e2b_gsm8k_8k.sh
} > ${LOG} 2>&1
RC=$?
echo "LAUNCHER_RC=${RC}" >> ${LOG}
exit ${RC}
