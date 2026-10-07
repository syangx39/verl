#!/usr/bin/env bash
# Serial sweep over models x rollout TP with run_grpo_colocated.sh. Each config waits for an idle cluster, runs, and
# prints the step-time report; a training-side OOM is retried once with a smaller batching budget (a reported system knob).
# usage: MODELS="4B 8B" TPS="1 2 4" [MODEL_ROOT=/tmp/models] [GPU_MEM_UTIL=..] [PPO_MAX_TOK=..] [EXTRA="key=val ..."] bash run_models_sweep.sh
#   MODEL_ROOT: where the Qwen3-<size> directories are loaded from (default: the shared /workspace/meta-RL/models; use a
#   node-local copy made by stage_model.py for large or tied-embedding models). Completeness is checked on the shared copy.
#   EXTRA: extra Hydra overrides passed to every run (space-separated, no spaces inside an override).
set -uo pipefail
source /workspace/setup_env.sh; export RAY_ADDRESS=auto
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
L=${GRPO_LOG_DIR:-/workspace/meta-RL/logs/grpo_perf}; mkdir -p "$L"; export GRPO_LOG_DIR=$L
SRC_ROOT=/workspace/meta-RL/models
wait_idle(){ for i in $(seq 1 30); do ray status 2>/dev/null | grep -qE "^ *0(\.0)?/64(\.0)? GPU" && return 0; sleep 10; done
             echo "!!! GPUs not idle after 5 min:"; ray status | grep -E "GPU|placement"; return 1; }
model_ok(){ "$DISAGG_PYTHON" -c "
import json, os, sys
from safetensors import safe_open
d = sys.argv[1]; idx = d + '/model.safetensors.index.json'
wm = json.load(open(idx))['weight_map'] if os.path.exists(idx) else None
shards = sorted(set(wm.values())) if wm else ['model.safetensors']
n = sum(len(list(safe_open(f'{d}/{x}', 'pt').keys())) for x in shards)
sys.exit(0 if (wm is None or n == len(wm)) and os.path.isfile(d + '/tokenizer.json') else 1)" "$1" 2>/dev/null; }
run_one(){  # $1 model size, $2 TP, $3 log suffix; knobs come from the environment
  # shellcheck disable=SC2086  # EXTRA is split into separate overrides on purpose
  MODEL_PATH=${MODEL_ROOT:-$SRC_ROOT}/Qwen3-$1 TP=$2 bash "$SCRIPT_DIR/run_grpo_colocated.sh" ${EXTRA:-} > "$L/tp$2_Qwen3-$1$3.log" 2>&1; }
report(){ RUN=$(ls -d "$L"/grpo_Qwen3-$1_tp$2_2*/ 2>/dev/null | sort | tail -1)
          [ -n "$RUN" ] && "$DISAGG_PYTHON" "$SCRIPT_DIR/step_time_report.py" "${RUN}driver.log" 2>&1 | head -4; }
for s in ${MODELS:-0.6B 1.7B 4B 8B 14B 32B}; do
  model_ok "$SRC_ROOT/Qwen3-$s" || { echo "=== SKIP Qwen3-$s: model missing or incomplete under $SRC_ROOT"; continue; }
  for TP in ${TPS:-1 2 4}; do
    wait_idle || exit 1
    echo "=== Qwen3-$s TP=$TP start $(date -u +%H:%M:%S)  knobs: PPO_MAX_TOK=${PPO_MAX_TOK:-32768} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6} MODEL_ROOT=${MODEL_ROOT:-$SRC_ROOT} EXTRA='${EXTRA:-}'"
    run_one "$s" "$TP" ""; rc=$?
    if [ $rc -ne 0 ] && grep -qiE "out of memory|OutOfMemoryError" "$L/tp${TP}_Qwen3-$s.log"; then
      echo "=== Qwen3-$s TP=$TP OOM -> retry with PPO_MAX_TOK=16384 GPU_MEM_UTIL=0.5"; sleep 60; wait_idle || exit 1
      PPO_MAX_TOK=16384 GPU_MEM_UTIL=0.5 run_one "$s" "$TP" "_retry"; rc=$?
    fi
    echo "=== Qwen3-$s TP=$TP rc=$rc end $(date -u +%H:%M:%S)"
    if [ $rc -eq 0 ]; then report "$s" "$TP"
    else grep -hE "Error|Exception|less than" "$L"/tp${TP}_Qwen3-$s*.log | grep -v "ModuleNotFoundError\|engine is not available\|warnings.warn" | tail -3 | cut -c1-250; fi
    sleep 30
  done
done
