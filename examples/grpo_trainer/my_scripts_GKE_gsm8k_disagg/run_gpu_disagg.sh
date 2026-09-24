#!/usr/bin/env bash
# Launch from the existing Ray head. Does not start/stop or reconfigure Ray.
set -euo pipefail
export RECIPE_DIR
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_REPO=${VERL_REPO:-/tmp/verl-disagg-9924801/src}   # node-local checkout made by prepare_env.py on every node (same path everywhere)
export DISAGG_PYTHON=${DISAGG_PYTHON:-/tmp/verl-disagg-9924801/bin/python}
export MODEL_PATH=${MODEL_PATH:-/workspace/meta-RL/models/Qwen3-0.6B}
export DATA_DIR=${DATA_DIR:-/workspace/meta-RL/data/gsm8k_boxed}
export LOG_DIR=${LOG_DIR:-/workspace/meta-RL/logs/wenjun_disagg}
export CKPT_DIR=${CKPT_DIR:-/workspace/meta-RL/ckpt/wenjun_disagg}
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export SEED=${SEED:-1}
export TOTAL_STEPS=${TOTAL_STEPS:-250}
export TEST_FREQ=${TEST_FREQ:-20}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0p6b_disagg_t16_r48_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}
export RUN_DIR=$LOG_DIR/$EXPERIMENT_NAME
export TB_DIR=/tmp/tb_local/wenjun_gpu_disagg_gsm8k/$EXPERIMENT_NAME
export PYTHONPATH=$VERL_REPO:$RECIPE_DIR
export PYTHONNOUSERSITE=1
export VERL_PLATFORM=nvidia VLLM_USE_V1=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=$SEED
export REWARD_FORMAT_SCORE=0.1 REWARD_OVERLONG_BUFFER=512
export REWARD_OVERLONG_PENALTY=1.0 REWARD_MAX_RESP_LEN=2048
export REWARD_PENALTY_SOURCES=gsm8k_boxed_train
unset LOGPROB_FIXTURE_DIR LOGPROB_FIXTURE_STEP INJECT_BATCH_NPZ INJECT_BATCH_STEP
PIN=9924801779415f86c807b5716a3d4479fa60f811
test -x "$DISAGG_PYTHON" || { echo "Run prepare_env.py first: $DISAGG_PYTHON missing" >&2; exit 2; }
test "$(git -C "$VERL_REPO" rev-parse HEAD)" = "$PIN" || { echo "Wrong verl commit; require $PIN" >&2; exit 2; }
test -z "$(git -C "$VERL_REPO" status --porcelain --untracked-files=no)" || { echo "Pinned checkout has tracked modifications" >&2; exit 2; }
"$DISAGG_PYTHON" - "$RECIPE_DIR" "$VERL_REPO" "$DISAGG_PYTHON" <<'PY'
import json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1], "ENV_MANIFEST_PATH").read_text().strip()
m = json.loads(pathlib.Path(manifest_path).read_text())
assert m["ok"], f"Environment preparation failed: {manifest_path}"
assert m["repo"] == sys.argv[2]
assert pathlib.Path(sys.argv[3]).parent.parent == pathlib.Path(m["venv"])
PY
mkdir -p "$LOG_DIR" "$CKPT_DIR"
mkdir "$RUN_DIR"  # Do not overwrite an earlier run's evidence.
mkdir -p "$RUN_DIR/tensorboard"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/latest_seed${SEED}.txt"
printf '%s\n' "$PIN" > "$RUN_DIR/verl_commit.txt"
printf '%q ' "$DISAGG_PYTHON" -m verl.trainer.main_ppo --config-path "$RECIPE_DIR" --config-name recipe_gpu_disagg "$@" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

cd "$VERL_REPO"
"$DISAGG_PYTHON" -m verl.trainer.main_ppo --config-path "$RECIPE_DIR" --config-name recipe_gpu_disagg \
  --cfg job --resolve "$@" > "$RUN_DIR/resolved_config.yaml" 2> "$RUN_DIR/config.stderr.log"
"$DISAGG_PYTHON" "$RECIPE_DIR/preflight.py" --config "$RUN_DIR/resolved_config.yaml" \
  --out "$RUN_DIR/preflight.json"
"$DISAGG_PYTHON" "$RECIPE_DIR/boxed_math_reward.py" > "$RUN_DIR/reward_selftest.log"
"$DISAGG_PYTHON" -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions())))' > "$RUN_DIR/packages.txt"
cp "$RECIPE_DIR/recipe_gpu_disagg.yaml" "$RUN_DIR/recipe_gpu_disagg.yaml"
cp "$(cat "$RECIPE_DIR/ENV_MANIFEST_PATH")" "$RUN_DIR/environment_manifest.json"
echo "[recipe] trainer=16 rollout=48 fsdp2/vllm TP=1 batch=128x16 mu=1 cap=2048 penalty=train-only TIS=token/3 detached single-forward"
echo "[recipe] lr=2e-6 warmup=10 cosine steps=$TOTAL_STEPS seed=$SEED sync=1 threshold=2/drop eval=1319"
echo "[run_dir] $RUN_DIR"
if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo '[preflight] OK; no training launched'
  exit 0
fi
date +%s > "$RUN_DIR/start_epoch.txt"
on_exit() {
  rc=$?; trap - EXIT
  date +%s > "$RUN_DIR/end_epoch.txt"
  printf '%s\n' "$rc" > "$RUN_DIR/exit_code.txt"
  echo "[driver] exited rc=$rc"
  "$DISAGG_PYTHON" "$RECIPE_DIR/collect_tb.py" --source "$TB_DIR" --out "$RUN_DIR/tensorboard" \
    > "$RUN_DIR/collect_tb.log" 2>&1 || echo "[evidence] TB collection failed; see $RUN_DIR/collect_tb.log"
  exit "$rc"
}
trap on_exit EXIT
"$DISAGG_PYTHON" -m verl.trainer.main_ppo --config-path "$RECIPE_DIR" --config-name recipe_gpu_disagg \
  "$@" 2>&1 | tee "$RUN_DIR/driver.log"
