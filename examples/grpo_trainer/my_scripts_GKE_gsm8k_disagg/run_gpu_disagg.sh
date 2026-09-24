#!/usr/bin/env bash
# Launch from the existing Ray head. Does not start/stop or reconfigure Ray.
set -euo pipefail
export RECIPE_DIR
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_REPO=${VERL_REPO:-/workspace/verl-pin}          # upstream checkout inside the image
export DISAGG_PYTHON=${DISAGG_PYTHON:-/workspace/verl-pin/.venv/bin/python}
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
# Upstream main_ppo.py passes ray_kwargs.ray_init to ray.init() via OmegaConf.to_container() WITHOUT resolve=True, so the
# ${oc.env:...} interpolations in the recipe's ray_kwargs block would reach Ray verbatim ("bad substitution" in the
# raylet's worker command). Pass every such value as a literal hydra override instead; overrides take precedence.
RE=ray_kwargs.ray_init.runtime_env
RAY_ENV_OVERRIDES=(
  "$RE.py_executable=$DISAGG_PYTHON"
  "$RE.env_vars.VERL_REPO=$VERL_REPO" "$RE.env_vars.RECIPE_DIR=$RECIPE_DIR" "$RE.env_vars.DISAGG_PYTHON=$DISAGG_PYTHON"
  "$RE.env_vars.MODEL_PATH=$MODEL_PATH" "$RE.env_vars.DATA_DIR=$DATA_DIR" "$RE.env_vars.RUN_DIR=$RUN_DIR" "$RE.env_vars.CKPT_DIR=$CKPT_DIR"
  "$RE.env_vars.EXPERIMENT_NAME=$EXPERIMENT_NAME" "$RE.env_vars.TB_DIR=$TB_DIR" "$RE.env_vars.TENSORBOARD_DIR=$TB_DIR"
  "$RE.env_vars.SEED=$SEED" "$RE.env_vars.PYTHONHASHSEED=$SEED" "$RE.env_vars.TOTAL_STEPS=$TOTAL_STEPS" "$RE.env_vars.TEST_FREQ=$TEST_FREQ"
  "$RE.env_vars.SAVE_FREQ=$SAVE_FREQ" "$RE.env_vars.VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN" "$RE.env_vars.PYTHONPATH=$VERL_REPO:$RECIPE_DIR"
)
PIN=ace775e87d8765bcdd114aac734ab71da5367a0f   # upstream verl-project/verl main (2026-09-22)
test -x "$DISAGG_PYTHON" || { echo "Run prepare_env.py first: $DISAGG_PYTHON missing" >&2; exit 2; }
if [[ ${DISAGG_DEV_SOURCE:-0} == 1 ]]; then
  # Development iteration only: VERL_REPO may be a patched working copy (e.g. on the shared bucket). The run is recorded
  # as dev source and is NOT a reference run. Formal runs must not set this: they require the pinned, clean image checkout.
  echo "[dev] DISAGG_DEV_SOURCE=1: using $VERL_REPO at $(git -C "$VERL_REPO" rev-parse --short HEAD) with $(git -C "$VERL_REPO" status --porcelain --untracked-files=no | wc -l) modified tracked file(s); not a reference run"
else
  test "$(git -C "$VERL_REPO" rev-parse HEAD)" = "$PIN" || { echo "Wrong verl commit; require $PIN" >&2; exit 2; }
  test -z "$(git -C "$VERL_REPO" status --porcelain --untracked-files=no)" || { echo "Pinned checkout has tracked modifications" >&2; exit 2; }
fi
if [[ ${DISAGG_IMAGE_MODE:-0} == 1 ]]; then
  # Environment comes from the derived container image (venv built from the pinned commit's uv.lock), identical on every
  # node by construction; there is no prepare_env manifest. Verify the venv against the lock and require DISAGG_IMAGE.
  test -n "${DISAGG_IMAGE:-}" || { echo "DISAGG_IMAGE must name the deployed image (registry path@digest)" >&2; exit 2; }
  "$DISAGG_PYTHON" "$RECIPE_DIR/verify_venv_lock.py" "$VERL_REPO"
else
"$DISAGG_PYTHON" - "$RECIPE_DIR" "$VERL_REPO" "$DISAGG_PYTHON" <<'PY'
import json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1], "ENV_MANIFEST_PATH").read_text().strip()
m = json.loads(pathlib.Path(manifest_path).read_text())
assert m["ok"], f"Environment preparation failed: {manifest_path}"
assert m["repo"] == sys.argv[2]
assert pathlib.Path(sys.argv[3]).parent.parent == pathlib.Path(m["venv"])
PY
fi
mkdir -p "$LOG_DIR" "$CKPT_DIR"
mkdir "$RUN_DIR"  # Do not overwrite an earlier run's evidence.
mkdir -p "$RUN_DIR/tensorboard"
printf '%s\n' "$RUN_DIR" > "$LOG_DIR/latest_seed${SEED}.txt"
printf '%s\n' "$PIN" > "$RUN_DIR/verl_commit.txt"
printf '%q ' "$DISAGG_PYTHON" -m verl.trainer.main_ppo --config-path "$RECIPE_DIR" --config-name recipe_gpu_disagg "${RAY_ENV_OVERRIDES[@]}" "$@" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

cd "$VERL_REPO"
"$DISAGG_PYTHON" -m verl.trainer.main_ppo --config-path "$RECIPE_DIR" --config-name recipe_gpu_disagg \
  --cfg job --resolve "${RAY_ENV_OVERRIDES[@]}" "$@" > "$RUN_DIR/resolved_config.yaml" 2> "$RUN_DIR/config.stderr.log"
"$DISAGG_PYTHON" "$RECIPE_DIR/preflight.py" --config "$RUN_DIR/resolved_config.yaml" \
  --out "$RUN_DIR/preflight.json"
"$DISAGG_PYTHON" "$RECIPE_DIR/boxed_math_reward.py" > "$RUN_DIR/reward_selftest.log"
"$DISAGG_PYTHON" -c 'import importlib.metadata as m; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in m.distributions())))' > "$RUN_DIR/packages.txt"
cp "$RECIPE_DIR/recipe_gpu_disagg.yaml" "$RUN_DIR/recipe_gpu_disagg.yaml"
if [[ ${DISAGG_IMAGE_MODE:-0} == 1 ]]; then
  { echo "mode: container image"; echo "image: ${DISAGG_IMAGE}"; echo "verl_pin: $PIN";
    if [[ ${DISAGG_DEV_SOURCE:-0} == 1 ]]; then echo "source: DEV working copy $VERL_REPO @ $(git -C "$VERL_REPO" rev-parse HEAD) (NOT a reference run)"; git -C "$VERL_REPO" status --porcelain --untracked-files=no; git -C "$VERL_REPO" diff > "$RUN_DIR/dev_source.diff"; else echo "source: pinned image checkout $VERL_REPO"; fi
    cat /etc/os-release | head -2; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; } > "$RUN_DIR/environment_manifest.txt"
else
  cp "$(cat "$RECIPE_DIR/ENV_MANIFEST_PATH")" "$RUN_DIR/environment_manifest.json"
fi
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
  "${RAY_ENV_OVERRIDES[@]}" "$@" 2>&1 | tee "$RUN_DIR/driver.log"