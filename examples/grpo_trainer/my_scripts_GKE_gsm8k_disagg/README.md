# Wenjun GPU disaggregated GSM8K recipe — candidate v1

This is a **new GPU baseline candidate**, using the learning settings of the successful 2K synchronous runs. Its asynchronous learning curve and speed have **not** been measured. Wenjun's TorchTitan + verl + torchtpuvllm stack can subsequently target the measured GPU result.

| Item | Setting |
|---|---|
| Hardware | 64 GB200: trainer 4 nodes × 4 GPUs; rollout 12 nodes × 4 GPUs |
| Stack | verl V1 `separate_async`, FSDP2 trainer, vLLM rollout; TP=1 |
| Pool separation | `hybrid_engine=false`; no borrowing trainer GPUs for rollout |
| Model | Existing stop-set-patched Qwen3-0.6B post-trained model |
| Data | Existing Meta boxed GSM8K parquet: 7,473 train / full 1,319 test |
| Prompt / response cap | 512 / 2,048 tokens; stop IDs 151643 and 151645 |
| Reward | Correct boxed answer: 1; wrong nonempty boxed answer: 0.1; otherwise 0 |
| Overlong penalty | Training only: 0 through 1,536 tokens, linear to −1 at 2,048; evaluation uses raw reward |
| Batch | 128 prompt groups × 16 responses = 2,048 completions/update |
| Update | One minibatch, one epoch (μ=1); fixed microbatch 8/GPU |
| Advantage / loss | GRPO sample std (ddof=1), epsilon 1e-6; global token mean; EOS included |
| Correction | Detached token TIS, upper cap 3; no weight normalization or rejection sampling |
| Forward path | REINFORCE + TIS; skip separate old-policy log-prob inference |
| Optimizer | AdamW, LR 2e-6, betas .9/.999, eps 1e-8, weight decay 0, gradient clip 1, fused=false |
| Schedule | 10 warmup steps, cosine over 250 steps; zero-indexed schedule |
| Precision | FP32 master weights, BF16 compute, FP32 gradient reduction |
| Sampling | T=1, top_p=1, top_k=−1; no dynamic group filtering |
| KL / entropy coefficients | Both 0; entropy still measured |
| Async controls | Synchronize weights each update; one warmup batch; threshold=2, strategy=drop |
| Eval / checkpoint | Greedy n=1; eval at 0,20,…,240,250; checkpoint every 50 plus final |
| Seeds | Start seed 1; validate a second seed after the candidate learns |

The threshold bounds prompt dispatch age according to this implementation, **not an exact one-version bound on every generated token**. Partial rollouts may span weight versions. Record actual staleness, trajectory spans, dropped groups and TIS ratio statistics.

The asynchronous loss is `−mean_valid_tokens(A * stop_gradient(min(exp(clamp(logp_current − logp_sampler, −20, 20)), 3)) * logp_current)`. The numerator uses the training forward, while the denominator comes from the sampler for each generated token. Activation checkpointing can recompute activations; “single forward” refers to removing the separate old-log-prob inference pass.

## 1. Source and environment

Pinned source: [`jialei777/verl-upstream@9924801779415f86c807b5716a3d4479fa60f811`](https://github.com/jialei777/verl-upstream/tree/9924801779415f86c807b5716a3d4479fa60f811). The same V1 controller has a [GPU FSDP2 separate-async example](https://github.com/jialei777/verl-upstream/blob/9924801779415f86c807b5716a3d4479fa60f811/tests/special_e2e/run_v1_separate_async.sh).

The pinned GPU lock uses Python 3.12, Torch 2.11/CUDA 13, vLLM 0.24 and Transformers 5.9. This is a software-stack change from the previous Meta runs. Keep the old checkout and environment for those results. The setup script creates a node-local environment at `/tmp/verl-disagg-9924801`, uses the GPU dependency lock, and explicitly overlays the **running cluster's exact Ray build**. It never restarts Ray. The manifest records that overlay; do not claim an unmodified full lock when Ray differs from 2.55.1.

Prerequisites: an existing idle Ray cluster with 16 four-GPU nodes; Python 3.12 and matching Ray on all nodes; the same shared `/workspace/meta-RL` paths visible on the head and workers. Downloads need GitHub/PyPI/wheelhouse access. Environment preparation uses substantial node-local disk and is outside run timing. The per-node CUDA probe will reject an incompatible driver; a driver/image change is outside this script.

Put the downloaded archive on the Ray head at `/workspace/meta-RL/wenjun_gpu_disagg_recipe.tar.gz`, then run:

```bash
source /workspace/setup_env.sh
export RAY_ADDRESS=auto
export VERL_REPO=/workspace/meta-RL/verl-disagg
export RECIPE_DIR=/workspace/meta-RL/recipes/wenjun_recipe
export MODEL_PATH=/workspace/meta-RL/models/Qwen3-0.6B
export DATA_DIR=/workspace/meta-RL/data/gsm8k_boxed
export LOG_DIR=/workspace/meta-RL/logs/wenjun_disagg
export CKPT_DIR=/workspace/meta-RL/ckpt/wenjun_disagg
export DISAGG_PYTHON=/tmp/verl-disagg-9924801/bin/python
mkdir -p /workspace/meta-RL/recipes "$LOG_DIR" "$CKPT_DIR"
tar -xzf /workspace/meta-RL/wenjun_gpu_disagg_recipe.tar.gz -C /workspace/meta-RL/recipes

# Dedicated checkout. If already present, verify its pin instead of cloning over it.
git clone --depth 1 --branch tpu-main https://github.com/jialei777/verl-upstream.git "$VERL_REPO"
git -C "$VERL_REPO" fetch --depth 1 origin 9924801779415f86c807b5716a3d4479fa60f811
git -C "$VERL_REPO" checkout --detach 9924801779415f86c807b5716a3d4479fa60f811

# Use the current image's Python here. Prepares all live nodes, including head.
set -o pipefail
python3 "$RECIPE_DIR/prepare_env.py" --repo "$VERL_REPO" --bundle "$RECIPE_DIR" \
  2>&1 | tee "$LOG_DIR/prepare_env.log"
```

Require exit code 0 and an environment manifest with `ok=true`. On a custom Ray build, provide `--ray-wheel /shared/path/to/exact-ray.whl`. After worker pods are recreated, rerun preparation because `/tmp` is node-local. `prepare_env.py --check` verifies all environments without installing. Do not use plain `uv run` afterward: it can replace the preserved Ray version with the lock's version.

## 2. Three-step smoke, using the full 16/48 topology

```bash
ray status
# Expect 0/64 GPU used. Run the smoke in the foreground.
SEED=1 EXPERIMENT_NAME=disagg_smoke_$(date -u +%Y%m%d_%H%M%S) \
TOTAL_STEPS=3 TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=false \
  bash "$RECIPE_DIR/run_gpu_disagg.sh" 2>&1 | tee "$LOG_DIR/smoke.log"

SMOKE_DIR=$(cat "$LOG_DIR/latest_seed1.txt")
"$DISAGG_PYTHON" "$RECIPE_DIR/check_smoke.py" "$SMOKE_DIR"
```

Require `SMOKE OK`. The check needs finite, nonzero gradients, finite loss/entropy, TIS metrics and actual staleness metrics for all three steps. This is a startup/update check, not evidence of convergence. An `old_log_prob` timing key can still exist in V1: its timer wraps a metadata-copy path, so key presence alone does not imply a second model inference.

The launcher checks the full resolved config, model stop IDs, data source labels/counts, no train/test question overlap, exact actor TIS wiring, node environments and a BF16 CUDA operation on each GPU node. `PREFLIGHT_ONLY=1 bash ...` runs these checks without starting training.

## 3. Formal 250-step run

After the smoke passes and its resources are released:

```bash
ray status
EXP=disagg_t16_r48_seed1_$(date -u +%Y%m%d_%H%M%S)
SEED=1 EXPERIMENT_NAME="$EXP" TOTAL_STEPS=250 TEST_FREQ=20 SAVE_FREQ=50 VAL_BEFORE_TRAIN=true \
  nohup bash "$RECIPE_DIR/run_gpu_disagg.sh" > "$LOG_DIR/$EXP.launch.log" 2>&1 &
echo "PID=$! EXP=$EXP"
tail -f "$LOG_DIR/$EXP.launch.log"
```

The script already creates the start/end timestamps. Do not wrap a second timestamp around environment installation. For seed 2, change only `SEED=2` and the experiment name; use the same dependency manifest and settings.

All recorded seeds are explicit in the data loader, FSDP engine and rollout engine. This controls randomness but does not imply bitwise reproducibility for asynchronous execution.

## 4. Outputs and interpretation

Each run writes `$LOG_DIR/$EXP/` with:

- `driver.log`, `resolved_config.yaml`, `preflight.json`, `environment_manifest.json`, `packages.txt`, `verl_commit.txt`.
- `start_epoch.txt`, `end_epoch.txt`, `exit_code.txt`; checkpoints under `$CKPT_DIR/$EXP/`.
- Full `val_dump/`, sampled training-batch `rollout_dump/` and `tensorboard/<node-id>/`.

TensorBoard writes to `/tmp/tb_local/wenjun_gpu_disagg_gsm8k/$EXP` on the TaskRunner's node during training, then gets copied into the run directory on exit. Use `tensorboard --logdir "$LOG_DIR/$EXP/tensorboard"`. If interrupted during collection, rerun `collect_tb.py --source /tmp/tb_local/wenjun_gpu_disagg_gsm8k/$EXP --out "$LOG_DIR/$EXP/tensorboard"` while those pods are alive.

Inspect full-test `val-core/gsm8k_boxed_test/acc/mean@1`, reward, response length/cap hits, gradient norm, entropy, `actor/rollout_corr/*`, `training/off_policy/trajectory_staleness*`, `trajectory_spans/*`, and `training/off_policy/evicted_samples`. The actor pool and rollout pool are both part of compute accounting.

**Time-to-quality:** use wall time from `start_epoch.txt` through the second consecutive full-set eval above the fixed target (initial target 0.80), including startup, eval and checkpoints. With this allocation, GPU-hours = elapsed seconds × 64 / 3600. Report failures to reach the target within 250 steps alongside full curves. V1's `timing_s/step` excludes validation in this source, so it is not directly comparable to the old V0 step timer; use end-to-end elapsed time for the primary comparison.

If trainer starvation dominates, investigate an 8/56 split; if the ready queue stays full and trainer computation dominates, investigate 32/32. Those are future recipe changes requiring corresponding config/preflight updates. Neither 64 GPUs nor the initial 16/48 allocation is guaranteed faster. Do not use the previous synchronous accuracy curve as proof this asynchronous candidate converges.

## Verification performed when creating this bundle

- Full Hydra composition against the pinned source, including explicit actor TIS config and optimizer/mixed-precision config.
- Shell/Python syntax checks; boxed scorer's 22 fixture/scope checks; reward-adapter synthetic interface checks.
- Source review of resource pools, replay-age rule, single-forward TIS dispatch, global token normalization and exact completion-length reward adapter.

No GPU training or deployment was performed while generating this bundle. Environment provisioning, distributed NCCL weight transfer and learning remain to be validated on the user's cluster by the provided smoke/full-run commands.
