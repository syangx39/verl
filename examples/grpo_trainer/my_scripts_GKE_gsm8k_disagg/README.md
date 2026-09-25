# GPU disaggregated-async GSM8K reference — recipe `gsm8k_2k_async1` (operator notes)

The GB200 baseline requested by the TPU TorchTitan team: Qwen3-0.6B (post-trained), GSM8K, response cap 2,048 with the overlong penalty,
GRPO with single-forward REINFORCE and a truncated importance weight (the colocated `gsm8k_2k_v1` algorithm), run **disaggregated and
asynchronously** on 64 GB200: **32 trainer GPUs + 32 rollout GPUs (profile B)**, sampler one batch ahead, weights pushed after every update.
Three seeds × 250 steps are done and packaged. Semantics, gates, reference numbers and the comparison rule are in
`TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md`; the TPU-side procedure is `Wenjun_TPU_Parity_Guide_gsm8k_2k_async1.md`.

## Results (three seeds, `band/summary.json` in the package)

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| GSM8K test acc, step 0 → 250 (greedy, 1,319) | 0.7066 → **0.8317** | 0.7066 → **0.8271** | 0.7028 → **0.8234** |
| confirming eval ≥ 0.80 (second of two consecutive) | step 80 | step 100 | step 100 |
| steady step time, median / p90 (steps 20–250 excl. eval/ckpt) | 7.16 / 10.07 s | 7.34 / 9.53 s | 7.22 / 10.25 s |
| end to end, 250 steps incl. startup, 14 evals, 5 checkpoints | 59.4 min | 59.2 min | 59.3 min |
| observed worst consumed staleness / max span / dropped groups (config allows staleness ≤ 2) | 1 / 2 / 11 | 1 / 2 / 8 | 1 / 2 / 8 |

Final mean 0.8274 [0.8234, 0.8317]; band width across the 14 checkpoints median 2.0 pp, max 2.9 pp. Colocated `gsm8k_2k_v1` on the same
64 GB200 (context): 14.1 s/step, 71.0 / 69.9 min end to end, final 0.8226 / 0.8180. Figures: `band/` in the package (3-seed band,
disagg-vs-colocated curves, step-time decomposition, per-seed six-panel diagnostics).

## Source and environment (frozen)

- verl: upstream `verl-project/verl` @ `ace775e87d8765bcdd114aac734ab71da5367a0f` (`VERL_PIN.txt`; hybrid_engine=False support in
  `separate_async`, standalone-rollout memory budget). No fork, no patches.
- Container: `IMAGE_REF.txt` — derived by `Dockerfile` / `build_and_push.sh` from `verlai/verl:uv-cu130-arm64` (the base ships a prefetched uv
  cache, not a venv): the pinned commit is cloned to `/workspace/verl-pin` and its own `uv.lock` installed into `/workspace/verl-pin/.venv`
  (Python 3.12, Torch 2.13.0+cu130, vLLM 0.29.0, Transformers 5.12.1, TransferQueue 0.1.10, FlashAttention 2.8.3), verified by
  `verify_venv_lock.py` at build time (`uv export` for the enabled extras, marker-aware; imports flash_attn and the V1 async trainer).
  Build on an arm64 machine (`bash build_and_push.sh`), then set the digest in the RayCluster manifest (4 places: `image:` ×2, `DISAGG_IMAGE` ×2).
- Cluster: `examples/grpo_trainer/my_scripts_GKE/verl-qwen3-raycluster.yaml` — venv on PATH, `PYTHONPATH=/workspace/verl-pin:<recipe dir>`,
  `DISAGG_IMAGE_MODE=1` (launcher verifies the venv against `uv.lock` instead of a prepare_env manifest). `prepare_env.py` is only for a
  venv-on-nodes setup without the image and is not used for the reference runs.

## Files

| file | role |
|---|---|
| `recipe_gpu_disagg.yaml` | the recipe (Tier 1 + Tier 2 + the chosen profile B knobs: `trainer.nnodes 8`, `rollout.nnodes 8`, `use_dynamic_bsz true`, `ppo_max_token_len_per_gpu 32768`, `dataloader_num_workers 0`, `VLLM_NO_USAGE_STATS`/`DO_NOT_TRACK` in the Ray env) |
| `run_gpu_disagg.sh` | launcher: venv check, pin check, preflight, `[recipe]` line from the resolved config, run dir with command/config/logs; passes every `${oc.env:…}` of `ray_kwargs` as literal overrides (upstream `main_ppo` does not resolve them) |
| `preflight.py` | 60+ frozen fields, data checks, 16-node CUDA probe, profile check (baseline 16/48 fixed micro-batch, A 16/48 dyn, B 32/32 dyn — B is the reference) |
| `boxed_math_reward.py`, `boxed_reward_v1.py` | scorer (shared with the Meta round) and the V1 reward-manager adapter passing the true response length |
| `check_smoke.py` | post-run checks: LR sequence, TIS stats, staleness/spans, evictions (`--max-worst-lag 1`, `--require-no-drops`) |
| `plot_phase0.py`, `plot_step_time.py` | six-panel diagnostics (V1 dumps: grouped by prompt uuid, accuracy not inferred from the penalized score) and step-time / wall-clock figures |
| `package_gsm8k_2k_async1.sh` | assembles and self-verifies the handoff package |
| `Dockerfile`, `build_and_push.sh`, `verify_venv_lock.py`, `VERL_PIN.txt`, `IMAGE_REF.txt` | image build and provenance |

## Running

```bash
source /workspace/setup_env.sh                       # RECIPE_DIR, VERL_REPO=/workspace/verl-pin, DISAGG_PYTHON, DISAGG_IMAGE, LOG_DIR, CKPT_DIR, MODEL_PATH, DATA_DIR
PREFLIGHT_ONLY=1 SEED=1 EXPERIMENT_NAME=preflight_$(date -u +%Y%m%d_%H%M%S) bash $RECIPE_DIR/run_gpu_disagg.sh      # [preflight] profile B … OK; no training launched
# 3-step smoke, then a 20-step check with eval/ckpt on:
SEED=1 TOTAL_STEPS=3  TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=false bash $RECIPE_DIR/run_gpu_disagg.sh
SEED=1 TOTAL_STEPS=20 TEST_FREQ=20 SAVE_FREQ=20 VAL_BEFORE_TRAIN=true  bash $RECIPE_DIR/run_gpu_disagg.sh
$DISAGG_PYTHON $RECIPE_DIR/check_smoke.py $(cat $LOG_DIR/latest_seed1.txt) --steps 20 --max-worst-lag 1 --require-no-drops
# reference run (defaults: 250 steps, eval every 20 + 250, checkpoint every 50):
SEED=1 nohup bash $RECIPE_DIR/run_gpu_disagg.sh > $LOG_DIR/seed1.driver.log 2>&1 &
```

All profile-B settings are in the YAML: do **not** pass `+ray_kwargs…VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` or the dynamic-batching /
topology overrides on the command line any more (Hydra rejects `+key` for keys that already exist; the YAML already holds them). Other
profiles for performance experiments only: `trainer.nnodes=4 actor_rollout_ref.rollout.nnodes=12 actor_rollout_ref.actor.use_dynamic_bsz=false` (baseline).

Expected per step in the driver log: `training/off_policy/trajectory_staleness_worst/max` 0 at step 1 then 1; `trajectory_spans/max` ≤ 2;
`actor/rollout_corr/rollout_is_mean` ≈ 1.000, `rollout_is_eff_sample_size` ≈ 0.9986, `k3_kl` ≈ 7e-4 (up to ≈ 1.6e-3 during steps 15–40);
`actor/lr` is logged after `scheduler.step()` (2e-7 at step 1 means update 1 used 0). `evicted_samples` appears only when a group older
than 2 policy versions is dropped (≈ 10 per 250 steps).

## Topology / batching selection (20-step trials, steps 11–19, median / p90 step time)

| profile | trainer / rollout GPUs | batching | step | update_actor | trainer waiting for sampler |
|---|---|---|---|---|---|
| baseline | 16 / 48 | fixed micro-batch 8 | 13.0 / 13.5 s | 9.7 / 10.5 s | 0.09 / 0.09 s |
| A | 16 / 48 | dynamic 32,768 tok/GPU | 6.8 / 12.8 s | 3.6 / 8.9 s | 0.08 / 2.1 s |
| **B (reference)** | **32 / 32** | dynamic 32,768 tok/GPU | **7.0 / 8.8 s** | 2.9 / 5.0 s | 0.07 / 2.1 s |

Dynamic batching removed the trainer bottleneck (MFU ≈ 2 % at fixed micro-batch 8); 32/32 has the same median as A with a much tighter tail.

## Packaging

```bash
nohup bash $RECIPE_DIR/package_gsm8k_2k_async1.sh > $LOG_DIR/package_2k_async1.log 2>&1 &     # ~20 min; SKIP_DUMPS=1 / SKIP_CKPT=1 for reruns
```

Output `/workspace/meta-RL/handoff/gsm8k_2k_async1/` → `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_2k_async1/`, verified by
`sha256sum -c --quiet PACKAGE_MANIFEST.sha256`; `README.md` inside the package lists every directory.

## Known gaps

No per-parameter gradient fixture for the V1 trainer (gate 5 is a loss-contract check); batch membership is completion-order dependent
(step manifests are references); the V1 rollout dumps carry the penalized score only (no `acc` field); evaluation on the 32-GPU sampler pool
costs ≈ 70 s per pass (≈ 16 of the 59 minutes).