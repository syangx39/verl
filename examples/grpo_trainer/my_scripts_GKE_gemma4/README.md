# Gemma-4 E2B (`gemma-4-E2B-it`) GRPO (`rl-05` parity) on GKE A4X (GB200)

Colocated `verl` GRPO (DAPO clip-higher) launcher, GKE A4X (`a4x-highgpu-4g-a4x-pool-7`) manifests, the 20-step scaling logs (`1n4g`–`16n64g`), and a 4-step smoke of the committed launcher for `google/gemma-4-E2B-it` on the `rl-05` workload.

> [!IMPORTANT]
> **Status: smoke passed; numerical validation and accuracy gate pending.**
> - Launch / training-flow smoke: **complete** (§3). Four steps with non-zero updates, driver `exit_code=0`.
> - Numerical correctness: **pending**. Rollout and trainer log-probs disagree (`k3_kl` 0.85–0.89 even with top-k/top-p off), and the packing gate does not cover the trainer's `flex_attention` path (§2.3, §3.4).
> - Accuracy gate: **pending** (§4). Eval is disabled in the smoke, so it says nothing about convergence or accuracy.

---

## 1. Frozen workload & recipe (`run_gemma4_e2b_rl05_parity.sh`)

| Parameter | Value | Notes |
|---|---|---|
| Model | `google/gemma-4-E2B-it` | `GEMMA_MODEL_PATH` (default `/workspace/meta-RL/models/gemma-4-E2B-it`); deliberately not `MODEL_PATH`, which `/workspace/setup_env.sh` sets to Qwen |
| Dataset | `OpenMathInstruct-2` | `GEMMA_DATA_DIR` (default `/workspace/meta-RL/data/openmathinstruct2`; `GEMMA_TRAIN_FILE` / `GEMMA_VAL_FILE` for single files); deliberately not `DATA_DIR`, which `setup_env.sh` sets to `gsm8k_boxed`. 950,000 train / 50,000 val rows |
| Batch & rollout | `train_batch_size=256`, `rollout.n=8` | 2,048 completions per step, `ppo_mini_batch_size=256` ($\mu=1$ update per step) |
| Sequence lengths | `max_prompt_length=8192`, `max_response_length=8192` | `filter_overlong_prompts=False`, `truncation=error` |
| Algorithm & loss | `adv_estimator=grpo`, `use_kl_loss=False` ($\beta=0$) | Clip-higher `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `entropy_coeff=0` |
| Optimizer | `lr=1e-6`, `grad_clip=1.0` | FSDP (`bfloat16`), gradient checkpointing, `use_remove_padding=True` |
| Dynamic batching | `ppo_max_token_len_per_gpu=16384` | Lowered from Qwen's 32768 because of Gemma-4's 262,144-token vocab (peak reserved HBM 51–53 GB/GPU in the smoke) |
| Rollout (vLLM) | `tp=1`, `gpu_memory_utilization=0.30` | `temperature=0.8`, `top_p=0.95`, `top_k=50`, `max_num_batched_tokens=32768`, chunked prefill, prefix caching |
| Trainer attention | `attn_implementation=flex_attention` (`ATTN_IMPL`) | With the `[GEMMA4]` Triton tile pinning for `head_dim=512` (§2) |
| Reward | `maxtext_math_reward.py` (`compute_score`) | `math-verify==0.9.0` + `sympy` |
| Eval | **disabled** | `trainer.test_freq=-1`, `trainer.val_before_train=False` (§4) |

---

## 2. Gemma-4 system & kernel notes on GB200

1. **vLLM attention**: Gemma-4 E2B mixes `head_dim=256` (sliding-window layers) and `head_dim=512` (global layers). The launcher does not set a vLLM attention backend. In this image, vLLM 0.29.0 used FlashInfer (smoke log: `Using TRTLLM prefill attention (auto-detected)`; TRT-LLM-gen cubins for both `H256` and `H512` are loaded).
2. **Trainer `flex_attention` tile pinning (`head_dim > 256`)**: Inductor's default `flex_attention` tiles overflow the GB200 227 KB SMEM/CTA limit at `head_dim=512`. `_verl_patch_flex_kernel_options()` (in `verl/models/transformers/monkey_patch.py`; `patch_verl_gemma4_flex_attn.py` applies it to a `/workspace/verl-pin` checkout) pins the measured tiles:
   - Forward: `BLOCK_M=32, BLOCK_N=128, num_warps=8, num_stages=1`
   - Backward: `BLOCK_M1=32, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=32, num_warps=8, num_stages=1`
3. **Packed-sequence isolation gate (`gemma_pack_gate.py`)**: single GPU, short sequences, `torch.no_grad()`, no memory measurement. It checks that HF's packed-sequence masking (restarted `position_ids`, `attention_mask=None`) keeps packed sequences from attending to each other by comparing `[A; B]` with `[A'; B]` (`A'` = token-shuffled `A`) at `96+80` tokens (inside the sliding window) and `640+560` tokens (crossing it). It passes only if `max |logits_B(A') - logits_B(A)| < 1e-3` and exits `1` otherwise.
   - Verified on GB200 with `--attn sdpa` (current file, sha256 `6855d7b7…`): LEAKAGE max `0.0000` in both rounds, exit `0`. A PASS only reports that no leakage was seen under these conditions; it does not by itself explain training metrics such as `grad_norm`.
   - **Not covered: `--attn flex_attention`, which the trainer uses.** On torch 2.13 the 80-token reference forward is routed to Inductor's flex-decoding kernel, which has no valid configs here (`NoValidChoicesError`), so the gate exits `1` before reaching a verdict. The gate also does not import verl's `head_dim > 256` tile patch. Packing isolation under `flex_attention` is therefore not verified by this gate.

---

## 3. 4-step smoke of the committed launcher (`4n16g`, `flex_attention`) — complete

### 3.1 What ran

- Launcher: `run_gemma4_e2b_rl05_parity.sh` at sha256 `c5f5a0f86c56f6991de1f656c19538c50b1e6a3f622062156363bee7a0b850d6`. The file in this directory (sha256 `8927660b03a1f99721287dc347887d9b16cfbd8e8bc943c9b4f13817be0ab318`) differs from it only in comment lines: the stale vLLM attention-backend notes were removed. The non-comment lines are identical; `grep -v '^[[:space:]]*#' run_gemma4_e2b_rl05_parity.sh | sha256sum` gives `5dc43e0787e4b928e25ebb174b637b72de0e5d5c77f2d2d9b52ce4fd87cdef76` for both.
- Command, in the Ray head pod, right after `source /workspace/setup_env.sh` (which exports `DATA_DIR=.../gsm8k_boxed` and `MODEL_PATH=.../Qwen3-0.6B`):
  `NNODES=4 TOTAL_STEPS=4 bash run_gemma4_e2b_rl05_parity.sh` — no extra overrides.
- Log checks (`logs/gemma4_e2b_4n16g_smoke4steps_1020_2258.log`):
  - data and model resolved to `openmathinstruct2/{train,val}.parquet` and `gemma-4-E2B-it` despite the Qwen exports;
  - `ray init kwargs` shows `'REWARD_DUMP_EVERY': '256', 'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1'` as strings;
  - `actor/lr = 1e-6` and non-zero `actor/grad_norm` at every step;
  - the log ends with `[driver] main_ppo finished with exit_code=0` (followed by the `set -x` trace `+ exit 0`).

### 3.2 Step time (`step_time_report.py`)

| Run | Log | Steps | Median (s) | Mean (s) | p90 (s) | Max (s) | Rollout (s) | Train (s) | `update_actor` (s) | `old_log_prob` (s) | Other (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-step reference (`4n16g`) | `logs/gemma4_e2b_4n16g_official_0902_0439.log` | 2–20 (n=19) | **76.2** | 76.5 | 79.6 | 79.8 | 41.4 (54%) | 28.9 | 21.8 | 7.1 | 6.4 |
| **Smoke, committed launcher** | `logs/gemma4_e2b_4n16g_smoke4steps_1020_2258.log` | 2–4 (n=3) | **77.7** | 77.9 | 80.6 | 81.3 | 38.2 (49%) | 33.6 | 25.4 | 8.3 | 4.3 |
| Smoke, incl. compile step | same | 1–4 (n=4) | 79.5 | 90.5 | 114.0 | 128.0 | 39.0 (49%) | 34.6 | 26.2 | 8.5 | 4.3 |
| Earlier run (pre-fix launcher + manual CLI overrides; superseded) | `logs/gemma4_e2b_4n16g_val4steps_1020_0945.log` | 2–4 (n=3) | 78.3 | 78.8 | 81.4 | 82.2 | 38.1 (49%) | 35.9 | 27.7 | 8.7 | 4.3 |

Per step (smoke):

| Step | Step (s) | Rollout (s) | Train (s) | `update_actor` (s) | `old_log_prob` (s) | Other (s) | Mean resp. len (tok) | Mean score | `grad_norm` | `lr` | Peak reserved HBM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (compile) | 128.0 | 50.2 | 73.6 | 37.0 | 36.6 | 4.2 | 1,763 | 0.727 | 4.65625 | 1e-6 | 51.2 |
| 2 | **74.8** | 37.0 | 33.5 | 25.3 | 8.3 | 4.3 | 1,878 | 0.695 | 6.4375 | 1e-6 | 51.6 |
| 3 | **77.7** | 39.8 | 33.6 | 25.4 | 8.2 | 4.3 | 1,771 | 0.765 | 9.0625 | 1e-6 | 51.9 |
| 4 | **81.3** | 38.2 | 35.7 | 27.0 | 8.7 | 7.4 | 2,084 | 0.672 | 7.3125 | 1e-6 | 52.9 |

"Mean score" is the sampled training reward (T=0.8) on that step's 256 prompts, not held-out accuracy.

### 3.3 Known exit-phase issue: `DataLoader worker ... killed`

This is recorded as an exit-phase issue: it does not change the driver exit code. `RuntimeError: DataLoader worker (pid 6527) is killed by signal: Killed.` appears after `Training Progress: 100% 4/4`. The traceback is raised by PyTorch's DataLoader SIGCHLD handler while the TaskRunner sits in Ray's `run_task_loop()`, with no training frames on the stack, i.e. after `TaskRunner.run()` returned. That is consistent with, but not proof of, Ray SIGKILL-ing the actor's child DataLoader workers at teardown. All four step lines were logged and the driver exited `0`. The same message is present at the end of all six 09-02 20-step logs.

### 3.4 Rollout vs trainer log-probs (`rollout_corr/k3_kl` ≈ 1.3) — pending

**Conventions (code, verl pin `ace775e8`, vLLM 0.29.0):**
- Rollout log-probs: `actor_rollout_ref.rollout.logprobs_mode=processed_logprobs` (verl default in `verl/trainer/config/rollout/rollout.yaml`, passed to the vLLM engine in `vllm_async_server.py`). vLLM applies temperature, then `apply_top_k_top_p` masking, then `log_softmax` (`v1/sample/ops/topk_topp_sampler.py`), i.e. log-probs renormalized over the kept top-k/top-p set.
- Trainer `old_log_prob`: `log_softmax(logits / T)` over the full 262,144-token vocabulary (`verl/workers/engine/fsdp/transformer_impl.py`).

**Can the convention difference explain the gap? No, by a bound on the logged metrics.** If vLLM and the trainer had the same temperature-scaled distribution, then for each sampled token, rollout − trainer log-prob $= -\log Z_t$, where $Z_t$ is the trainer's mass on the kept set. The kept set always contains the argmax, so $Z_t \ge p_{\max,t} \ge e^{-H_t}$, giving $-\log Z_t \le H_t$. Token-averaging over the same response tokens gives `rollout_corr/kl` ≤ `actor/entropy`. The smoke violates this by about 4×:

| Smoke step | `rollout_corr/kl` | `actor/entropy` | `rollout_corr/k3_kl` | `training_log_ppl` / `rollout_log_ppl` | Pearson(p_train, p_rollout) |
|---|---|---|---|---|---|
| 1 | 1.549 | 0.398 | 1.306 | 1.336 / 0.115 | 0.409 |
| 2 | 1.596 | 0.408 | 1.349 | 1.393 / 0.125 | 0.416 |
| 3 | 1.552 | 0.404 | 1.308 | 1.357 / 0.117 | 0.410 |
| 4 | 1.648 | 0.399 | 1.398 | 1.433 / 0.126 | 0.407 |

**A/B with truncation off** (same launcher plus `actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.top_p=1.0`, 2 steps; with no top-k/top-p, `processed_logprobs` uses exactly the trainer's convention; `logs/gemma4_e2b_4n16g_diag_notrunc_1021_0610.log`). The resolved config shows `top_k: -1`, `top_p: 1.0`:

| Run | Step | `rollout_corr/kl` | `actor/entropy` | `rollout_corr/k3_kl` | `training_log_ppl` / `rollout_log_ppl` | Pearson(p_train, p_rollout) |
|---|---|---|---|---|---|---|
| Smoke (top-k 50, top-p 0.95) | 1 | 1.549 | 0.398 | 1.306 | 1.336 / 0.115 | 0.409 |
| Smoke (top-k 50, top-p 0.95) | 2 | 1.596 | 0.408 | 1.349 | 1.393 / 0.125 | 0.416 |
| A/B, truncation off | 1 | 0.972 | 0.413 | 0.846 | 1.018 / 0.146 | 0.482 |
| A/B, truncation off | 2 | 1.018 | 0.437 | 0.886 | 1.057 / 0.157 | 0.485 |

**Reading.** With truncation off the two log-probs use the same convention, so a matched rollout/trainer pair should give `kl` and `k3_kl` near zero, at bf16-noise level. For scale, the Qwen3-0.6B recipe on the same verl pin reports `k3_kl` ≈ 7e-4 (`my_scripts_GKE_gsm8k_disagg/README.md`). Here `k3_kl` stays at 0.85–0.89, `kl` at 0.97–1.02, and Pearson at 0.48. The probability convention is therefore not the main cause: there is a real Gemma-4 rollout-vs-trainer mismatch, not yet localized. Candidates, none tested here:
- the trainer's packed `flex_attention` path (the gate covers `sdpa` only, §2.3);
- vLLM-vs-HF numerics for Gemma-4;
- rollout weight sync;
- token/log-prob alignment.

Next step (separate from this smoke): keep the truncation-off sampling and switch `ATTN_IMPL=sdpa` for 2 steps. This separates the trainer's `flex_attention` path, which the gate does not cover, from the rest. If the gap persists, follow up with an offline single-GPU HF-vs-vLLM log-prob comparison on identical token sequences.

The 09-02 runs used an older verl checkout that did not log these metrics, so they cannot show whether the gap existed then.

---

## 4. Accuracy gate — pending (not evaluated here)

- **Gate (customer, Meta)**: TPU accuracy ≥ GPU accuracy − 1 percentage point. Example: if the GPU reference is 55%, the TPU run must reach ≥ 54%. No GPU reference accuracy has been measured in this directory yet.
- **This smoke cannot be used for it**: eval is disabled, and 4 steps × 256 prompts cover 1,024 of the 950,000 training prompts. The gate needs a separate eval-enabled GPU run. Extra Hydra overrides after the launcher are applied last (verified in §3.4: `top_k=-1` replaced the launcher's `top_k=50`), so `trainer.test_freq=<N> trainer.val_before_train=True` turns eval on without editing the frozen block.
- **Fix on both sides before reading a 1-pp gate**: the held-out set and its size, eval decoding settings, prompt rendering, and number of seeds. At 55% accuracy the binomial standard error is 1.57 pp at N=1,000, 0.70 pp at N=5,000, and 0.22 pp at N=50,000 (the full `val.parquet`). A GPU−TPU difference between two independent evaluations has √2× that error, so N≈1,000 cannot resolve 1 pp.
- **Prompt rendering**: the OMI2 parquet's user content already contains `<start_of_turn>user … <end_of_turn>\n<start_of_turn>model`, and verl wraps it again in Gemma-4's chat template (`<|turn>user … <turn|>`; see the `PREFLIGHT` sample in the smoke log). The TPU side has to render prompts identically for the comparison to be like-for-like.
- **Resolve §3.4 first**: whatever its cause, a rollout/trainer gap this large means the trainer is updating on samples from a measurably different policy.

---

## 5. Scaling sweep (`1n4g` to `16n64g`, steps 2–20)

Generated by `python3 step_time_report.py <log> 2 20` (exact output in [`step_time_summary.txt`](step_time_summary.txt)). All runs: `train_batch_size=256`, `n=8`, `rollout_tp=1`, `ppo_max_token_len_per_gpu=16384`, eval off.

| Scale | GPUs ($W$) | Trainer attn | Median (s) | Mean (s) | p90 (s) | Max (s) | Rollout (s) | Train (s) | `update_actor` (s) | `old_log_prob` (s) | Other (s) | Log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `1n4g` | 4 | `flex_attention` | **172.7** | 172.4 | 180.3 | 183.4 | 72.4 (42%) | 95.9 | 77.2 | 19.4 | 5.7 | `logs/gemma4_e2b_1n4g_official_0902_0645.log` |
| `2n8g` | 8 | `flex_attention` | **108.8** | 112.7 | 116.6 | 164.0 | 51.1 (47%) | 51.2 | 40.2 | 11.2 | 6.3 | `logs/gemma4_e2b_2n8g_official_0902_0536.log` |
| `4n16g` | 16 | `flex_attention` | **76.2** | 76.5 | 79.6 | 79.8 | 41.4 (54%) | 28.9 | 21.8 | 7.1 | 6.4 | `logs/gemma4_e2b_4n16g_official_0902_0439.log` |
| `8n32g` | 32 | `flex_attention` | **63.2** | 63.0 | 63.9 | 66.2 | 37.6 (60%) | 19.0 | 13.6 | 5.4 | 6.3 | `logs/gemma4_e2b_8n32g_official_0902_0336.log` |
| `16n64g` | 64 | `flex_attention` | **57.8** | 57.7 | 58.6 | 59.1 | 36.3 (63%) | 15.3 | 10.3 | 4.9 | 6.1 | `logs/gemma4_e2b_16n64g_official_0902_0114.log` |
| `16n64g` | 64 | `sdpa` | **78.0** | 78.2 | 80.6 | 82.0 | 36.2 (46%) | 35.7 | 29.0 | 6.9 | 5.9 | `logs/gemma4_e2b_rl05_16n64g_official_sdpa_0902_0220.log` |

Component columns are per-component medians, so they do not sum to the step median.

---

## 6. Directory contents

| File | Description |
|---|---|
| `run_gemma4_e2b_rl05_parity.sh` | Launcher: frozen `rl-05` block, `GEMMA_DATA_DIR` / `GEMMA_MODEL_PATH` isolation, chat-template preflight, string-quoted Ray env vars, prints `[driver] main_ppo finished with exit_code=N` and exits with it |
| `maxtext_math_reward.py` | Math reward (`compute_score`) matching MaxText `rl-05` grading (`math-verify` + `sympy`) |
| `patch_verl_gemma4_flex_attn.py` | Idempotent patcher adding `_verl_patch_flex_kernel_options` (`head_dim > 256` tiles) to `monkey_patch.py` |
| `gemma_pack_gate.py` | Packed-sequence isolation gate (§2.3); exits `1` on FAIL or error |
| `step_time_report.py` | Step-time median/mean/p90/max and phase breakdown from a driver log |
| `step_time_summary.txt` | Exact `step_time_report.py` output for the smoke and the six scaling logs |
| `verl-qwen3-raycluster.yaml` | 16-node (64-GPU) KubeRay `RayCluster` on `a4x-highgpu-4g-a4x-pool-7` with MNNVL `ComputeDomain` and GCS Fuse mounts |
| `compute-domain-a4x.yaml` | `ComputeDomain` (`numNodes: 16`) for MNNVL |
| `priorityclass-verl-high.yaml` | `PriorityClass` `verl-high` |
| `logs/` | Scaling logs, the smoke log, the truncation-off A/B log, and the superseded pre-fix run |

---

## 7. How to deploy & run

### Deploy (GKE `a4x-baker`)
```bash
kubectl apply -f priorityclass-verl-high.yaml
kubectl apply -f compute-domain-a4x.yaml
kubectl apply -f verl-qwen3-raycluster.yaml
kubectl get raycluster verl-bench-xiaotongyang
```

### Prepare workers (`verl-disagg-cu130-arm64` image, `/workspace/verl-pin`)
```bash
for pod in $(kubectl get pods -l ray.io/cluster=verl-bench-xiaotongyang -o jsonpath='{.items[*].metadata.name}'); do
  container=$(kubectl get pod $pod -o jsonpath='{.spec.containers[0].name}')
  kubectl exec $pod -c $container -- bash -c "
    uv pip install --python /workspace/verl-pin/.venv/bin/python --quiet math-verify==0.9.0
    python3 /workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4/patch_verl_gemma4_flex_attn.py /workspace/verl-pin/verl/models/transformers/monkey_patch.py
  " &
done
wait
```

### Optional packing gate (GPU worker pod)
```bash
python3 /workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4/gemma_pack_gate.py \
  --model /workspace/meta-RL/models/gemma-4-E2B-it --attn sdpa
```

### Smoke (`4n16g`) or 20-step run (Ray head pod)
```bash
source /workspace/setup_env.sh
SCRIPT_DIR=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4
LOG=/workspace/meta-RL/logs/gemma4_e2b_4n16g_smoke4steps_$(date +%m%d_%H%M).log
NNODES=4 TOTAL_STEPS=4 bash "$SCRIPT_DIR/run_gemma4_e2b_rl05_parity.sh" 2>&1 | tee "$LOG"
grep -a "\[driver\]" "$LOG"                     # [driver] main_ppo finished with exit_code=0
python3 "$SCRIPT_DIR/step_time_report.py" "$LOG" 2 4
```
