# Gemma-4 E2B (`gemma-4-E2B-it`) GRPO (`rl-05` Parity) on GKE A4X (GB200)

Colocated `verl` GRPO (DAPO clip-higher) implementation, GKE A4X (`a4x-highgpu-4g-a4x-pool-7`) cluster manifests, scaling logs (`1n4g` to `16n64g`), and 4-step validation results for `google/gemma-4-E2B-it` on the `rl-05` parity workload.

---

## 1. Frozen Workload & Recipe (`run_gemma4_e2b_rl05_parity.sh`)

| Parameter | Value | Notes |
|---|---|---|
| Model | `google/gemma-4-E2B-it` | Mounted at `/workspace/meta-RL/models/gemma-4-E2B-it` |
| Dataset | `OpenMathInstruct-2` | `/workspace/meta-RL/data/openmathinstruct2/{train,val}.parquet` |
| Batch & Rollout | `train_batch_size=256`, `rollout.n=8` | `2,048` completions / step, `ppo_mini_batch_size=256` ($\mu=1$ update/step) |
| Sequence Lengths | `max_prompt_length=8192`, `max_response_length=8192` | `filter_overlong_prompts=False`, `truncation=error` |
| Algorithm & Loss | `adv_estimator=grpo`, `use_kl_loss=False` ($\beta=0$) | Clip-higher `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `entropy_coeff=0` |
| Optimizer | `lr=1e-6`, `grad_clip=1.0` | FSDP (`bfloat16`), gradient checkpointing enabled, `use_remove_padding=True` |
| Dynamic Batching | `ppo_max_token_len_per_gpu=16384` | Lowered from Qwen's `32768` due to Gemma-4's `262,144` vocab size (peak reserved HBM ~52 GB/GPU) |
| Rollout (vLLM) | `tp=1`, `gpu_memory_utilization=0.30` | `temperature=0.8`, `top_p=0.95`, `top_k=50`, `max_num_batched_tokens=32768`, `enable_chunked_prefill=True`, `enable_prefix_caching=True` |
| Trainer Attention | `attn_implementation=flex_attention` | With `[GEMMA4]` Triton tile-pinning for `head_dim=512` (see §2) |
| Reward Function | `maxtext_math_reward.py` (`compute_score`) | `math-verify==0.9.0` + `sympy` verification |

---

## 2. Gemma-4 System & Kernel Adaptations on GB200

1. **vLLM Attention Backend (`TRITON_ATTN`)**:
   - Gemma-4 E2B uses heterogeneous head dimensions (`256` for local sliding-window attention layers, `512` for global full-attention layers). vLLM automatically detects `head_dim=512` and selects `TRITON_ATTN` (`Forcing TRITON_ATTN backend`).
2. **Trainer `flex_attention` Triton Tile Pinning (`head_dim > 256`)**:
   - PyTorch Inductor's default `flex_attention` Triton kernel configuration overflows the GB200 shared memory limit (`227 KB` SMEM/CTA) when `head_dim=512`.
   - `_verl_patch_flex_kernel_options()` in `verl/models/transformers/monkey_patch.py` (and standalone helper `patch_verl_gemma4_flex_attn.py` for `/workspace/verl-pin` containers) pins the measured GB200 winning tile sizes for `head_dim > 256`:
     - Forward: `BLOCK_M=32, BLOCK_N=128, num_warps=8, num_stages=1`
     - Backward: `BLOCK_M1=32, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=32, num_warps=8, num_stages=1`
3. **Packing Gate (`gemma_pack_gate.py`)**:
   - Validates single-GPU forward+backward memory under `ppo_max_token_len_per_gpu=16384` with `262k` vocabulary logits before launching multi-node runs.

---

## 3. 4-Step Validation Run on `a4x-highgpu-4g-a4x-pool-7` (`4n16g`, `flex_attention`)

Validated on `a4x-baker` nodepool `a4x-highgpu-4g-a4x-pool-7` (`NNODES=4`, 16 GB200 GPUs) against the 20-step reference run (`logs/gemma4_e2b_4n16g_official_0902_0439.log`, steady-state median **76.2 s/step**).

### Validation vs. Reference Summary (`step_time_report.py`)

| Run | Log File | Steps | Step Median (s) | Step Mean (s) | Step p90 (s) | Rollout (s) | Train Total (s) | `update_actor` (s) | `old_log_prob` (s) | Other (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **20-Step Reference (`4n16g`)** | `logs/gemma4_e2b_4n16g_official_0902_0439.log` | 2–20 (`n=19`) | **76.2** | 76.5 | 79.6 | 41.4 (54%) | 28.9 | 21.8 | 7.1 | 6.4 |
| **4-Step Validation (`4n16g`)** | `logs/gemma4_e2b_4n16g_val4steps_1020_0945.log` | 2–4 (`n=3`) | **78.3** | 78.8 | 81.4 | 38.1 (49%) | 35.9 | 27.7 | 8.7 | 4.3 |

### Per-Step Breakdown (`logs/gemma4_e2b_4n16g_val4steps_1020_0945.log`)

| Step | Step Time (s) | Rollout `gen` (s) | Train Total (s) | `update_actor` (s) | `old_log_prob` (s) | Other / Weight Sync (s) | Mean Output Len (tok) | Mean Score | Peak HBM Reserved (GB) |
|---|---|---|---|---|---|---|---|---|---|
| **Step 1** (compile warmup) | 131.6 | 51.7 | 75.6 | 35.9 | 39.7 | 4.3 | 1,763 | 0.726 | 51.1 |
| **Step 2** | **75.9** | 36.9 | 34.7 | 26.0 | 8.7 | 4.3 | 1,892 | 0.692 | 52.2 |
| **Step 3** | **78.3** | 38.1 | 35.9 | 27.7 | 8.2 | 4.3 | 1,743 | 0.762 | 52.2 |
| **Step 4** | **82.2** | 38.7 | 39.1 | 30.3 | 8.8 | 4.4 | 2,049 | 0.679 | 52.2 |

---

## 4. Full Scaling Sweep (`1n4g` to `16n64g`, Steps 2–20 Steady State)

All runs use `train_batch_size=256`, `n=8` (`2,048` completions/step), `rollout_tp=1`, `ppo_max_token_len_per_gpu=16384`.

| Scale | GPUs ($W$) | Trainer Attn | Step Median (s) | Step Mean (s) | Step p90 (s) | Rollout (s) | Train Total (s) | `update_actor` (s) | `old_log_prob` (s) | Other (s) | Log File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `1n4g` | 4 | `flex_attention` | **172.7** | 172.5 | 178.9 | 72.4 (42%) | 95.9 | 73.1 | 22.8 | 4.4 | `logs/gemma4_e2b_1n4g_official_0902_0645.log` |
| `2n8g` | 8 | `flex_attention` | **108.8** | 109.7 | 115.6 | 51.1 (47%) | 51.2 | 38.7 | 12.1 | 6.2 | `logs/gemma4_e2b_2n8g_official_0902_0536.log` |
| `4n16g` | 16 | `flex_attention` | **76.2** | 76.5 | 79.6 | 41.4 (54%) | 28.9 | 21.8 | 7.1 | 6.4 | `logs/gemma4_e2b_4n16g_official_0902_0439.log` |
| `8n32g` | 32 | `flex_attention` | **63.2** | 64.0 | 69.2 | 37.6 (59%) | 19.0 | 14.5 | 4.6 | 6.6 | `logs/gemma4_e2b_8n32g_official_0902_0336.log` |
| `16n64g` | 64 | `flex_attention` | **57.8** | 58.6 | 61.9 | 36.3 (63%) | 15.3 | 12.1 | 3.1 | 6.7 | `logs/gemma4_e2b_16n64g_official_0902_0114.log` |
| `16n64g` | 64 | `sdpa` | **78.0** | 77.7 | 83.2 | 36.2 (46%) | 35.7 | 28.1 | 7.9 | 6.6 | `logs/gemma4_e2b_rl05_16n64g_official_sdpa_0902_0220.log` |

---

## 5. Directory Contents

| File | Description |
|---|---|
| `run_gemma4_e2b_rl05_parity.sh` | Launcher script with frozen `rl-05` hyperparameters, tokenizer/chat-template preflight, and `flex_attention` defaults |
| `maxtext_math_reward.py` | Math reward scorer (`compute_score`) matching MaxText `rl-05` grading semantics (`math-verify` + `sympy`) |
| `patch_verl_gemma4_flex_attn.py` | Idempotent patcher adding `_verl_patch_flex_kernel_options` (`head_dim > 256` Triton tiles) to `monkey_patch.py` |
| `gemma_pack_gate.py` | Single-GPU forward+backward memory gate for Gemma-4 (`262k` vocab) at `ppo_max_token_len_per_gpu=16384` |
| `step_time_report.py` | Parses `verl` driver logs and summarizes step-time median/mean/p90 and phase decomposition (`rollout`, `update_actor`, `old_log_prob`, `other`) |
| `verl-qwen3-raycluster.yaml` | 16-node (64-GPU) KubeRay `RayCluster` manifest targeting `a4x-highgpu-4g-a4x-pool-7` with MNNVL `ComputeDomain` and GCS Fuse mounts |
| `compute-domain-a4x.yaml` | `resource.nvidia.com/v1beta1` `ComputeDomain` (`numNodes: 16`) for Multi-Node NVLink (MNNVL) |
| `priorityclass-verl-high.yaml` | `PriorityClass` (`verl-high`) for the RayCluster pods |
| `logs/` | Official scaling logs (`1n4g`–`16n64g`, `flex_attention` & `sdpa`) and the 4-step validation log (`gemma4_e2b_4n16g_val4steps_1020_0945.log`) |

---

## 6. How to Deploy & Run

### Step 1: Deploy ComputeDomain and RayCluster on GKE (`a4x-baker`)
```bash
kubectl apply -f priorityclass-verl-high.yaml
kubectl apply -f compute-domain-a4x.yaml
kubectl apply -f verl-qwen3-raycluster.yaml
kubectl get raycluster verl-bench-xiaotongyang
```

### Step 2: Ensure `math-verify==0.9.0` and `[GEMMA4]` Flex-Attention Patch Across Workers
If running on the `verl-disagg-cu130-arm64` image (`/workspace/verl-pin`):
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

### Step 3: Launch 4-Step Validation (`4n16g`) or Full 20-Step Run
Inside the Ray head pod (`verl-bench-xiaotongyang-head-*`):
```bash
SCRIPT_DIR=/workspace/meta-RL/verl/examples/grpo_trainer/my_scripts_GKE_gemma4
LOG=/workspace/meta-RL/logs/gemma4_e2b_4n16g_val4steps_$(date +%m%d_%H%M).log

NNODES=4 TOTAL_STEPS=4 bash "$SCRIPT_DIR/run_gemma4_e2b_rl05_parity.sh" 2>&1 | tee "$LOG"

# Generate step-time summary (steady-state steps 2-4):
python3 "$SCRIPT_DIR/step_time_report.py" "$LOG" 2 4
```
