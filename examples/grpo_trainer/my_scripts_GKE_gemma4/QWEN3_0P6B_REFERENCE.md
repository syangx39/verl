# Qwen3-0.6B learning references for Gemma-4 E2B

## Gist

Four historical **post-trained `Qwen/Qwen3-0.6B`** recipes learned on 64 GB200 GPUs. This ledger records their fixed-endpoint accuracy, learning curves, recipe semantics, timing definitions and source artifacts so Gemma-4 E2B optimization has a concrete reference.

The successful GSM8K runs are **not `Qwen3-0.6B-Base`**, despite `base` remaining in old script and experiment names. The old GSM README and bare launcher defaults describe earlier experiments; use the successful-run rulebooks and archived resolved configurations below. Likewise, OMI2 `stab_kl0` is distinct from the later 20-step performance benchmark.

This is a historical comparison, not a requirement that Gemma gain the same number of percentage points. Models, initial competence, held-out sets, reward, sampling and training budgets differ. No training configuration is changed here.

## 1. Fixed-endpoint results

Accuracy is greedy held-out answer correctness, **not training reward**. Values are percentages; gains are percentage points (pp). Means give every seed equal weight and use the declared last checkpoint, never the best checkpoint.

| Model | Training recipe | Updates × seeds | Eval set / interval | Initial mean | Final mean | Gain |
|---|---|---:|---|---:|---:|---:|
| Qwen3-0.6B post-trained | GSM8K colocated, 2K + training length penalty | 250 × 2 | Full GSM8K 1,319 / every 20 | 70.66% | 82.03% | +11.37 pp |
| Qwen3-0.6B post-trained | GSM8K colocated, 8K, no length penalty | 250 × 3 | Full GSM8K 1,319 / every 20 | 74.78% | 80.57% | ≈+5.8 pp |
| Qwen3-0.6B post-trained | GSM8K disaggregated async, 32 trainer / 32 rollout, 2K + training length penalty | 250 × 3 | Full GSM8K 1,319 / every 20 | 70.53% | 82.74% | +12.21 pp |
| Qwen3-0.6B post-trained | OMI2 colocated, `stab_kl0`, 8K + length penalty | 300 × 3 | Internal OMI2 held-out 1,000 / every 10 | ≈36.9% | 55.0%† | ≈+18.1 pp† |

**Evidence and precision.** GSM values come from the committed [8K rulebook](../my_scripts_GKE_gsm8k/TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md) and [async rulebook](../my_scripts_GKE_gsm8k_disagg/TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md). Their published fractions are rounded to four decimals. For 8K, those rounded endpoints imply +5.78 pp; the original TensorBoard deltas reported in the run transcript imply +5.79 pp. We use ≈+5.8 pp rather than mix precision.

† OMI2 final per-seed accuracies and the final GSM8K transfer result below are the owner's historical handoff summary. The committed [Tianyu guide](../my_scripts_GKE/Tianyu_TPU_Parity_Guide.md) verifies the recipe, initial scale, eval schedule and archive locations, but the raw 31-point GPU band JSON was not re-read for this documentation PR. These endpoints are recorded, not presented as a fresh recomputation. The approximately 49.0% OMI2 result at step 100 is also recorded in [UPDATE_PRECISION_RESULTS.md §4.5](UPDATE_PRECISION_RESULTS.md#45-reading).

### Per-seed endpoints

| Recipe | Seed | Step 0 | Fixed final step | Final accuracy |
|---|---:|---:|---:|---:|
| GSM8K colocated 2K + penalty | 1 | 70.36% | 250 | 82.26% |
| GSM8K colocated 2K + penalty | 2 | 70.96% | 250 | 81.80% |
| GSM8K colocated 8K, no penalty | 1 | 74.53% | 250 | 82.26% |
| GSM8K colocated 8K, no penalty | 2 | 74.00% | 250 | 79.61% |
| GSM8K colocated 8K, no penalty | 3 | 75.82% | 250 | 79.83% |
| GSM8K disaggregated 2K + penalty | 1 | 70.66% | 250 | 83.17% |
| GSM8K disaggregated 2K + penalty | 2 | 70.66% | 250 | 82.71% |
| GSM8K disaggregated 2K + penalty | 3 | 70.28% | 250 | 82.34% |
| OMI2 `stab_kl0` | 1 | Not transcribed individually | 300 | 55.4%† |
| OMI2 `stab_kl0` | 2 | Not transcribed individually | 300 | 55.2%† |
| OMI2 `stab_kl0` | 3 | Not transcribed individually | 300 | 54.4%† |

Keep the stronger GSM8K 8K seed 1 in the reference. A two/three-seed min–max envelope describes observed variation; it is not a confidence interval or an automatic parity pass/fail rule.

## 2. Learning curves and eval protocols

**GSM8K:** train on 7,473 questions; evaluate all 1,319 test questions at steps `0, 20, …, 240, 250` (14 evaluations). Eval uses greedy decoding, one completion per question, with the recipe's response cap. The 2K and 8K results therefore differ in **eval budget as well as training cap and penalty**; this is not a single-factor cap ablation.

**OMI2:** train on the question-level split from `OpenMathInstruct-2/train_1M`; evaluate at `0, 10, …, 300` (31 evaluations). Each evaluation covers the internal OMI2 held-out 1,000 plus GSM8K's full 1,319, reported separately, greedy with an 8K response cap. OMI2 training reaches approximately **36.9% → 49.0% at step 100 → 55.0% at step 300†**. Its concurrent GSM8K result is approximately **71.9% → 79.7%†**: this is transfer from OMI2 training, not another GSM8K-trained run.

The [OMI2 split builder](../my_scripts_GKE/build_eval_data_v3.py) excludes all training rows belonging to a pool of 2,000 held-out question identities and evaluates 1,000 unique questions from that pool. This is an internal split, not an official OMI2 test set. Exact row counts and hashes belong to the archived manifests. Do not substitute the later grain-seed-42 performance workload for this training order.

### Published GSM8K colocated 8K curve

Transcribed from the 8K rulebook; numbers are accuracy **percentages** at its published precision. No smoothing or best-checkpoint selection is applied.

| Step | Seed 1 | Seed 2 | Seed 3 |
|---:|---:|---:|---:|
| 0 | 74.53 | 74.00 | 75.82 |
| 20 | 73.62 | 74.83 | 77.26 |
| 40 | 74.68 | 74.00 | 75.74 |
| 60 | 77.86 | 77.41 | 75.28 |
| 80 | 79.30 | 79.23 | 76.35 |
| 100 | 79.76 | 78.17 | 79.45 |
| 120 | 80.74 | 77.18 | 74.83 |
| 140 | 81.12 | 78.01 | 77.71 |
| 160 | 82.26 | 77.41 | 78.09 |
| 180 | 82.18 | 80.44 | 79.98 |
| 200 | 81.65 | 79.98 | 79.61 |
| 220 | 81.96 | 79.45 | 79.76 |
| 240 | 81.88 | 79.08 | 80.97 |
| 250 | 82.26 | 79.61 | 79.83 |

### Published GSM8K disaggregated 2K curve

These are checkpoint-wise aggregates across three seeds. The minimum and maximum need not belong to the same seed at successive checkpoints.

| Step | Min (%) | Mean (%) | Max (%) |
|---:|---:|---:|---:|
| 0 | 70.28 | 70.53 | 70.66 |
| 20 | 71.42 | 72.96 | 74.30 |
| 40 | 76.65 | 76.80 | 77.03 |
| 60 | 78.77 | 79.58 | 80.59 |
| 80 | 80.36 | 81.05 | 81.65 |
| 100 | 80.06 | 81.50 | 82.41 |
| 120 | 80.74 | 81.75 | 82.94 |
| 140 | 80.74 | 81.85 | 83.32 |
| 160 | 81.35 | 82.21 | 83.47 |
| 180 | 81.27 | 82.89 | 83.70 |
| 200 | 81.43 | 81.93 | 82.79 |
| 220 | 82.11 | 82.71 | 83.24 |
| 240 | 81.80 | 82.61 | 84.23 |
| 250 | 82.34 | 82.74 | 83.17 |

The full colocated 2K curve and OMI2 per-seed numerical curves are not reconstructed here from plots or terminal excerpts. Retrieve their archived TensorBoard / eval dumps for a new joint plot.

## 3. What actually differed between the recipes

| Setting | GSM8K colocated 2K / 8K | GSM8K disaggregated 2K | OMI2 `stab_kl0` |
|---|---|---|---|
| Model | Qwen3-0.6B post-trained, thinking on | Same model family/identity | Qwen3-0.6B post-trained, thinking on |
| Resources | 64 GPUs, colocated trainer/sampler, rollout TP=1 | 32 trainer + 32 rollout, TP=1 | 64 GPUs, colocated, TP=1 |
| Batch | 128 prompts × 16 completions = 2,048 | Same | 256 × 8 = 2,048 |
| Update semantics | One optimizer update per complete batch; GRPO advantages | Same; asynchronous data | One optimizer update per complete batch; GRPO advantages |
| Policy objective | REINFORCE-style + detached token TIS, cap 3 | Same | PPO clipping, low 0.2 / high 0.28 |
| Aggregation | Global valid-response-token mean | Same | Same |
| Reference KL / entropy coefficient | 0 / 0 | 0 / 0 | 0 / 0 |
| LR | 2e-6 peak, 10-update warmup, cosine to 0 over 250 | Same | 1e-6 constant, no warmup |
| AdamW | β=(0.9,0.999), ε=1e-8, wd=0, grad clip=1 | Same | Same β/ε/clip; wd=0.01 |
| Precision | FP32 persistent/master weights + Adam, BF16 compute / rollout | Same | Same |
| Sampling | T=1, top_p=1, top_k disabled | Same | Same |
| Prompt / response caps | 512 / 2,048 or 8,192 | 512 / 2,048 | 8,192 / 8,192 |
| Prompt/scorer | Boxed-answer prompt + last balanced box, minimal string normalization | Same | Historical reasoning/answer wrapper + math-verify |
| Format reward | Exclusive correct=1, parseable wrong=0.1, unparseable=0 | Same | Format coefficient 0 |
| Length penalty | 2K: train-only, buffer512, coefficient1; 8K: disabled | Train-only, buffer512, coefficient1 | Buffer1,024, coefficient1 |
| Training budget | 250 updates; 32,000 prompt instances | Same consumed batch count; some generated groups dropped | 300 updates; 76,800 prompt instances |

### Loss and reward details that matter when reusing the reference

For GSM8K, with response mask $m_{it}$ and group-normalized advantage $A_i$:

$$
w_{it}=\operatorname{stopgrad}\left[\min\left(\exp\left(\operatorname{clip}(\log\pi_{\theta,it}-\log\pi_{\mathrm{sampler},it},-20,20)\right),3\right)\right],
\qquad
L=-\frac{\sum_{i,t}m_{it}w_{it}A_i\log\pi_{\theta,it}}{\sum_{i,t}m_{it}}.
$$

Advantages are detached and use group sample standard deviation (`correction=1`) plus `1e-6`. Zero-advantage groups remain in the denominator. The successful path is `bypass_mode=true`, `loss_type=reinforce`: PPO/dual clipping are not active in this objective. OMI2 uses the PPO clipped-ratio objective instead; “GRPO” describes its advantage construction, not a unique loss formula.

The [boxed scorer](../my_scripts_GKE_gsm8k/boxed_math_reward.py) pays **1.0 total** for a correct boxed answer, not 1.1. Its 0.1 reward is the alternative for a parseable but incorrect box. OMI2 has no format bonus, and Gemma's current OMI2 recipe instead adds `accuracy + 0.1 × format`.

For capped responses, the 2K training penalty starts above 1,536 tokens and reaches −1 at 2,048; the OMI2 penalty starts above 7,168 and reaches −1 at 8,192. GSM8K 8K has **no length penalty**. Eval accuracy always reads the correctness flag independently of score.

### Execution provenance and async observations

The colocated GSM/OMI runs used the historical fork and vLLM `0.20.2rc1.dev49+g9b4e83934`; the OMI2 trainer was FSDP1 (`strategy=fsdp`). Use each package's `env/` and model hashes for the exact executed versions. Today's source-document commit is not the historical runtime commit.

The disaggregated reference used verl `ace775e87d8765bcdd114aac734ab71da5367a0f`, FSDP2, Torch `2.13.0+cu130`, vLLM `0.29.0`, Transformers `5.12.1`, and dynamic training batches of 32,768 tokens/GPU. Its [pin](../my_scripts_GKE_gsm8k_disagg/VERL_PIN.txt), [image digest](../my_scripts_GKE_gsm8k_disagg/IMAGE_REF.txt) and [recipe](../my_scripts_GKE_gsm8k_disagg/recipe_gpu_disagg.yaml) are tracked.

Async sync occurs after every update, with one warmup batch. Observed worst token-policy lag was 0 initially and at most 1 thereafter, but responses could span two weight versions. Maximum span 2 occurred in 136 / 138 / 144 of 250 steps; 11 / 8 / 8 prompt groups were dropped for age. The configured `threshold=2/drop` bounds prompt dispatch age, not token staleness. Do not rename this as “always exactly one stale version, zero drops.”

## 4. Performance with explicit measurement boundaries

| Recipe | Steady step median per seed (s) | Launch-to-finish per seed (min) | Second consecutive eval ≥80%, confirming step |
|---|---|---|---|
| GSM8K colocated 2K | 14.15 / 14.06 | 71.0 / 69.9 | 80 / 100 |
| GSM8K colocated 8K | 39.5 / 39.4 / 40.7 | Not transcribed | 140 / not reached / not reached |
| GSM8K disaggregated 2K | 7.16 / 7.34 / 7.22 | 59.4 / 59.2 / 59.3 | 80 / 100 / 100 |

These GSM steady summaries use steps 20–250 excluding eval/checkpoint steps (216 steps/run). Disaggregated p90 is 10.07 / 9.53 / 10.25 seconds. Launch-based totals include startup, eval and checkpoint overhead. The steady median is about 49% lower (a ratio of approximately 1.95), while mean end-to-end duration is about 15.8% lower in the 2K comparison. This is a **whole-stack comparison**: batching and software changed alongside topology. It does not isolate the causal benefit of disaggregation.

The OMI2 historical diagnostic plot reports GPU median **45.2 seconds**, non-eval steps ≥3; the guide notes roughly 2.8 seconds of rollout-dump overhead. This is a different workload and timing window, not the later 20-step TP=2 single-forward benchmark or an apples-to-apples comparison to the table above.

Response cap is not average generated length. GSM8K 8K averaged about 2,303 tokens over training; its last-10-step means were 2,364 / 2,312 / 2,612. The 2K colocated late means were 586 / 572. For future comparisons report arithmetic mean over a declared steady window, median/p90, generated tokens and full launch-to-finish time; async variance makes a median alone insufficient.

## 5. What this suggests for Gemma — and what it does not establish

The current OMI2 Gemma optimization control is **`google/gemma-4-E2B-it`, A3a**, a single-seed 100-step screen, not a frozen three-seed baseline. It reached **36.06% → 40.96% (+4.90 pp)**; A0 reached 35.76% → 39.62%. See [RECIPE_ABLATION_PLAN.md](RECIPE_ABLATION_PLAN.md) for the full curves and paired intervals.

| Factor | Qwen OMI2 `stab_kl0` | Gemma OMI2 A3a |
|---|---|---|
| Loss / batch | PPO clip 0.2/0.28; 256 × 8; one update | Same broad objective and batch |
| Persistent precision | FP32 weights and Adam; BF16 compute | Same precision intent, with Gemma RoPE fixes/guards |
| Sampling | T=1, top_p=1, no top_k | T=0.8, top_p=0.95, top_k=50 |
| LR | 1e-6 constant | 2e-6 constant |
| Reward | Accuracy, no format bonus, length penalty | Accuracy + 0.1 format, no length penalty |
| Response cap | 8,192 | 8,192 |
| Held-out | Internal question-level split, 1,000 | Frozen deduplicated non-overlapping 5,000, different split |
| Observed step-100 gain | ≈+12.1 pp | +4.90 pp |

The Qwen result motivates recipe optimization, but does not make Gemma “invalid.” These held-out populations cannot be used for a question-paired cross-model delta. Gemma's customer hardware gate compares **Gemma TPU vs Gemma GPU under one frozen protocol**, not Gemma vs Qwen or a required match to Qwen's learning gain.

The existing [signal-density comparison](update_precision/fp32_seed1_100steps/signal_density_s100.json) finds mixed-correctness groups around 50–59% for Qwen versus 25.5–27.1% for Gemma A0. This is a useful diagnostic, not causal proof: reward variance can also come from format/length terms even when all answers have the same correctness. The historical double wrapper also existed in the successful Qwen OMI2 recipe, so its presence alone does not explain Gemma's smaller improvement.

For a direct **GSM8K 8K/no-length-penalty Gemma** experiment, retain the historical task, scoring, batch and eval contract; adapt the tokenizer/chat template and required Gemma numerical fixes explicitly. Treat it as a new model baseline. Do not silently carry OMI2's PPO loss, additive formatting reward or 256×8 batch into a purported reuse of the GSM8K recipe.

## 6. Where the evidence lives

Source documents were checked at repository commit `39ca638358c121b83b0f8a1c7646932ca993250a`. This documentation did not download or re-hash external archives.

| Reference | Tracked authority | External handoff root |
|---|---|---|
| GSM8K 8K, plus 2K appendix | [8K rulebook](../my_scripts_GKE_gsm8k/TPU_GPU_RL_Parity_Rulebook_gsm8k_8k.md), [packager](../my_scripts_GKE_gsm8k/package_gsm8k_8k.sh) | `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_8k/` |
| GSM8K disaggregated 2K | [Async rulebook](../my_scripts_GKE_gsm8k_disagg/TPU_GPU_RL_Parity_Rulebook_gsm8k_2k_async1.md), [results README](../my_scripts_GKE_gsm8k_disagg/README.md) | `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/gsm8k_2k_async1/` |
| OMI2 `stab_kl0` | [Tianyu guide](../my_scripts_GKE/Tianyu_TPU_Parity_Guide.md), [launcher/stab preset](../my_scripts_GKE/run_qwen3_0p6b_rl05_parity.sh), [split builder](../my_scripts_GKE/build_eval_data_v3.py) | `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/handoff/stab_kl0/` |

Expected artifacts inside those roots:

- **GSM8K 8K:** `band/gb200_band_gsm8k_8k.png/.json`, `gb200_curves_gsm8k_8k.png`, `gb200_train_gsm8k_8k.png`, `summary.json`, per-seed diagnostics and step-0 variability report.
- **GSM8K async:** `band/gb200_band_gsm8k_2k_async1.png/.json`, `gb200_vs_colocated_2k.png`, `gb200_step_time.png`, `summary.json` and per-seed diagnostics.
- **OMI2:** `band/gb200_band_stab_kl0.json/.png`; `runs/<GPU_EXP>/val_dump/<step>.jsonl`; model identity, ordered prompt manifests, checkpoints and runtime records. Raw training rollout dumps are under `gs://xiaotongyang-bucket/meta-rl/GKE_repro/meta-RL/logs/<GPU_EXP>/rollout_dump/`.
- GSM packages use `runs/seed{1,2,3}/` for TensorBoard/eval/rollout dumps. All packages should be verified against their own `PACKAGE_MANIFEST.sha256` before a new reproduction; today's documentation is not that verification.

Existing tracked OMI2 figures are **historical TPU-vs-GPU comparisons**, not GPU-only curves or Gemma results: [OMI2 held-out comparison](../my_scripts_GKE/download%20%281%29.png), [GSM8K transfer comparison](../my_scripts_GKE/download.png), [training diagnostics](../my_scripts_GKE/download%20%282%29.png). Their old ±1.5 pp envelope is not the customer's newer 1 pp gate.

### Run identifiers

| Recipe | Seed | Experiment |
|---|---:|---|
| GSM8K colocated 2K | 1 | `qwen3_0p6b_base_sf_tis_16n_seed1_16n64g_20260921_1823` |
| GSM8K colocated 2K | 2 | `qwen3_0p6b_base_sf_tis_16n_seed2_16n64g_20260922_0040` |
| GSM8K colocated 8K | 1 | `qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed1_16n64g_20260922_0254` |
| GSM8K colocated 8K | 2 | `qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed2_16n64g_20260922_1735` |
| GSM8K colocated 8K | 3 | `qwen3_0p6b_base_sf_tis_16n_cap8k_nopen_seed3_16n64g_20260922_2105` |
| GSM8K disaggregated 2K | 1 | `disagg_t32_r32_dyn_seed1_20260925_020445` |
| GSM8K disaggregated 2K | 2 | `disagg_t32_r32_dyn_seed2_20260925_042156` |
| GSM8K disaggregated 2K | 3 | `disagg_t32_r32_dyn_seed3_20260925_052306` |
| OMI2 `stab_kl0` | 1 | `qwen3_0p6b_phase0_stab_kl0_seed1_16n64g_20260913_0844` |
| OMI2 `stab_kl0` | 2 | `qwen3_0p6b_phase0_stab_kl0_seed2_16n64g_20260913_1914` |
| OMI2 `stab_kl0` | 3 | `qwen3_0p6b_phase0_stab_kl0_seed3_16n64g_20260913_2333` |

Async experiment IDs are retained from the owner's launch logs; the rulebook separately verifies their endpoint summaries. When expanding this ledger, import raw band/eval data with hashes rather than filling missing checkpoints by interpolation.
